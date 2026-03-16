import os
import torch
import torch.nn as nn
import Models

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def replace_all_conv_bn_pairs(module):
    previous_path = None
    last_conv = None
    for current_path, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            previous_path = current_path
            last_conv = child
        elif (
            isinstance(child, nn.BatchNorm2d)
            and last_conv is not None
        ):
            module.set_submodule(
                previous_path,
                FusedConvBnQuantLayer(
                    original_conv_layer=last_conv,
                    original_bn_layer=child
                ),
                # Check that a child named previous_path exists.
                strict=True
            )
            # Replace batchnorm with a do-nothing layer.
            module.set_submodule(
                current_path,
                nn.Identity(),
                # Check that a child named previous_path exists.
                strict=True
            )
            # Mark the conv layer as consumed.
            previous_path = None
            last_conv = None
        else:
            replace_all_conv_bn_pairs(child)


class round_with_redefined_gradient(
    torch.autograd.Function
):
    @staticmethod
    def forward(x):
        return torch.round(x)
    
    # If we do not define this,
    # forward() gets passed (self, x)
    # instead of just x,
    # despite the staticmethod decorator.
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        # Without the ctx argument,
        # backward() complains about getting
        # two arguments while expecting 1.
        return grad_output


def quantize(
    x, scale,
    x_min=-127, x_max=127
):
    x = round_with_redefined_gradient.apply(x / scale)
    x = torch.clamp(x, x_min, x_max)
    x = x * scale
    return x


def get_fused_weights_from_conv_and_nb_parameters(
        conv_weight, conv_bias,
        bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps,
    ):
        bn_scale = bn_weight / torch.sqrt(bn_running_var + bn_eps)
        bn_shift = bn_bias
        
        fused_weight = conv_weight * bn_scale.view(-1, 1, 1, 1)

        if conv_bias is not None:
            fused_bias = (conv_bias - bn_running_mean) * bn_scale + bn_shift
        else:
            fused_bias = -bn_running_mean * bn_scale + bn_shift
        
        return fused_weight, fused_bias


class FusedConvBnQuantLayer(nn.Module):
    def __init__(
        self, original_conv_layer, original_bn_layer,
        quantization_scale=None
    ):
        super().__init__()
        self.original_conv_layer = original_conv_layer
        self.original_bn_layer = original_bn_layer
        self.quantization_scale = nn.Parameter(quantization_scale)

    def get_fused_weights(self):
        return get_fused_weights_from_conv_and_nb_parameters(
            conv_weight=self.original_conv_layer.weight,
            conv_bias=self.original_conv_layer.bias,
            bn_weight=self.original_bn_layer.weight,
            bn_bias=self.original_bn_layer.bias,
            bn_running_mean=self.original_bn_layer.running_mean,
            bn_running_var=self.original_bn_layer.running_var,
            bn_eps=self.original_bn_layer.eps
        )
    
    def forward(self, x): #, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps, scale, stride, padding, dilation, groups):
        fused_weight, fused_bias = self.get_fused_weights()
        return F.conv2d(
            input=x,
            weight=quantize(fused_weight, scale=self.quantization_scale),
            bias=quantize(fused_bias, scale=self.quantization_scale),
            stride=self.original_conv_layer.stride,
            padding=self.original_conv_layer.padding,
            dilation=self.original_conv_layer.dilation,
            groups=self.original_conv_layer.groups
        )
    
    def initialize_scale(self):
        # Though not necessary,
        # let's wrap it in no_grad()
        # just in the hope for speedup.
        with torch.no_grad():
            fused_weight, fused_bias = self.get_fused_weights()
            w_abs = torch.abs(fused_weight).flatten()
            scale_val = torch.quantile(w_abs, 0.995) / 127
            # Reaching the scale through its data attribute
            # should be enough to side-step the autograd system.
            self.quantization_scale.data = torch.clamp(scale_val, 1e-6)


def initialize_scale_if_a_fused_layer(module):
    if isinstance(module, FusedConvBnQuantLayer):
        module.initialize_scale()
        print(f"scale={module.quantization_scale.item():.6f} (from fused weight)")

        
class FuseOnTheFlyQAT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.original_model = model
        replace_all_conv_bn_pairs(self.original_model)
        self.original_model.apply(
            initialize_scale_if_a_fused_layer
        )
        
    def forward(self, x):
        return self.original_model.forward(x)
    
    def get_scales_stats(self):
        return {
            f'{name}.scale'
            : module.quantization_scale.item()
            for name, module in self.named_modules()
            if isinstance(module, FusedConvBnQuantLayer)
        }
        

def validate_ste(model_qat, val_loader, device):
    model_qat.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model_qat(imgs) 
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def train_fuse_ste_qat(model_qat, train_loader, val_loader, device, epochs=10):
    print("\n" + "="*60)
    print("STEP 1: Initializing QAT model")
    print("="*60)
    
    model_qat = model_qat.to(device)
    
    # print(f"\nParameters: {len(list(model_qat.original_model.parameters()))} non-scale, {len(model_qat.scales)} scale parameters")
    
    optimizer = torch.optim.AdamW(model_qat.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print("\nSTEP 2: Evaluating PTQ baseline...")
    ptq_acc = validate_ste(model_qat, val_loader, device)
    print(f"PTQ Baseline: {ptq_acc:.2f}%")
    print(f"Initial scales: {model_qat.get_scales_stats()}")
    
    best_val = ptq_acc
    
    for epoch in range(epochs):
        model_qat.train()
        train_correct, train_total = 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model_qat(imgs)
            loss = criterion(outputs, labels)
            loss.backward()

            if batch_idx % 10 == 0:
                print(f"\n--- Batch {batch_idx} ---")
                has_grads = False
                for name, layer in model_qat.named_modules():
                    if not isinstance(layer, FusedConvBnQuantLayer):
                        continue
                    s = layer.quantization_scale
                    if s.grad is not None:
                        has_grads = True
                        print(f"    {name}.scale: value={s.item():.6f}, grad={s.grad.item():.6f}")
                    else:
                        print(f"    {name}.scale: grad is None")
                
                if not has_grads:
                    print("NO GRADIENTS FOR ANY SCALE!")

                for name, param in model_qat.original_model.named_parameters():
                    if 'conv' in name and param.grad is not None:
                        print(f"  ✓ {name} has grad, norm={param.grad.norm().item():.6f}")
                        break
            
            torch.nn.utils.clip_grad_norm_(model_qat.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                train_acc = 100. * train_correct / max(train_total, 1)
                scales_str = ", ".join([f"{v:.4f}" for v in list(model_qat.get_scales_stats().values())[:5]])
                print(f'E[{epoch:2d}]B[{batch_idx:3d}]: Loss={loss.item():.3f}, '
                      f'Train={train_acc:.1f}%, Scales=[{scales_str}]')
        
        scheduler.step()
        val_acc = validate_ste(model_qat, val_loader, device)
        train_acc = 100. * train_correct / max(train_total, 1)
        
        print(f'\n=== Epoch {epoch} === Train:{train_acc:.2f}% | Val:{val_acc:.2f}%')
        print(f"Scales: {model_qat.get_scales_stats()}")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                'model_state_dict': model_qat.original_model.state_dict(),
                'scales': model_qat.get_scales_stats(),
                'best_val_acc': best_val
            }, 'best_fuse_ste_qat.pth')
            print(f"NEW BEST: {best_val:.2f}%")
    
    return model_qat

