import os
import torch
import torch.nn as nn
import Models

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def find_all_conv_bn_pairs(module, prefix=''):
    pairs = []
    for name, child in module.named_children():
        current_path = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Conv2d):
            pairs.append((current_path, child, None))
        elif isinstance(child, nn.BatchNorm2d) and len(pairs) > 0 and pairs[-1][2] is None:
            conv_path, conv, _ = pairs[-1]
            pairs[-1] = (conv_path, conv, child)
        else:
            pairs.extend(find_all_conv_bn_pairs(child, current_path))
    
    return [(path, conv, bn) for path, conv, bn in pairs if bn is not None]

class FusedConvBnQuantFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps, scale, stride, padding, dilation, groups):
        ctx.save_for_backward(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, scale)
        ctx.bn_eps = bn_eps
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        bn_scale = bn_weight / torch.sqrt(bn_running_var + bn_eps)
        bn_shift = bn_bias
        
        fused_weight = conv_weight * bn_scale.view(-1, 1, 1, 1)

        q_weight = torch.round(fused_weight / scale)
        q_weight = torch.clamp(q_weight, -127, 127)
        q_weight = q_weight * scale
        

        if conv_bias is not None:
            fused_bias = (conv_bias - bn_running_mean) * bn_scale + bn_shift
        else:
            fused_bias = -bn_running_mean * bn_scale + bn_shift
        
        q_bias = torch.round(fused_bias / scale)
        q_bias = torch.clamp(q_bias, -127, 127)
        q_bias = q_bias * scale
        
        ctx.fused_weight = fused_weight
        ctx.fused_bias = fused_bias
        ctx.q_weight = q_weight
        ctx.q_bias = q_bias
        ctx.bn_scale = bn_scale

        out = F.conv2d(x, q_weight, q_bias, stride, padding, dilation, groups)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        print("🔙 BACKWARD called")
        
        x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, scale = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        bn_eps = ctx.bn_eps
        
        # Получаем сохраненные тензоры
        fused_weight = ctx.fused_weight
        fused_bias = ctx.fused_bias
        q_weight = ctx.q_weight
        q_bias = ctx.q_bias
        bn_scale = ctx.bn_scale
        
        print(f"   grad_output shape: {grad_output.shape}")
        print(f"   scale: {scale.item():.6f}")
        
        # Градиент для x
        grad_x = F.conv2d(grad_output, q_weight, None, stride, padding, dilation, groups)
        
        # Упрощенные градиенты для scale
        grad_scale = grad_output.mean() * 0.001
        print(f"   grad_scale: {grad_scale.item():.6f}")
        
        # Градиенты для conv_weight
        grad_conv_weight = F.conv2d(x.transpose(0,1), grad_output.transpose(0,1), None, stride, padding, dilation, groups).transpose(0,1)
        grad_conv_weight = grad_conv_weight * bn_scale.view(-1, 1, 1, 1) * 0.01
        
        # Градиенты для conv_bias
        grad_conv_bias = None
        if conv_bias is not None:
            grad_conv_bias = grad_output.sum(dim=(0,2,3)) * bn_scale * 0.01
        
        # Упрощенные градиенты для BN
        grad_bn_weight = grad_output.mean() * 0.001
        grad_bn_bias = grad_output.mean() * 0.001
        
        # Градиенты для статистик BN (не нужны)
        grad_bn_running_mean = None
        grad_bn_running_var = None
        
        return (grad_x, grad_conv_weight, grad_conv_bias, grad_bn_weight, grad_bn_bias, 
                grad_bn_running_mean, grad_bn_running_var, None, grad_scale, 
                None, None, None, None)
    
class FuseOnTheFlyQAT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.original_model = model  
        self.scales = nn.ParameterList()
        self._initialized = False
        self.conv_bn_pairs = []
        
    def initialize(self):
        if self._initialized:
            return
        
        device = next(self.original_model.parameters()).device
        
        self.conv_bn_pairs = find_all_conv_bn_pairs(self.original_model)
        
        print(f"Found {len(self.conv_bn_pairs)} Conv+BN pairs:")
        for i, (path, conv, bn) in enumerate(self.conv_bn_pairs):
            with torch.no_grad():
                bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                fused_weight = conv.weight * bn_scale.view(-1, 1, 1, 1)
                w_abs = torch.abs(fused_weight).flatten()
                scale_val = torch.quantile(w_abs, 0.995) / 127  
            
            scale_param = nn.Parameter(torch.clamp(scale_val, 1e-6).to(device))
            self.scales.append(scale_param)
            print(f"  [{i}] {path}: scale={scale_val.item():.6f} (from fused weight)")
        
        print(f"Initialized with {len(self.scales)} scale parameters")
        self._initialized = True
    
    def forward(self, x):
        self.initialize()
        
        def forward_with_fuse(module, x_input):
            if isinstance(module, nn.Sequential):
                for child in module:
                    x_input = forward_with_fuse(child, x_input)
                return x_input
            elif isinstance(module, nn.Conv2d):
                for idx, (path, conv, bn) in enumerate(self.conv_bn_pairs):
                    if conv is module:
                        scale = self.scales[idx]

                        out = FusedConvBnQuantFunction.apply(
                            x_input,
                            conv.weight,
                            conv.bias,
                            bn.weight,
                            bn.bias,
                            bn.running_mean,
                            bn.running_var,
                            bn.eps,
                            scale,
                            conv.stride,
                            conv.padding,
                            conv.dilation,
                            conv.groups
                        )
                        return out

                return module(x_input)
            else:
                try:
                    return module(x_input)
                except Exception as e:
                    return x_input
        
        return forward_with_fuse(self.original_model, x)
    
    def parameters(self):
        return list(self.original_model.parameters()) + list(self.scales.parameters())
    
    def get_scales_stats(self):
        return {f'scale_{i}': s.item() for i, s in enumerate(self.scales)}

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
    model_qat.initialize()
    
    print(f"\nParameters: {len(list(model_qat.original_model.parameters()))} non-scale, {len(model_qat.scales)} scale parameters")
    
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
                for i, s in enumerate(model_qat.scales[:3]):
                    if s.grad is not None:
                        has_grads = True
                        print(f"    scale_{i}: value={s.item():.6f}, grad={s.grad.item():.6f}")
                    else:
                        print(f"    scale_{i}: grad is None")
                
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
                'scales': {f'scale_{i}': s.detach().cpu() for i, s in enumerate(model_qat.scales)},
                'best_val_acc': best_val
            }, 'best_fuse_ste_qat.pth')
            print(f"NEW BEST: {best_val:.2f}%")
    
    return model_qat

