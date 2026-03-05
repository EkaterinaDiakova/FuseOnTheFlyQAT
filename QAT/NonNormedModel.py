import os
import torch
import torch.nn as nn
import Models

def scale_conv_weights_by_scalar_thresh(state_dict):
    """
    Умножает ВСЕ conv.weight на скаляр из предыдущего .thresh (размер [1])
    """
    new_state_dict = state_dict.copy()
    keys = list(state_dict.keys())

    processed = 0
    i = 0
    while i < len(keys) - 1:
        current_key = keys[i]
        next_key = keys[i + 1]

        # thresh[1] -> следующий .weight
        if (current_key.endswith('.thresh') and
            next_key.endswith('.weight')):

            thresh_scalar = state_dict[current_key][0].item()  # извлекаем скаляр
            conv_weight = new_state_dict[next_key]

            new_state_dict[next_key] = conv_weight * thresh_scalar  # scalar * tensor
            processed += 1
            print(f"{next_key} *= {thresh_scalar:.3f}")

        i += 1

    print(f"\processed conv: {processed}")
    return new_state_dict

def add_full_shortcut_scaling(model, old_state_dict):

    shortcut_scales = {

        'conv2_x.0': old_state_dict['conv1.2.thresh'][0].item(),        
        'conv2_x.1': old_state_dict['conv2_x.0.act.thresh'][0].item(),  
        'conv2_x.2': old_state_dict['conv2_x.1.act.thresh'][0].item(),  

        'conv3_x.0': old_state_dict['conv2_x.2.act.thresh'][0].item(),  
        'conv3_x.1': old_state_dict['conv3_x.0.act.thresh'][0].item(),  
        'conv3_x.2': old_state_dict['conv3_x.1.act.thresh'][0].item(),  

        'conv4_x.0': old_state_dict['conv3_x.2.act.thresh'][0].item(),  
        'conv4_x.1': old_state_dict['conv4_x.0.act.thresh'][0].item(),  
        'conv4_x.2': old_state_dict['conv4_x.1.act.thresh'][0].item()   
    }

    print("\nВСЕ Shortcut масштабы:")
    for block_name, scale in shortcut_scales.items():
        print(f"{block_name} shortcut input *= {scale:.3f}")

    for i, block_name in enumerate(['conv2_x', 'conv3_x', 'conv4_x']):
        for j in range(3):  # 3 блока в каждом
            block = getattr(model, block_name)[j]
            scale = shortcut_scales[f'{block_name}.{j}']

            def make_patch(block, scale):
                def new_forward(self, x):
                    if len(block.shortcut) == 0:  
                        identity = x * scale
                    else:  
                        identity = block.shortcut(x * scale)

                    out = block.residual_function(x)
                    out += identity
                    return block.act(out)
                return new_forward.__get__(block)

            block.forward = make_patch(block, scale)

    return model

def full_conversion(old_state_dict, args):

    model = Models.modelpool(args['model'], args['dataset'])

    new_state_dict = scale_conv_weights_by_scalar_thresh(old_state_dict)
    model.load_state_dict(new_state_dict)

    model = add_full_shortcut_scaling(model, old_state_dict)

    return model

class NormalizedResNet4Cifar(Models.ResNet4Cifar):
    """
    Класс-обертка, который наследует все методы оригинальной модели
    и добавляет нормализацию
    """
    def __init__(self, original_model):
        # Копируем все атрибуты
        self.__dict__.update(original_model.__dict__)

        # Находим thresh
        self.thresh_value = None
        for name, param in self.named_parameters():
            if 'avg_pool_act.thresh' in name:
                self.thresh_value = param.item()
                print(f"Thresh: {self.thresh_value}")
                break

        if self.thresh_value is None:
            self.thresh_value = 1.0

        self.original_forward = self.forward

        self.forward = self.normalized_forward

    def normalized_forward(self, x):
        T = self.T

        if T > 0:
            x.unsqueeze_(1)
            x = x.repeat(T, 1, 1, 1, 1)
            x = self.merge(x)

        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)

        output = output * self.thresh_value

        output = self.avg_pool(output)
        output = self.avg_pool_act(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.fc_act(output)

        if T > 0:
            output = self.expand(output)

        return output

def get_norm_model(args):
  model_dir = f"{args['dataset']}-checkpoints"
  old_state_dict = torch.load(os.path.join(model_dir, args['identifier'] + '.pth'), map_location=torch.device('cpu'))
    
  model = full_conversion(old_state_dict, args)
  model_norm_pooling = NormalizedResNet4Cifar(model)

  return model_norm_pooling

