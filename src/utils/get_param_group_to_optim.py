import torch
from src.utils import stride_hook, get_params_from_layers


def param_groups_to_optim(model: torch.nn.Module, backbone_block: torch.nn.Module, backbone_name: str, base_lr: float):
    param_group = [] #группа параметров - является массивом, содержащим словари. Формат словарей {'param': параметры, 'lr': наш рейт, 'weight_decay': наше значение}

    blocks_in_backbone = {b_name: b_block for b_name, b_block in backbone_block.named_children()} #собираем дочерние модули внутр бэкбона для определения оглубления
    print(blocks_in_backbone.keys())

    backbone_body_name = input("Если внутри бэкбона нашлись другие дочерние блоки, которые не являются прямыми блоками бэкбона, введите имя блока, который содержит именно слои бэкбона (если всё хорошо, то оставьте поле пустым): ")
    backbone_main_module = blocks_in_backbone[backbone_body_name] #выбираем по какому блоку бэкбона (если есть углубление) будем проходиться

    
    backbone_full_name = f"{backbone_name}.{backbone_body_name}" #составляем полное имя выбранного блока (как во всей модели)
    print(backbone_full_name)

    layers_dict = stride_hook.main_func(model, backbone_main_module, backbone_full_name, 2) #получаем словарь слоёв бэкбона, которые разделены по размерности карт признаков, которые получаются на их выходе 
    print(layers_dict)

    map_rezolutions = [key for key, _ in layers_dict.items()] #создаём массив scale'ов карт признаков для того, чтобы составить словарь scale_factors, который будет определять, для каких параметров как уменьшать base_lr (base_lr используется для neck,head)
    lrs_scales = [] #массив, хранящий во сколько раз медленее будут учиться параметры с map_rezolutions[i], чем base_lr
    for i in range(len(map_rezolutions)):
        scale = float(input(f"Введите значение, во сколько раз параметры с размером карт признаков {map_rezolutions[i]} должны обучаться медленее, чем base_lr: "))
        lrs_scales.append(scale)
    scale_factors = {key: (1.0 / lr) for key, lr in zip(map_rezolutions, lrs_scales)}

    param_group = get_params_from_layers.get_parameters_from_layers(model, layers_dict, base_lr, scale_factors) #получаем группу параметров с распределёнными lr для оптимизатора
    return param_group
