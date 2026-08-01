import torch


#Функция, которая исходя из названий слоёв (в layers_dict) ищет их обуч. параметры и формирует группы параметров для оптимизатора
def get_parameters_from_layers(model: torch.nn.Module, layers_dict: dict[str, list], base_lr: float, scale_factors: dict[str, int]):
    
    """
    --layers_dict - словарь в формате {'размерность карт признаков': list('название блока/слоя')}
    -- base_lr - базовое значение лёрнинг рейта

    --scale_factors - словарь формата {'размер карты признаков': число, на которое умножается base_lr}
        чем больше scale_factors.key, то есть чем больше размерность карты признаков, тем меньше scale_factors для этого элемента,
        чтобы не переобучать/ломать нахождение низкоуровневых признаков 
    """
    named_parameters = {name: param for name, param in model.named_parameters()} #находим и сохраняем все параметры модели с их именами

    param_group = [] #итоговый массив с группами параметров
    assigned_param_names = set() #сохранение названий уже сохранённых параметров (будет использоваться для того, чтобы остальные параметры не из backbone тоже попали в param_groups)

    #теперь пишем цикл, который будет брать из layers_dict название блока\слоя и в named_parameters искать его параметры
    for map_size, module_names in layers_dict.items():#цикл, который итерирует массивы названия блоков\слоёв, разделённых по map_size

        group_params = []#массив, который сохраняет подходящие параметры для текущего map_size, module_names

        for param_name, param in named_parameters.items(): #цикл, который итерирует все названия параметров сети

            for module_name in module_names:#цикл, который итерирует названия слоёв\блоков внутри массива module_names

                if param_name.startswith(f'{module_name}.') or param_name == module_name:
                    group_params.append(param)
                    assigned_param_names.add(param_name)#сохраняем название уже добавленного в группу параметров параметра

        if group_params:
            #разделяем параметры группы на "decay" (ndim>1, обычно веса свёрток/Linear) и
            #"no_decay" (ndim<=1, всегда bias и gamma/beta у BatchNorm/GroupNorm) —
            #к no_decay параметрам weight_decay применять не нужно, иначе мы "стягиваем к нулю"
            #сам механизм нормализации (gamma) и bias, для которых регуляризация обычно вредна
            decay_params = [p for p in group_params if p.ndim > 1]
            no_decay_params = [p for p in group_params if p.ndim <= 1]

            group_lr = base_lr * scale_factors.get(map_size, 1.0) #по размерности карт признаков (map_size, 1.0) находим scale_factors для текущих параметров и умножаем на base_lr

            if decay_params:
                param_group.append({
                    'params': decay_params, #сохраняем найденные по совпадению имён параметры (только весовые, ndim>1)
                    'lr': group_lr,
                    'weight_decay': 1e-3
                })
            if no_decay_params:
                param_group.append({
                    'params': no_decay_params, #bias и normalization-параметры этой же группы, БЕЗ weight_decay
                    'lr': group_lr,
                    'weight_decay': 0.0
                })


    #заново перебираем все параметры модели, проверяя по имени, добавлены ли они уже в группу параметров или нет,
    #если нет, добавляем сам параметр в remaining_params (это будут параметры из neck и head)
    remaining_params = [
        param for name, param in named_parameters.items()
        if name not in assigned_param_names
    ]

    if remaining_params:
        #та же логика decay/no_decay применяется и к параметрам neck+head, по той же причине
        remaining_decay = [p for p in remaining_params if p.ndim > 1]
        remaining_no_decay = [p for p in remaining_params if p.ndim <= 1]

        if remaining_decay:
            param_group.append({
                'params': remaining_decay,
                'lr': base_lr, #здесь логика обучения такова, что neck и head Обучаются с base_lr
                'weight_decay': 1e-2
            })
        if remaining_no_decay:
            param_group.append({
                'params': remaining_no_decay,
                'lr': base_lr,
                'weight_decay': 0.0
            })

    return param_group