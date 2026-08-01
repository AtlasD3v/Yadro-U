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
            param_group.append({
                'params': group_params, #сохраняем найденные по совпадению имён параметры
                'lr': base_lr * scale_factors.get(map_size, 1.0), #по размерности карт признаков (map_size, 1.0) находим scale_factors для текущих параметров и умножаем на base_lr
                'weight_decay': 1e-3
            })


    #заново перебираем все параметры модели, проверяя по имени, добавлены ли они уже в группу параметров или нет,
    #если нет, добавляем сам параметр в remaining_params (это будут параметры из neck и head)
    remaining_params = [
        param for name, param in named_parameters.items()
        if name not in assigned_param_names
    ]

    if remaining_params:
        param_group.append({
            'params': remaining_params,
            'lr': base_lr, #здесь логика обучения такова, что neck и head Обучаются с base_lr
            weight_decay': 1e-2
        })

    return param_group