import torch

from src.models import retinanet
from src.utils import get_param_group_to_optim
from src.training import training_engine
from src.training.optim import gen_optim

import os

# Динамически находим корень проекта:
# __file__ -> src/training/training_faster_rcnn.py
# .dirname().dirname().dirname() -> корень проекта (где лежит папка checkpoints)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CHECKPOINTS_PATH = os.path.join(BASE_DIR, "checkpoints")

def main():
    base_lr = 0.1e-3 #гиперпараметр скорости обучения конкретной модели
    num_classes = 7

    model, optim_param_groups = init_model(base_lr, num_classes)
    optimizer = gen_optim.build_optimizer(optim_param_groups, base_lr)

    model_name = 'retinanet'
    num_epochs = 100
    batch_size = 16
    checkpoint_path = DEFAULT_CHECKPOINTS_PATH
    need_scheduler = True
    is_warmup = True
    need_scaler = True
    max_grad_norm = None
    dataset_root = "D:\\practicum\\data\\raw\\final_dataset" #пока что None, когда загружу датасет, тут укажу путь на него

    learned_model = training_engine.fit(
        model= model, 
        optimizer= optimizer, 
        model_name= model_name, 
        num_epochs= num_epochs, 
        batch_size= batch_size, 
        lr = base_lr, 
        checkpoint_path= checkpoint_path,
        need_scheduler= need_scheduler,
        is_warmup=is_warmup,
        need_scaler= need_scaler,
        max_grad_norm = max_grad_norm,
        dataset_root = dataset_root
    )

    return learned_model


def init_model(base_lr: float, num_classes: int):
    model = retinanet.build_model(num_classes= num_classes) #строим модель

    modules_dict = {m_name: module for m_name, module in model.named_children()} #получаем словарь верхнеуровневых модулей модели
    

    print(modules_dict.keys())

    # init_weights_kaiming(model) #Инициализируем веса для не предобученных модулей (в нашем случае neck, head)

    backbone_block_name = input("Введите имя блока бэкбона: ")
    backbone_module = modules_dict[backbone_block_name]

    param_group_to_optimizer = get_param_group_to_optim.param_groups_to_optim(model, backbone_module, backbone_block_name, base_lr)

    return model, param_group_to_optimizer



def init_weights_kaiming(model: torch.nn.Module):
    for m_name, module in model.named_children():
        if m_name != "transform" and m_name != "backbone":

            if isinstance(module, (torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.zeros_(module.bias)



if __name__ == '__main__':
    main()