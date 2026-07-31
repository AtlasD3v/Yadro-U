import torch

def build_optimizer(param_groups: list, base_lr: float):
    optim = torch.optim.AdamW(
        param_groups,
        lr=base_lr # lr здесь — дефолт для групп, где lr явно не указан (если вдруг так получится, но не должно,так как param_groups охватывает все параметры)
    )

    return optim