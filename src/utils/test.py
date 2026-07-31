# import torch
# import torchvision

# shapes = {}

# input_H = None
# input_W = None

# dummy_input = torch.randn((1, 3, 640, 640))
# input_size = dummy_input.shape[-2:]
# input_H = input_size[0]
# input_W = input_size[1]

# print(type(input_H), input_W)

# import torch

# tensors = [torch.randn(1) for _ in range(3)]
# targets = {"a": tensors[0], "b": tensors[1], "c": tensors[2]}

# d = [{} for t in targets]

# print(targets)

import torch
import matplotlib.pyplot as plt

# Модель тебе не нужна для этой проверки — только оптимизатор с реальными param_groups
dummy_params = [torch.nn.Parameter(torch.randn(1)) for _ in range(3)]
optimizer = torch.optim.AdamW([
    {'params': [dummy_params[0]], 'lr': 1e-5},   # имитация backbone
    {'params': [dummy_params[1]], 'lr': 1e-4},   # имитация neck
    {'params': [dummy_params[2]], 'lr': 1e-3},   # имитация head
])

total_steps = 10000
warmup_steps = 500

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

lr_history = {0: [], 1: [], 2: []}   # по одному списку на каждую param_group
for step in range(total_steps):
    for i, group in enumerate(optimizer.param_groups):
        lr_history[i].append(group['lr'])
    optimizer.step()      # формально пустой шаг — dummy_params не участвуют в реальном градиенте,
                            # но optimizer.step() должен быть вызван хотя бы формально до scheduler.step()
                            # в некоторых версиях PyTorch (предупреждение LR scheduler order) — на практике
                            # для ЭТОЙ диагностики можно и пропустить, если версия PyTorch не ругается
    scheduler.step()

plt.plot(lr_history[0], label='backbone (base=1e-5)')
plt.plot(lr_history[1], label='neck (base=1e-4)')
plt.plot(lr_history[2], label='head (base=1e-3)')
plt.axvline(x=warmup_steps, color='gray', linestyle='--', label='конец warmup')
plt.yscale('log')   # ЛОГАРИФМИЧЕСКАЯ шкала обязательна — иначе кривая backbone (1e-5) будет
                      # неразличима на фоне head (1e-3), они отличаются на 2 порядка
plt.legend()
plt.xlabel('step')
plt.ylabel('lr')
plt.show()