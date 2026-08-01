import torch

def build_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps, is_warmup = False, pct_start=0.05, max_lr=0.001):
    """
    total_steps, warmup_steps - считаются в эпохах
    """
     # тренировочный планировщик (линейный), который с начала обучения начинает постепенно приводить lr 
     # к тем, что были заданы в качестве гиперпараметров. Это делается, чтобы избежать взрывов градиентов в начале из-за высокого lr
    scheduler = None
    max_lrs = [group['lr'] for group in optimizer.param_groups] # берём УЖЕ настроенные per-group lr из оптимизатора
    if is_warmup:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs, 
            total_steps=total_steps,
            pct_start=pct_start,       # эквивалент твоего warmup_steps
            anneal_strategy='cos',     # косинусное затухание
            cycle_momentum=True        # та самая магия с моментом!
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max= total_steps,
            eta_min=1e-7
        )

    return scheduler