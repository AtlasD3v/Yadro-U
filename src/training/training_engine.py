import torch
from torch.utils.data import DataLoader

from src.dataset.data_setup import MyCustomDataset
from src.dataset.collate import collate_fn_torchvision
from src.training.scheduler import gen_sched
from src.training.scaler import gen_scaler
from src.eval.evaluate import validate_one_epoch


import time
from typing import List
import os


def build_loader(batch_size = 32, num_workers = 2, dataset_root = None):
    custom_dataset_train = MyCustomDataset(model_type="torchvision", split="train", dataset_root= dataset_root)
    custom_dataset_val = MyCustomDataset(model_type="torchvision", split="val", dataset_root= dataset_root)

    loader_train = DataLoader(
        dataset = custom_dataset_train,
        batch_size = batch_size, # Если используешь 2 x GPU, батч можно увеличить (например, до 32 или 64)
        shuffle = True,
        num_workers = num_workers,
        drop_last= True,
        collate_fn = collate_fn_torchvision,
        pin_memory= True, # <--- Ускоряет передачу из оперативной памяти (RAM) в видеопамять (VRAM)
        persistent_workers=True # <--- Не пересоздает воркеры каждую эпоху
    )

    loader_val = DataLoader(
        dataset = custom_dataset_val,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        collate_fn = collate_fn_torchvision,
        pin_memory= True
    )


    return loader_train, loader_val

def train_one_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, imgs: List[torch.Tensor], targets: List[dict[str, torch.Tensor]], device: torch.device, max_grad_norm=None, scaler: torch.amp.GradScaler = None, scheduler = None):
    #переносим обуч.данные на device
    imgs = [img.to(device=device, non_blocking=True) for img in imgs]
    targets = [{k: v.to(device, non_blocking= True) for k,v in target.items()} for target in targets]

    optimizer.zero_grad(set_to_none= True) # обнуляем градиенты с прошлого шага

    with torch.autocast(device_type = device.type, enabled= scaler is not None): #тут мы выполняем Часть операций forward - в float16 (если scaler is not None)
        loss_dict = model(imgs, targets) #делаем предсказания, получаем потери по батчу
        losses = sum(loss for loss in loss_dict.values()) #суммируем значения потерь по батчу

    if scaler is not None:
        scaler.scale(losses).backward() #делаем обратное распространение, масштабируем (умножаем) loss ПЕРЕД backward — избегаем underflow градиентов

        if max_grad_norm is not None:
            #если max_grad_norm != None, то мы должны градиенты внутри оптимизатора сначала unscale, а потом уже проводить Gradient_clipping_norm
            #иначе мы применим clip_grad_norm_ к "завышенным" (масштабированным) градиентам, а порог max_grad_norm окажется бессмысленным
            scaler.unscale_(optimizer) #восстанавливаем мастшаб градиента
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
        scaler.step(optimizer) #делаем шаг оптимизатора, вернее scaler определяет, стоит ли делать шаг оптимизатора (внутри себя проверяет, есть ли inf/nan)
        scaler.update() #обновляем скейлер (мы просто вызываем функцию обновления, а он уже сам внутри меняет свои параметры)

    else: #если скейлер не был передан
        losses.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    if scheduler is not None:
        scheduler.step() #делаем шаг планировщика тут, потому что у нас есть warmup, следователь планировщик становится per-step

    return losses.item() #возвращаем сумму потерь по батчу


def train_one_epoch(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device:torch.device, num_epoch, print_freq=50, scaler: torch.amp.GradScaler = None, scheduler=None, scheduler_type = None, max_grad_norm = None):
    model.train() #переводим модель в режим train перед началом эпохи

    total_loss = 0.0 #размер потерь по всей эпохе
    loss_components = {}

    start_time = time.time() #время начала эпохи

    lr_scheduler_step = None
    sched_per_epoch = scheduler is not None #если планировщик есть, автоматически присваиваем ему статус "per-epoch"

    if scheduler_type == "per-step": #если передано, что тип планировщика per-step, то sched_per_epoch = False, а lr_scheduler_step = scheduler (lr_scheduler_step передаётся в train_one_step)
        lr_scheduler_step = scheduler
        sched_per_epoch = False

    for step, (imgs, targets) in enumerate(data_loader):
        loss = train_one_step(model = model, optimizer=optimizer, imgs=imgs, targets=targets, device=device, scaler= scaler, scheduler= lr_scheduler_step, max_grad_norm=max_grad_norm) #получаем суммарный лосс по текущему батчу
        total_loss += loss

        if step % print_freq == 0:
            print(f"  Эпоха: {num_epoch} | шаг: {step}/{len(data_loader)} | loss={loss}")

    if sched_per_epoch:
        scheduler.step()

    elapsed = time.time() - start_time #время, затраченное на эпоху

    avg_loss = total_loss / len(data_loader)

    print(f"Epoch {num_epoch} завершена за {elapsed:.1f}с, средний loss: {avg_loss:.4f}")

    return avg_loss


def safe_checkpoint(model, optim, epoch, checkpoint_dir, model_name, is_best):
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
    }

    filename = f"{model_name}_epoch_{epoch}.pth"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)
        print(f"  [Сохранён лучший чекпоинт] {best_path}")

    return path




def fit(model:torch.nn.Module, optimizer: torch.optim.Optimizer, model_name, num_epochs, batch_size, checkpoint_path = "D:\\Yadro-U\\checkpoints", device = None, need_scheduler = False, is_warmup = False, need_scaler = False, max_grad_norm = None, dataset_root = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на устройстве: {device}")


    model.to(device)

    t_loader, val_loader = build_loader(batch_size=batch_size, num_workers=1, dataset_root=dataset_root)

    scheduler = None
    scheduler_type = None
    scaler = None
    if need_scheduler: #инициализируем планировшик, если передан параметр, что он нужен
        if is_warmup:
            scheduler_type = "per-step"
        else:
            scheduler_type = "per-epoch"

        total_steps = len(t_loader) * num_epochs
        pct_start = 0.01 # 1 от всех шагов

        scheduler = gen_sched.build_lr_scheduler(optimizer, total_steps, is_warmup=is_warmup, pct_start=pct_start) #планировщик изменения весов

    if need_scaler: #инициализируем скейлер, если передан параметр, что он нужен
        scaler = gen_scaler.build_scaler(device.type)

    best_map = float("-inf")
    best_loss = float("inf")

    print("-----------------ПЕРЕХОДИМ К ЭПОХАМ ОБУЧЕНИЯ-----------------------")
    for epoch in range(num_epochs):
        epoch_avg_loss = train_one_epoch(model=model, data_loader=t_loader, optimizer=optimizer, device=device, num_epoch=epoch, scaler = scaler, scheduler= scheduler, scheduler_type=scheduler_type, max_grad_norm=max_grad_norm)

        val_map_50, val_map_all = validate_one_epoch(model, val_loader, device) #валидация
        
        is_best = val_map_50 > best_map #сравниваем mAp'ы после каждой эпохи
        if is_best:
            best_map = val_map_50

        is_best_train_loss = epoch_avg_loss < best_loss
        if is_best_train_loss:
            best_loss = epoch_avg_loss

        safe_checkpoint(model, optimizer, epoch, checkpoint_path, model_name, is_best)


    print(f"Обучение {model_name} завершено. Лучший train loss: {best_loss:.4f}")
    return model

