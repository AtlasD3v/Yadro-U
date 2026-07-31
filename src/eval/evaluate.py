import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.inference_mode()
def validate_one_epoch(model:torch.nn.Module, val_loader: torch.utils.data.DataLoader, device):
    model.eval()#Переводим модель в режим оценки

    #Инициализируем метрику для формата pascal_voc (xyxy)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for imgs, targets in val_loader:
        imgs = [img.to(device) for img in imgs]

        with torch.autocast(device_type=device.type, enabled=True):
            preds = model(imgs)

        # Переносим данные на CPU перед передачей в torchmetrics
        # (torchmetrics требует CPU-тензоры для подсчета mAP, чтобы не забивать VRAM)
        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

        #Обновляем накапливаемую статистику
        metric.update(preds_cpu, targets_cpu)

    #Финальный расчет всех mAP метрик
    results = metric.compute()

    # Извлекаем основные числа:
    map_val = results['map'].item()         # mAP@0.5:0.95 
    map_50_val = results['map_50'].item()   # mAP@0.50 
    map_75_val = results['map_75'].item()   # mAP@0.75

    print(f"📊 [Val Metrics] mAP@0.5: {map_50_val:.4f} | mAP@0.5:0.95: {map_val:.4f}")

    return map_50_val, map_val 