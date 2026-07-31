import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import cv2
import albumentations as A

import os
import configparser
from pathlib import Path
from typing import Optional

from src.dataset.augmentation import buld_transform_lite as b_t

# Динамически находим корень проекта:
# __file__ -> src/dataset/data_setup.py
# .parents[2] -> корень проекта (где лежит папка cfg)
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH = BASE_DIR / "cfg" / "config.ini"

class MyCustomDataset(Dataset):
    def __init__(
        self, 
        model_type: str = "torchvision", 
        split: str = "train", 
        cfg_path: Optional[str] = None,
        dataset_root: Optional[str] = None
    ):

        """
        :param model_type: "torchvision" или кастомная архитектура
        :param split: "train" или "val"
        :param cfg_path: явный путь к config.ini (если None, ищется в коде проекта)
        :param dataset_root: явный путь к папке с датасетом (например, "D:/practicum/final_dataset")
        """

        # 1. Если путь не передан, используем относительный дефолтный
        cfg_path_obj = Path(cfg_path) if cfg_path else DEFAULT_CFG_PATH
        self.config = configparser.ConfigParser()

        if cfg_path_obj.exists():
            self.config.read(cfg_path_obj, encoding='utf-8')
        else:
            raise FileNotFoundError(f"Файл конфигурации не найден по пути: {cfg_path_obj}")
        
        self.model_type = model_type

        # 2. Определяем корневую папку датасета (dataset_root)
        # Иерархия: Аргумент функции -> Переменная окружения -> Корень проекта
        if dataset_root:
            root = Path(dataset_root)
        elif "DATASET_ROOT" in os.environ:
            root = Path(os.environ["DATASET_ROOT"])
        else:
            root = BASE_DIR

        # 3. Извлекаем относительные пути из config.ini
        
        rel_img_path = self.config.get('dataset', f'{split}_images_path', fallback=f'{split}/images')
        rel_lbl_path = self.config.get('dataset', f'{split}_labels_path', fallback=f'{split}/labels')

        # 4. Склеиваем корневой путь и относительные пути из конфига
        # (Path подменяет \ на / автоматически в зависимости от OS)
        clean_img_path = rel_img_path.replace("\\", "/")
        clean_lbl_path = rel_lbl_path.replace("\\", "/")

        self.img_dir = (root / Path(clean_img_path)).resolve()
        self.labels_dir = (root / Path(clean_lbl_path)).resolve()

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Директория с изображениями не найдена по пути: {self.img_dir}")
        
        # 5. Целевое разрешение
        self.resolution_w = self.config.getint('images', 'resolution_w', fallback=640)
        self.resolution_h = self.config.getint('images', 'resolution_h', fallback=640)
        
        # 6. Чтение списка файлов
        self.image_files = sorted([
            entry for entry in os.listdir(self.img_dir)
            if (self.img_dir / entry).is_file()
        ])

        self.transform = b_t.build_transformations(width = self.resolution_w, height = self.resolution_h)


    def __len__(self):
        return len(self.image_files)

    def load_image(self, path):
        img = cv2.imread(path)
        image_rgb = cv2.cvtColor(img, code = cv2.COLOR_BGR2RGB)

        return image_rgb
    
    def read_labels(self, img_path):
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        labels_path = os.path.join(self.labels_dir, file_name + ".txt")

        boxes = []
        classes = []

        if not os.path.exists(labels_path):
            return boxes, classes

        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                classes.append(class_id)
                boxes.append(bbox)

        return boxes, classes


    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.image_files[index])

        image = self.load_image(image_path)
        boxes, classes = self.read_labels(image_path)

        transform_results = self.transform(
            image=image,
            bboxes=boxes,
            class_labels=classes
        )

        aug_image = transform_results['image']         # Тензор PyTorch [C, H, W] благодаря ToTensorV2()
        aug_boxes = transform_results['bboxes']         # Преобразованные боксы
        aug_classes = transform_results['class_labels'] # Обновленные классы (если какие-то боксы отфильтровались)

        new_h, new_w = aug_image.shape[1], aug_image.shape[2]

        if self.model_type == "torchvision":

            boxes_abs = [self._yolo_to_pascal_voc(b, new_w, new_h) for b in aug_boxes]
            boxes_tensor = torch.as_tensor(boxes_abs, dtype=torch.float32).reshape(-1, 4)
            labels_tensor = torch.as_tensor(aug_classes, dtype=torch.int64)

            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "image_id": torch.tensor([index]),
            }
        
        return aug_image, target
    

    @staticmethod
    def _yolo_to_pascal_voc(bbox, img_w, img_h):
        """
        Конвертирует нормализованный yolo-формат (cx, cy, w, h)
        в абсолютные пиксельные координаты (x_min, y_min, x_max, y_max),
        которые ожидают torchvision-детекторы (Faster R-CNN, SSD, RetinaNet и т.д.)
        """
        cx, cy, w, h = bbox
        x_min = (cx - w / 2) * img_w
        y_min = (cy - h / 2) * img_h
        x_max = (cx + w / 2) * img_w
        y_max = (cy + h / 2) * img_h
        return [x_min, y_min, x_max, y_max]