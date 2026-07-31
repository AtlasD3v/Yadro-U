import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def build_transformations(width, height):
    target_size = max(width, height)

    return A.Compose(
        [
            # 1. Изменение размера без искажения пропорций
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
            ),
            
            # 2. Быстрые геометрические трансформации
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), # Практически «бесплатно» для CPU
            
            # Легкий Аффин (без сложных аберраций)
            A.Affine(
                scale=(0.85, 1.15), 
                translate_percent=(-0.05, 0.05), 
                rotate=(-45, 45), 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=(114, 114, 114), 
                p=0.5
            ),

            # 3. Цветовые и контрастные изменения (быстрые пиксельные операции)
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=25, 
                val_shift_limit=15,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2), # Снизили p до 0.2 для скорости

            # 4. Быстрые размытия (убрали тяжелые Defocus и ZoomBlur)
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 9), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 9), p=0.5),
                ], 
                p=0.4
            ),

            # 5. Легкие шумы (убрали тяжелый MultiplicativeNoise)
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.03, 0.08), p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.03), p=0.5),
                ],
                p=0.4
            ),

            # 6. Нормализация и тензор
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
            min_area=16, # Отфильтровывает боксы меньше 4x4 пикселей, которые могут сломать RPN
        ),
    )