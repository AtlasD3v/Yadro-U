import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def build_transformations(width, height):
    target_size = max(width, height)

    return A.Compose(
        [
            #ресайз без потерь мастшаба
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(
                min_height= height,
                min_width= width,
                border_mode= cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ),
            #геометрические преобразования
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-180, 180), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.6),
            A.Perspective(scale=(0.02, 0.08), p=0.5),

            #пиксельные преобразования
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), p = 0.7),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20,
                p = 0.6
            ),
            A.CLAHE(
                clip_limit=3,
                tile_grid_size=(8,8),
                p = 0.55
            ),
            A.OneOf( #блюры
                [
                    A.MotionBlur(blur_limit=(3,15), p = 0.5),
                    A.GaussianBlur(blur_limit=(3,15), p=0.5),
                    A.Defocus(p = 0.5),
                    A.ZoomBlur(p = 0.2)
                ], p=0.6
            ),
            A.OneOf( #шумы
                [
                    A.GaussNoise(std_range=(0.07, 0.11), p=0.5),
                    A.ISONoise(color_shift=(0.02, 0.045), p=0.5),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.4), p=0.5)
                ],
                p = 0.6
            ),
            A.OneOf(#погодные эффекты
                [
                    A.RandomRain(p = 0.5),
                    A.RandomSunFlare(src_radius= 150, p = 0.5),
                    A.RandomShadow(p = 0.5)
                ],
                p = 0.15
            ),
            #нормализация и преобразование в нужный тензор
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
        ),
    )