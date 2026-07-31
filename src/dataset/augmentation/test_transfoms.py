import albumentations as A
import cv2

pic_path = "C:\\Users\\atlas\\Pictures\\111.jpg"

def read_img(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

trans = A.RandomShadow(p=1.0)

img_rgb = read_img(pic_path)
aug_rgb = trans(image=img_rgb)["image"]

aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

cv2.imshow("Test MotionBlur", aug_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()