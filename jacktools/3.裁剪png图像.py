import os
import cv2
from pathlib import Path

input_dir = Path(r"C:\Users\Administrator\Desktop\320\images")   # 放PNG/JPG
output_dir = Path(r"C:\Users\Administrator\Desktop\320\images_crop320")
output_dir.mkdir(exist_ok=True)

crop_size = 320

for img_path in input_dir.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    # 中心裁剪
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    crop = img[start_y:start_y+crop_size, start_x:start_x+crop_size]

    save_path = output_dir / img_path.name
    cv2.imwrite(str(save_path), crop)

    print(f"✅ 已裁剪: {save_path}")

print("🎉 全部图片裁剪完成")