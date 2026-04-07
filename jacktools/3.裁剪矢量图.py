import os
import re
from pathlib import Path

input_dir = Path(r"C:\Users\Administrator\Desktop\320\images_svg")   # 你原来的SVG目录
output_dir = Path(r"C:\Users\Administrator\Desktop\320\images_svg_crop320")
output_dir.mkdir(exist_ok=True)

# 原图是 640x640，裁中心 320x320
crop_x = 160
crop_y = 160
crop_w = 320
crop_h = 320

for svg_file in input_dir.glob("*.svg"):
    text = svg_file.read_text(encoding="utf-8")

    # 改 width / height
    text = re.sub(r'width="[^"]+"', f'width="{crop_w}"', text, count=1)
    text = re.sub(r'height="[^"]+"', f'height="{crop_h}"', text, count=1)

    # 改 viewBox
    if 'viewBox="' in text:
        text = re.sub(r'viewBox="[^"]+"', f'viewBox="{crop_x} {crop_y} {crop_w} {crop_h}"', text, count=1)
    else:
        text = text.replace(
            "<svg ",
            f'<svg viewBox="{crop_x} {crop_y} {crop_w} {crop_h}" ',
            1
        )

    out_path = output_dir / svg_file.name
    out_path.write_text(text, encoding="utf-8")

    print(f"✅ 已裁剪: {out_path}")

print("🎉 全部SVG已裁成中心320×320")