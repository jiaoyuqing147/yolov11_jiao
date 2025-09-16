import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_and_crop(
    img_path: str,
    x: int, y: int, w: int, h: int,
    out_dir: str,
    rect_color=(255, 0, 0),  # 红色
    rect_width=4,            # 边框粗细
    label: str | None = None # 可选：在框上方写点字
):
    """
    在图像上画出矩形 (x,y,w,h)，保存标注图；再裁剪该区域保存到 out_dir。
    坐标单位：像素，(x,y) 是左上角。
    """

    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 打开图像
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # 边界裁剪（避免越界）
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))

    # 如果给的框超出边界，这里会自动夹到图像内；也可以选择直接报错
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"无效的裁剪区域：({x},{y},{w},{h}) 在图像尺寸 ({W}x{H}) 外。")

    # 1) 在原图上画矩形与可选标签
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=rect_color, width=rect_width)

    if label:
        try:
            # 新版 Pillow 优先用 textbbox 计算文本尺寸
            l, t, r, b = draw.textbbox((0, 0), label)
            tw, th = r - l, b - t
        except Exception:
            # 兼容旧版
            font = ImageFont.load_default()
            tw, th = draw.textsize(label, font=font)

        tx = max(0, min(x1, W - tw - 1))
        ty = max(0, y1 - th - 2)
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=rect_color)
        draw.text((tx, ty), label, fill=(0, 0, 0))

    # 保存带框图
    marked_name = f"{img_path.stem}_box_{x1}_{y1}_{x2-x1}x{y2-y1}.jpg"
    marked_path = out_dir / marked_name
    img.save(marked_path, quality=95)

    # 2) 裁剪并保存
    crop = Image.open(img_path).convert("RGB").crop((x1, y1, x2, y2))
    crop_name = f"{img_path.stem}_crop_{x1}_{y1}_{x2-x1}x{y2-y1}.png"
    crop_path = out_dir / crop_name
    crop.save(crop_path)

    print(f"[OK] 标注图: {marked_path}")
    print(f"[OK] 裁剪图: {crop_path}")
    return str(marked_path), str(crop_path)


if __name__ == "__main__":
    # === 示例参数 ===
    img_path = r"E:\DataSets\ceshi\9447 - 副本640.jpg"   # 原图路径
    out_dir  = r"E:\DataSets\ceshi_crops"      # 输出目录

    # 左上角+宽高（像素）
    x, y = 150, 200
    w, h = 490, 200

    # 可选标签（显示在框上面）
    draw_and_crop(img_path, x, y, w, h, out_dir, rect_color=(0, 112, 192), rect_width=4, label="ROI")
