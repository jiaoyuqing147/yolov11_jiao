import os
import base64
from pathlib import Path
from xml.sax.saxutils import escape

# ================= 路径配置 =================
label_dir = Path(r"C:\Users\Administrator\Desktop\320\labels")
image_dir = Path(r"C:\Users\Administrator\Desktop\320\images")
save_dir = Path(r"C:\Users\Administrator\Desktop\320\images_svg")

save_dir.mkdir(exist_ok=True)

# 支持的图片后缀
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}

# 参数
CONF_THRES = 0.50
SKIP_CLASS_811 = True

# 适合淡蓝背景的亮黄色
BOX_COLOR = "#FFFF00"   # SVG里直接用RGB十六进制
TEXT_COLOR = "#FFFFFF"
LABEL_BG_OPACITY = 0.35
BOX_STROKE_WIDTH = 1.2

FONT_SIZE = 10
FONT_FAMILY = "Arial"


def image_to_data_uri(img_path: Path) -> str:
    ext = img_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_map.get(ext, "image/png")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def estimate_text_box(label: str, font_size: int):
    # 粗略估计文字框尺寸，够用
    text_w = int(len(label) * font_size * 0.62)
    text_h = int(font_size * 1.35)
    return text_w, text_h


for imgname in os.listdir(image_dir):
    img_path = image_dir / imgname
    if not img_path.is_file():
        continue
    if img_path.suffix not in img_exts:
        continue

    basename = img_path.stem
    txt_path = label_dir / f"{basename}.txt"

    from PIL import Image
    with Image.open(img_path) as im:
        width_real, height_real = im.size

    image_data_uri = image_to_data_uri(img_path)

    svg_parts = []
    svg_parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width_real}" height="{height_real}" viewBox="0 0 {width_real} {height_real}">''')
    svg_parts.append(f'''  <image href="{image_data_uri}" x="0" y="0" width="{width_real}" height="{height_real}"/>''')

    draw_count = 0

    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            a = line.split()
            if len(a) < 6:
                continue

            cls_id = int(float(a[0]))
            x = float(a[1])
            y = float(a[2])
            w = float(a[3])
            h = float(a[4])
            conf = float(a[5])

            if SKIP_CLASS_811 and cls_id == 811:
                continue
            if conf < CONF_THRES:
                continue

            x_center = width_real * x
            y_center = height_real * y
            box_w = width_real * w
            box_h = height_real * h

            x1 = x_center - box_w / 2.0
            y1 = y_center - box_h / 2.0
            x2 = x_center + box_w / 2.0
            y2 = y_center + box_h / 2.0

            x1 = max(0, min(x1, width_real - 1))
            y1 = max(0, min(y1, height_real - 1))
            x2 = max(0, min(x2, width_real - 1))
            y2 = max(0, min(y2, height_real - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            label = f"{cls_id} {conf:.2f}"
            label_esc = escape(label)

            text_w, text_h = estimate_text_box(label, FONT_SIZE)

            outside = (y1 - text_h - 4) >= 0
            if outside:
                bg_x = x1
                bg_y = y1 - text_h - 4
                text_x = x1 + 3
                text_y = y1 - 6
            else:
                bg_x = x1
                bg_y = y1
                text_x = x1 + 3
                text_y = y1 + text_h - 6

            # 矢量框
            svg_parts.append(
                f'''  <rect x="{x1:.2f}" y="{y1:.2f}" width="{(x2-x1):.2f}" height="{(y2-y1):.2f}" '''
                f'''fill="none" stroke="{BOX_COLOR}" stroke-width="{BOX_STROKE_WIDTH}" shape-rendering="geometricPrecision"/>'''
            )

            # 半透明标签底
            svg_parts.append(
                f'''  <rect x="{bg_x:.2f}" y="{bg_y:.2f}" width="{text_w + 6:.2f}" height="{text_h:.2f}" '''
                f'''fill="{BOX_COLOR}" fill-opacity="{LABEL_BG_OPACITY}"/>'''
            )

            # 矢量文字
            svg_parts.append(
                f'''  <text x="{text_x:.2f}" y="{text_y:.2f}" fill="{TEXT_COLOR}" '''
                f'''font-family="{FONT_FAMILY}" font-size="{FONT_SIZE}" '''
                f'''font-weight="normal">{label_esc}</text>'''
            )

            draw_count += 1

    svg_parts.append("</svg>")

    out_path = save_dir / f"{basename}.svg"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

    print(f"✅ 已保存: {out_path}，画了 {draw_count} 个框")

print("🎉 SVG 矢量标注图已全部生成")