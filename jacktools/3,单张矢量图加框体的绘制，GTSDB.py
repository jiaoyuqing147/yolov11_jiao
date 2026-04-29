# =========================================================
# 1. 类别名称列表
#    下标必须和 YOLO txt 里的类别编号一一对应
# =========================================================
CLASS_NAMES = [
  "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
  "speed limit 70", "speed limit 80", "restriction ends 80", "speed limit 100",
  "speed limit 120", "no overtaking", "no overtaking (trucks)", "priority at next intersection",
  "priority road", "give way", "stop", "no traffic both ways",
  "no trucks", "no entry", "danger", "bend left", "bend right",
  "bend", "uneven road", "slippery road", "road narrows", "construction",
  "traffic signal", "pedestrian crossing", "school crossing", "cycles crossing",
  "snow", "animals", "restriction ends", "go right", "go left",
  "go straight", "go right or straight", "go left or straight", "keep right",
  "keep left", "roundabout", "restriction ends (overtaking)", "restriction ends (overtaking trucks)"
]

# =========================================================
# 2. 输入输出路径
# =========================================================
image_path = r"C:\Users\JACKJIAO\Desktop\00239\00239_resize_640.jpg"   # 你的图像路径
# txt_path   = r"E:\DataSets\GTSDB\yolo43\labels\train\00239.txt"
# txt_path   = r"E:\DataSets\resultGTSDBtrain\yolo11_train200\00239.txt"
txt_path   = r"E:\DataSets\resultGTSDBtrain\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation\00239.txt"         # 你的 YOLO txt 路径txt_path   = r"E:\DataSets\resultGTSDBtrain\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation\00239.txt"         # 你的 YOLO txt 路径
save_path  = r"C:\Users\JACKJIAO\Desktop\00239\00239_yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation.svg"   # 输出SVG路径


import os
import base64
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image

# =========================================================
# 3. 参数设置
# =========================================================
FONT_SIZE = 9                  # 文字大小（SVG里直接控制）
FONT_FAMILY = "Arial"
FONT_WEIGHT = "bold"

TEXT_COLOR = "#000000"          # 文字颜色
LABEL_BG_OPACITY = 0.75         # 标签背景透明度
BOX_STROKE_WIDTH = 2.0         # 框线宽
CONF_DECIMALS = 2               # 置信度保留几位小数

SHOW_CONF = True                # 是否显示置信度
CONF_THRES = 0.0                # 预测框置信度阈值（GT无影响）

LABEL_PADDING_X = 4             # 标签左右内边距
LABEL_PADDING_Y = 2             # 标签上下内边距
TEXT_BASELINE_OFFSET = 3        # 文字基线微调
LABEL_OUTSIDE_MARGIN = 0        # 标签放在框外时，与框的间距

# 是否把标签放到框上方；如果上方放不下就放框内
PREFER_LABEL_OUTSIDE = True

# 是否裁掉超出标签背景区域的文字
CLIP_TEXT_TO_LABEL = False

# =========================================================
# 4. HEX颜色转BGR（保留给你原逻辑兼容用）
#    这里SVG主要直接用HEX
# =========================================================
def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

# =========================================================
# 5. 固定调色板（返回 HEX，SVG 直接可用）
# =========================================================
def generate_class_colors(num_classes: int):
    base_colors = [
        "#FF0000", "#00FF00", "#0000FF",
        "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#FF4D4F", "#0066FF",
        "#FF1493", "#7FFF00", "#FFD700",
        "#FF4500", "#00CED1", "#9400D3",
    ]

    color_map = {}
    for i in range(num_classes):
        color_map[i] = base_colors[i % len(base_colors)]

    # 如需单独指定某类颜色，可在此覆盖
    color_map[98] = "#E63946"

    return color_map

CLASS_COLORS = generate_class_colors(len(CLASS_NAMES))

# =========================================================
# 6. 读取 YOLO txt
#    支持:
#    1) GT:   class x y w h
#    2) Pred: class x y w h conf
# =========================================================
def load_yolo_labels(txt_file):
    labels = []

    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"找不到 txt 文件: {txt_file}")

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            conf = None
            if len(parts) >= 6:
                try:
                    conf = float(parts[5])
                except ValueError:
                    conf = None

            labels.append((class_id, x_center, y_center, width, height, conf))

    return labels

# =========================================================
# 7. 获取类别名
# =========================================================
def get_class_name(class_id):
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"

# =========================================================
# 8. 图像转 base64，嵌入 SVG
# =========================================================
def image_to_data_uri(img_path: Path) -> str:
    ext = img_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/png")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"

# =========================================================
# 9. 估算文字宽高
#    这里只是近似，用于生成标签背景框
# =========================================================
from PIL import ImageFont

def estimate_text_box(label: str, font_size: int):
    """
    使用真实字体计算文本宽高（精准版）
    """

    # Windows字体路径（你是Windows环境）
    font_path = "C:/Windows/Fonts/arial.ttf"

    font = ImageFont.truetype(font_path, font_size)

    # 获取文本边界框
    bbox = font.getbbox(label)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    return text_w, text_h

# =========================================================
# 10. YOLO归一化坐标 -> 像素坐标
# =========================================================
def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    x1 = (x_center - width / 2) * img_w
    y1 = (y_center - height / 2) * img_h
    x2 = (x_center + width / 2) * img_w
    y2 = (y_center + height / 2) * img_h

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2

# =========================================================
# 11. 构建单张 SVG
# =========================================================
def build_svg(image_path, txt_path, save_path):
    img_path = Path(image_path)
    txt_path = Path(txt_path)
    out_path = Path(save_path)

    if not img_path.exists():
        raise FileNotFoundError(f"图像不存在: {img_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取图像尺寸
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    # 图像嵌入SVG
    image_data_uri = image_to_data_uri(img_path)

    # 读取标签
    labels = load_yolo_labels(txt_path)

    svg_parts = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{img_w}" height="{img_h}" '
        f'viewBox="0 0 {img_w} {img_h}">'
    )

    svg_parts.append("  <defs>")
    svg_parts.append("  </defs>")

    # 原图
    svg_parts.append(
        f'  <image href="{image_data_uri}" x="0" y="0" width="{img_w}" height="{img_h}"/>'
    )

    draw_count = 0

    for idx, item in enumerate(labels):
        class_id, x_center, y_center, width, height, conf = item

        # 如果是预测结果并且设置了阈值，则过滤
        if conf is not None and conf < CONF_THRES:
            continue

        class_name = get_class_name(class_id)
        color = CLASS_COLORS.get(class_id, "#00FF00")

        # 显示标签
        if SHOW_CONF and conf is not None:
            label = f"{class_name} {conf:.{CONF_DECIMALS}f}"
        else:
            label = class_name

        label_esc = escape(label)

        # 坐标转换
        x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h)

        if x2 <= x1 or y2 <= y1:
            continue

        rx1 = int(round(x1))
        ry1 = int(round(y1))
        rx2 = int(round(x2))
        ry2 = int(round(y2))

        box_w = max(1, rx2 - rx1)
        box_h = max(1, ry2 - ry1)

        # 估算标签框尺寸
        text_w, text_h = estimate_text_box(label, FONT_SIZE)
        label_w = text_w + LABEL_PADDING_X * 2
        label_h = text_h + LABEL_PADDING_Y * 2

        # 默认放在框上方
        label_x = rx1
        if PREFER_LABEL_OUTSIDE and (ry1 - label_h - LABEL_OUTSIDE_MARGIN) >= 0:
            label_y = ry1 - label_h - LABEL_OUTSIDE_MARGIN
        else:
            # 放到框内顶部
            label_y = ry1

        # 防止右边越界
        if label_x + label_w > img_w:
            label_x = max(0, img_w - label_w)

        # 防止下边越界
        if label_y + label_h > img_h:
            label_y = max(0, img_h - label_h)

        # 防止上边越界
        if label_y < 0:
            label_y = 0

        # 文字位置
        text_x = label_x + LABEL_PADDING_X
        text_y = label_y + label_h - TEXT_BASELINE_OFFSET

        clip_id = f"clip_label_{idx}"

        if CLIP_TEXT_TO_LABEL:
            svg_parts.append(
                f'  <clipPath id="{clip_id}">'
                f'<rect x="{label_x}" y="{label_y}" width="{label_w}" height="{label_h}"/></clipPath>'
            )

        # 画框
        svg_parts.append(
            f'  <rect x="{rx1}" y="{ry1}" '
            f'width="{box_w}" height="{box_h}" '
            f'fill="none" stroke="{color}" '
            f'stroke-width="{BOX_STROKE_WIDTH}" '
            f'shape-rendering="geometricPrecision"/>'
        )

        # 画标签背景
        svg_parts.append(
            f'  <rect x="{label_x}" y="{label_y}" '
            f'width="{label_w}" height="{label_h}" '
            f'fill="{color}" fill-opacity="{LABEL_BG_OPACITY}"/>'
        )

        # 画文字
        if CLIP_TEXT_TO_LABEL:
            svg_parts.append(
                f'  <text x="{text_x}" y="{text_y}" '
                f'fill="{TEXT_COLOR}" '
                f'font-family="{FONT_FAMILY}" '
                f'font-size="{FONT_SIZE}" '
                f'font-weight="{FONT_WEIGHT}" '
                f'clip-path="url(#{clip_id})">{label_esc}</text>'
            )
        else:
            svg_parts.append(
                f'  <text x="{text_x}" y="{text_y}" '
                f'fill="{TEXT_COLOR}" '
                f'font-family="{FONT_FAMILY}" '
                f'font-size="{FONT_SIZE}" '
                f'font-weight="{FONT_WEIGHT}">{label_esc}</text>'
            )

        draw_count += 1

    svg_parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

    print(f"[INFO] 已保存 SVG 到: {out_path}")
    print(f"[INFO] 共绘制 {draw_count} 个框")

# =========================================================
# 12. 程序入口
# =========================================================
if __name__ == "__main__":
    build_svg(image_path, txt_path, save_path)