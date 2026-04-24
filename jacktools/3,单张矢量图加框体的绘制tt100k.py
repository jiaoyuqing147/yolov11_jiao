import base64
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image

# =========================================================
# 这里直接指定：一张图片 + 一个txt + 一个输出svg
# =========================================================
# img_path = Path(r"E:\DataSets\tt100k_2021\yolojack\images\train\23858.jpg")
# txt_path = Path(r"E:\DataSets\tt100k_2021\yolojack\labels\train\23858.txt")
# out_path = Path(r"C:\Users\JACKJIAO\Desktop\23858\23858_original.svg")

#TT100K的检测图像的绘制
img_path = Path(r"E:\DataSets\tt100k_2021\yolojack\images\train\23858.jpg")
txt_path = Path(r"E:\DataSets\resultTT100k130train\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation\23858.txt")
out_path = Path(r"C:\Users\JACKJIAO\Desktop\23858\23858_yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation.svg")


#GTSDB的检测图像的绘制
# img_path = Path(r"C:\Users\JACKJIAO\Desktop\00239\00239_resize_640.jpg")
#
# txt_path = Path(r"E:\DataSets\resultGTSDBtrain\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation\00239.txt")
# # txt_path = Path(r"E:\DataSets\GTSDB\yolo43\labels\train\00239.txt")
# out_path = Path(r"C:\Users\JACKJIAO\Desktop\00239\00239_yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation.svg")
#


# 如果输出目录不存在，就创建
out_path.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# 参数配置
# =========================================================
CONF_THRES = 0.25
SKIP_CLASS_IDS = set()

NUM_CLASSES = 130

TEXT_COLOR = "#000000"
LABEL_BG_OPACITY = 0.6
BOX_STROKE_WIDTH = 2.0
FONT_SIZE =6
FONT_FAMILY = "Arial"

TEXT_AVOID_OVERLAP = False
LABEL_PADDING = 0
LABEL_LINE_GAP = 2

SHOW_CONF = True

# 背景右侧额外留白
LABEL_RIGHT_PAD = 4

# 标签左边是否和框体最外视觉边界对齐
ALIGN_TO_OUTER_STROKE = True

# 标签宽度是否强制等于检测框宽度
LABEL_SAME_AS_BOX_WIDTH = False

# 文本是否拉伸/压缩到占满整个标签框宽度
USE_TEXT_LENGTH_STRETCH = False

# 文本垂直位置微调
TEXT_BASELINE_OFFSET = 2

# 是否裁掉超出标签背景的文字
CLIP_TEXT_TO_LABEL = True

# =========================================================
# TT100K 类别名称
# =========================================================
CLASS_NAMES = [
    "pl80", "p6", "ph4.2", "pa13", "im", "w58", "pl90", "il70", "p5", "pm55",
    "pl60", "ip", "p11", "pdd", "wc", "i2r", "w30", "pmr", "p23", "pl15",
    "pm10", "pss", "w34", "iz", "p1n", "pr70", "pg", "il80", "pb", "pbm",
    "pm40", "ph4", "w45", "i4", "pl70", "i14", "p29", "pne", "pr60", "ph4.5",
    "p12", "p3", "pl5", "w13", "p14", "i4l", "pr30", "p17", "ph3", "pt",
    "pl30", "pctl", "pr50", "pm35", "i1", "pcd", "pbp", "pcr", "ps", "w18",
    "p10", "pn", "pa14", "p2", "ph2.5", "w55", "pw3", "pw4.5", "i12", "phclr",
    "i10", "i13", "w10", "p26", "p8", "w42", "il50", "p13", "pr40", "p25",
    "w41", "pl20", "pm30", "pl40", "pmb", "pr20", "p18", "i2", "w22", "w47",
    "pl120", "ph2.8", "w32", "pm15", "ph5", "pw3.2", "pl10", "il60", "w57", "pl100",
    "p16", "pl110", "w59", "w20", "ph2", "p9", "il100", "p19", "ph3.5", "pa10",
    "pcl", "pl35", "p15", "phcs", "w3", "pl25", "il110", "p1", "w46", "pn-2",
    "w63", "pm20", "i5", "il90", "w21", "p27", "pl50", "ph2.2", "pm2", "pw4"
]

# =========================================================
# 为每个类别生成固定颜色
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

    # color_map[43] = "#0057B8"  # w13
    color_map[98] = "#E63946"  # w57

    return color_map

# =========================================================
# 图片转 base64，嵌入 SVG
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
    }
    mime = mime_map.get(ext, "image/png")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"

# =========================================================
# 估算标签框大小
# =========================================================
def estimate_text_box(label: str, font_size: int):
    text_w = int(len(label) * font_size * 0.62)
    text_h = max(1, int(font_size * 1.2))
    return text_w, text_h

# =========================================================
# 判断两个标签框是否重叠
# =========================================================
def boxes_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw <= bx or
        bx + bw <= ax or
        ay + ah <= by or
        by + bh <= ay
    )

# =========================================================
# 标签避让
# =========================================================
def place_label_with_avoidance(
    anchor_x, anchor_y,
    label_w, label_h,
    width_real, height_real,
    used_label_boxes,
    prefer_outside=True,
    margin=0,
    line_gap=2,
):
    candidates = []

    if prefer_outside:
        candidates.append((anchor_x, anchor_y - label_h - margin, "outside"))

    candidates.append((anchor_x, anchor_y, "inside"))

    for start_x, start_y, mode in candidates:
        bg_x = max(0, min(start_x, width_real - label_w))
        bg_y = start_y

        if bg_y < 0:
            bg_y = 0

        while True:
            candidate_box = (bg_x, bg_y, label_w, label_h)

            overlapped = any(
                boxes_overlap(candidate_box, old_box)
                for old_box in used_label_boxes
            )

            if bg_y + label_h > height_real:
                break

            if not overlapped:
                used_label_boxes.append(candidate_box)
                return bg_x, bg_y

            bg_y += label_h + line_gap

    bg_x = max(0, min(anchor_x, width_real - label_w))
    bg_y = max(0, min(anchor_y, height_real - label_h))
    used_label_boxes.append((bg_x, bg_y, label_w, label_h))
    return bg_x, bg_y

# =========================================================
# 根据类别ID获取类别名称
# =========================================================
def get_class_name(cls_id: int) -> str:
    if 0 <= cls_id < len(CLASS_NAMES):
        return CLASS_NAMES[cls_id]
    return f"cls_{cls_id}"

# =========================================================
# 标签左边与框体左边视觉对齐
# =========================================================
def get_label_left(rx1: int) -> int:
    if ALIGN_TO_OUTER_STROKE:
        label_left = int(round(rx1 - BOX_STROKE_WIDTH / 2))
    else:
        label_left = rx1
    return max(0, label_left)

# =========================================================
# 核心函数：生成单张 SVG
# =========================================================
def build_svg(img_path: Path, txt_path: Path, out_path: Path, class_colors: dict):
    if not img_path.exists():
        print(f"❌ 图片不存在: {img_path}")
        return

    with Image.open(img_path) as im:
        width_real, height_real = im.size

    image_data_uri = image_to_data_uri(img_path)

    svg_parts = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_real}" height="{height_real}" '
        f'viewBox="0 0 {width_real} {height_real}">'
    )

    # 可选的裁剪定义
    svg_parts.append("  <defs>")

    svg_parts.append("  </defs>")

    svg_parts.append(
        f'  <image href="{image_data_uri}" x="0" y="0" '
        f'width="{width_real}" height="{height_real}"/>'
    )

    draw_count = 0
    used_label_boxes = []

    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            a = line.split()
            if len(a) < 5:
                continue

            cls_id = int(float(a[0]))
            x = float(a[1])
            y = float(a[2])
            w = float(a[3])
            h = float(a[4])

            is_prediction = len(a) >= 6
            conf = float(a[5]) if is_prediction else 1.0

            if cls_id in SKIP_CLASS_IDS:
                continue

            if is_prediction and conf < CONF_THRES:
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

            rx1 = int(round(x1))
            ry1 = int(round(y1))
            rx2 = int(round(x2))
            ry2 = int(round(y2))

            box_w_px = max(1, rx2 - rx1)
            box_h_px = max(1, ry2 - ry1)

            box_color = class_colors.get(cls_id, "#FFFF00")
            class_name = get_class_name(cls_id)

            if SHOW_CONF and is_prediction:
                label = f"{class_name} {conf:.2f}"
            else:
                label = class_name

            label_esc = escape(label)

            text_w0, text_h = estimate_text_box(label, FONT_SIZE)
            label_h = text_h

            label_left = get_label_left(rx1)
            box_right_visual = rx2 + BOX_STROKE_WIDTH / 2
            max_label_w = max(1, int(round(box_right_visual - label_left)))

            desired_label_w = text_w0 + LABEL_RIGHT_PAD

            if LABEL_SAME_AS_BOX_WIDTH:
                label_w = max_label_w
            else:
                label_w = min(desired_label_w, max_label_w)

            if TEXT_AVOID_OVERLAP:
                bg_x, bg_y = place_label_with_avoidance(
                    anchor_x=label_left,
                    anchor_y=ry1,
                    label_w=label_w,
                    label_h=label_h,
                    width_real=width_real,
                    height_real=height_real,
                    used_label_boxes=used_label_boxes,
                    prefer_outside=True,
                    margin=0,
                    line_gap=LABEL_LINE_GAP,
                )
            else:
                outside = (ry1 - label_h) >= 0
                if outside:
                    bg_x = label_left
                    bg_y = max(0, int(round(ry1 - label_h)))
                else:
                    bg_x = label_left
                    bg_y = ry1

            text_x = bg_x
            text_y = bg_y + label_h - TEXT_BASELINE_OFFSET

            clip_id = f"clip_label_{idx}"

            svg_parts.append(
                f'  <clipPath id="{clip_id}">'
                f'<rect x="{bg_x}" y="{bg_y}" width="{label_w}" height="{label_h}"/></clipPath>'
            )

            svg_parts.append(
                f'  <rect x="{rx1}" y="{ry1}" '
                f'width="{box_w_px}" height="{box_h_px}" '
                f'fill="none" stroke="{box_color}" '
                f'stroke-width="{BOX_STROKE_WIDTH}" '
                f'shape-rendering="geometricPrecision"/>'
            )

            svg_parts.append(
                f'  <rect x="{bg_x}" y="{bg_y}" '
                f'width="{label_w}" height="{label_h}" '
                f'fill="{box_color}" fill-opacity="{LABEL_BG_OPACITY}"/>'
            )

            if USE_TEXT_LENGTH_STRETCH:
                svg_parts.append(
                    f'  <text x="{text_x}" y="{text_y}" '
                    f'fill="{TEXT_COLOR}" '
                    f'font-family="{FONT_FAMILY}" '
                    f'font-size="{FONT_SIZE}" '
                    f'font-weight="bold" '
                    f'textLength="{max(1, label_w)}" '
                    f'lengthAdjust="spacingAndGlyphs" '
                    f'clip-path="url(#{clip_id})">'
                    f'{label_esc}</text>'
                )
            else:
                if CLIP_TEXT_TO_LABEL:
                    svg_parts.append(
                        f'  <text x="{text_x}" y="{text_y}" '
                        f'fill="{TEXT_COLOR}" '
                        f'font-family="{FONT_FAMILY}" '
                        f'font-size="{FONT_SIZE}" '
                        f'font-weight="bold" '
                        f'clip-path="url(#{clip_id})">'
                        f'{label_esc}</text>'
                    )
                else:
                    svg_parts.append(
                        f'  <text x="{text_x}" y="{text_y}" '
                        f'fill="{TEXT_COLOR}" '
                        f'font-family="{FONT_FAMILY}" '
                        f'font-size="{FONT_SIZE}" '
                        f'font-weight="bold">'
                        f'{label_esc}</text>'
                    )

            draw_count += 1
    else:
        print(f"⚠️ txt不存在: {txt_path}")

    svg_parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

    print(f"✅ 已保存: {out_path}")
    print(f"✅ 共绘制 {draw_count} 个框")

if __name__ == "__main__":
    class_colors = generate_class_colors(NUM_CLASSES)
    build_svg(img_path, txt_path, out_path, class_colors)