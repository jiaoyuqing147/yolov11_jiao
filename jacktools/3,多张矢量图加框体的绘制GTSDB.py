import base64
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image

# =========================================================
# 一张图片 + 多个预测txt + 批量输出多个SVG
# =========================================================
img_path = Path(r"E:\DataSets\resultGTSDBtrain\compare_3\TopK_vis\00112.jpg")

txt_dirs = [
    Path(r"E:\DataSets\resultGTSDBtrain\yolo11_train200"),
    Path(r"E:\DataSets\resultGTSDBtrain\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200"),
    Path(r"E:\DataSets\resultGTSDBtrain\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation"),
]

output_dir = Path(r"C:\Users\JACKJIAO\Desktop\对比实验")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# 参数配置
# =========================================================
CONF_THRES = 0.25
SKIP_CLASS_IDS = set()
NUM_CLASSES = 130

TEXT_COLOR = "#000000"
LABEL_BG_OPACITY = 0.85
BOX_STROKE_WIDTH = 3.0
FONT_SIZE =45
FONT_FAMILY = "Arial"

SHOW_CONF = True

# 这里专门控制“贴合程度”
LABEL_PAD_X = 2   # 左右内边距，想更紧就改成1
LABEL_PAD_Y = 1   # 上下内边距，想更紧就改成0

TEXT_AVOID_OVERLAP = False
LABEL_LINE_GAP = 2
ALIGN_TO_OUTER_STROKE = True
TEXT_BASELINE_OFFSET = 2

# =========================================================
# TT100K 类别名称
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
# 为每个类别生成固定颜色
# =========================================================
def generate_class_colors(num_classes: int):
    # base_colors = [
    #     "#FF0000", "#00FF00", "#0000FF",
    #     "#FFFF00", "#FF00FF", "#00FFFF",
    #     "#FFA500", "#FF4D4F", "#0066FF",
    #     "#FF1493", "#7FFF00", "#FFD700",
    #     "#FF4500", "#00CED1", "#9400D3",
    # ]
    base_colors = [
        "#FF3B3B",  # 亮红
        "#00FF7F",  # 荧光绿
        "#00BFFF",  # 亮蓝
        "#FFFF33",  # 亮黄
        "#FF00FF",  # 洋红
        "#00FFFF",  # 青色
        "#FF8C00",  # 亮橙
        "#FF69B4",  # 亮粉
        "#1E90FF",  # 道奇蓝（很亮）
        "#ADFF2F",  # 黄绿
        "#FFD700",  # 金色（亮）
        "#FF6347",  # 番茄红
        # "#40E0D0",  # 亮青绿
        "#BA55D3",  # 亮紫
        "#FFFFFF",  # 白色（在黑底最清晰）
    ]

    color_map = {}
    for i in range(num_classes):
        color_map[i] = base_colors[i % len(base_colors)]
    color_map[98] = "#E63946"
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
# 更贴合文本的尺寸估算
# =========================================================
def estimate_text_box(label: str, font_size: int):
    text_w = int(len(label) * font_size * 0.58)
    text_h = int(font_size * 1.00)
    return text_w, text_h

# =========================================================
# 标签框重叠判断
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
        candidates.append((anchor_x, anchor_y - label_h - margin))

    candidates.append((anchor_x, anchor_y))

    for start_x, start_y in candidates:
        bg_x = max(0, min(start_x, width_real - label_w))
        bg_y = max(0, start_y)

        while True:
            candidate_box = (bg_x, bg_y, label_w, label_h)
            overlapped = any(boxes_overlap(candidate_box, old_box) for old_box in used_label_boxes)

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
# 生成单张 SVG
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
    svg_parts.append(f'  <image href="{image_data_uri}" x="0" y="0" width="{width_real}" height="{height_real}"/>')

    draw_count = 0
    used_label_boxes = []

    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
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

            x1 = max(0, min(x_center - box_w / 2.0, width_real - 1))
            y1 = max(0, min(y_center - box_h / 2.0, height_real - 1))
            x2 = max(0, min(x_center + box_w / 2.0, width_real - 1))
            y2 = max(0, min(y_center + box_h / 2.0, height_real - 1))

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

            label = f"{class_name} {conf:.2f}" if (SHOW_CONF and is_prediction) else class_name
            label_esc = escape(label)

            text_w, text_h = estimate_text_box(label, FONT_SIZE)

            # 完全贴合文本
            label_w = text_w + LABEL_PAD_X * 2
            label_h = text_h + LABEL_PAD_Y * 2

            label_left = get_label_left(rx1)

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
                if ry1 - label_h >= 0:
                    bg_x = max(0, min(label_left, width_real - label_w))
                    bg_y = ry1 - label_h
                else:
                    bg_x = max(0, min(label_left, width_real - label_w))
                    bg_y = ry1

            text_x = bg_x + LABEL_PAD_X
            text_y = bg_y + LABEL_PAD_Y + text_h - TEXT_BASELINE_OFFSET

            # 画框
            svg_parts.append(
                f'  <rect x="{rx1}" y="{ry1}" '
                f'width="{box_w_px}" height="{box_h_px}" '
                f'fill="none" stroke="{box_color}" '
                f'stroke-width="{BOX_STROKE_WIDTH}" '
                f'shape-rendering="geometricPrecision"/>'
            )

            # 画标签背景（紧贴文字）
            svg_parts.append(
                f'  <rect x="{bg_x}" y="{bg_y}" '
                f'width="{label_w}" height="{label_h}" '
                f'fill="{box_color}" fill-opacity="{LABEL_BG_OPACITY}"/>'
            )

            # 画文字
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

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    class_colors = generate_class_colors(NUM_CLASSES)

    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在：{img_path}")

    stem = img_path.stem  # 88619

    for txt_dir in txt_dirs:
        model_name = txt_dir.name
        txt_path = txt_dir / f"{stem}.txt"
        out_path = output_dir / f"{stem}_{model_name}.svg"

        print("=" * 80)
        print(f"处理模型：{model_name}")
        print(f"txt路径：{txt_path}")
        build_svg(img_path, txt_path, out_path, class_colors)