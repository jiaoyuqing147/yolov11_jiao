import os
import base64
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image

# =========================================================
# 路径配置
# =========================================================

# 原图目录
image_dir = Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\compare_strict_small\TopK_vis")

# 三个模型预测 txt 所在目录
model_dirs = {
    "yolo11_train200": Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\yolo11_train200"),
    "yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200": Path(
        r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200"
    ),
    "yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation": Path(
        r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation"
    ),
}

# 输出 SVG 目录
save_dir = Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\compare_strict_small\TopK_vis_svg")
save_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# 参数配置
# =========================================================

# 支持的图像后缀
img_exts = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"
}

NUM_CLASSES = 130         # TT100K 类别数
CONF_THRES = 0.25         # 置信度阈值，小于这个值的框不画
SKIP_CLASS_IDS = set()    # 如果要跳过某些类别，在这里写，例如 {5, 7}

# TT100K 类别名称，索引必须与类别ID严格对应
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

# 框和文字的整体视觉参数
# TEXT_COLOR = "#FFFFFF"      # 文字颜色
TEXT_COLOR = "#000000"      # 黑色
LABEL_BG_OPACITY = 0.7     # 标签背景透明度，1=完全不透明
BOX_STROKE_WIDTH = 1.5      # 框体线宽
FONT_SIZE = 13               # 字体大小
FONT_FAMILY = "Arial"       # 字体名称

# 是否开启文字避让
# True  = 开启，标签尽量不重叠
# False = 关闭，仍然使用“紧贴框体”的老方式
TEXT_AVOID_OVERLAP = False

# 标签与框体的微小间距（用于关闭避让时的紧贴风格）
LABEL_PADDING = 1

# 开启文字避让后，标签每次往下挪动的额外间距
LABEL_LINE_GAP = 2

# 是否显示置信度
# True  = 标签内容显示 “类别名 + 置信度”
# False = 标签内容只显示 “类别名”
SHOW_CONF = True

# =========================================================
# 工具函数：为每个类别生成固定颜色
# =========================================================
def generate_class_colors(num_classes: int):
    """
    为每个类别生成固定颜色。
    同一个类别在不同模型中，颜色保持一致。

    返回：
        color_map: {class_id: "#RRGGBB"}
    """
    base_colors = [
        "#FF0000", "#00FF00", "#0000FF",
        "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#00FF7F", "#1E90FF",
        "#FF1493", "#7FFF00", "#FFD700",
        "#FF4500", "#00CED1", "#9400D3",
    ]

    color_map = {}
    for i in range(num_classes):
        color_map[i] = base_colors[i % len(base_colors)]
    return color_map


# =========================================================
# 工具函数：把图片转成 base64，嵌入到 SVG 里
# =========================================================
def image_to_data_uri(img_path: Path) -> str:
    """
    将图片文件读成 base64 data URI，
    这样 SVG 文件内部就能直接嵌入原图，不依赖外部图片路径。
    """
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
# 工具函数：估算标签文字框大小
# =========================================================
def estimate_text_box(label: str, font_size: int):
    """
    粗略估算文字背景框大小。

    参数：
        label: 文字内容，例如 "pl80 0.91"
        font_size: 字体大小

    返回：
        text_w: 估算文字宽度
        text_h: 估算文字高度
    """
    text_w = int(len(label) * font_size * 0.55)
    text_h = int(font_size * 1.10)
    return text_w, text_h


# =========================================================
# 工具函数：判断两个矩形是否重叠
# =========================================================
def boxes_overlap(a, b):
    """
    判断两个矩形区域是否重叠。

    参数：
        a, b 格式都是 (x, y, w, h)

    返回：
        True  = 重叠
        False = 不重叠
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    return not (
        ax + aw <= bx or
        bx + bw <= ax or
        ay + ah <= by or
        by + bh <= ay
    )


# =========================================================
# 工具函数：给标签寻找一个尽量不重叠的位置
# =========================================================
def place_label_with_avoidance(
    x1, y1, x2, y2,
    text_w, text_h,
    width_real, height_real,
    used_label_boxes,
    prefer_outside=True,
    margin=1,
    line_gap=2,
):
    """
    给标签寻找一个尽量不重叠的位置。

    逻辑：
    1. 优先尝试放在框的上方
    2. 若重叠，则不断向下挪
    3. 如果上方不合适，再尝试框内顶部
    4. 最终返回背景框和文字的位置
    """
    label_w = text_w + 4
    label_h = text_h

    candidates = []

    # 方案1：优先放在框外上方
    if prefer_outside:
        candidates.append((x1, y1 - label_h - margin, "outside"))

    # 方案2：放在框内顶部
    candidates.append((x1, y1, "inside"))

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

                text_x = bg_x + 1
                text_y = bg_y + label_h - 2

                return bg_x, bg_y, text_x, text_y

            bg_y += label_h + line_gap

    # 如果上面都失败，则给一个尽量可见的位置
    bg_x = max(0, min(x1, width_real - label_w))
    bg_y = max(0, min(y1, height_real - label_h))
    used_label_boxes.append((bg_x, bg_y, label_w, label_h))

    text_x = bg_x + 1
    text_y = bg_y + label_h - 2

    return bg_x, bg_y, text_x, text_y


# =========================================================
# 工具函数：根据类别ID获取类别名称
# =========================================================
def get_class_name(cls_id: int) -> str:
    """
    根据类别ID返回类别名称。
    如果类别ID越界，则返回一个兜底名称，避免程序报错。
    """
    if 0 <= cls_id < len(CLASS_NAMES):
        return CLASS_NAMES[cls_id]
    return f"cls_{cls_id}"


# =========================================================
# 核心函数：构建单张 SVG
# =========================================================
def build_svg(img_path: Path, txt_path: Path, out_path: Path, class_colors: dict):
    """
    给一张图片 + 一个模型对应的 txt 检测结果，
    生成对应的 SVG 可视化结果。
    """
    # 读取原图尺寸
    with Image.open(img_path) as im:
        width_real, height_real = im.size

    # 把原图编码成 base64，嵌进 SVG
    image_data_uri = image_to_data_uri(img_path)

    # 用列表逐段拼接 SVG
    svg_parts = []

    # SVG 头部
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_real}" height="{height_real}" '
        f'viewBox="0 0 {width_real} {height_real}">'
    )

    # 嵌入原图
    svg_parts.append(
        f'  <image href="{image_data_uri}" x="0" y="0" '
        f'width="{width_real}" height="{height_real}"/>'
    )

    draw_count = 0

    # 已放置的标签区域，用于文字避让
    used_label_boxes = []

    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 预测格式：cls x y w h conf
            a = line.split()
            if len(a) < 6:
                continue

            cls_id = int(float(a[0]))
            x = float(a[1])
            y = float(a[2])
            w = float(a[3])
            h = float(a[4])
            conf = float(a[5])

            # 过滤类别
            if cls_id in SKIP_CLASS_IDS:
                continue

            # 过滤低置信度
            if conf < CONF_THRES:
                continue

            # YOLO归一化坐标 -> 像素坐标
            x_center = width_real * x
            y_center = height_real * y
            box_w = width_real * w
            box_h = height_real * h

            x1 = x_center - box_w / 2.0
            y1 = y_center - box_h / 2.0
            x2 = x_center + box_w / 2.0
            y2 = y_center + box_h / 2.0

            # 防止越界
            x1 = max(0, min(x1, width_real - 1))
            y1 = max(0, min(y1, height_real - 1))
            x2 = max(0, min(x2, width_real - 1))
            y2 = max(0, min(y2, height_real - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # 同类固定颜色
            box_color = class_colors.get(cls_id, "#FFFF00")

            # 类别编号 -> 类别名称
            class_name = get_class_name(cls_id)

            # 标签内容
            if SHOW_CONF:
                label = f"{class_name} {conf:.2f}"
            else:
                label = class_name

            # SVG 文字转义
            label_esc = escape(label)

            # 估算文字框大小
            text_w, text_h = estimate_text_box(label, FONT_SIZE)

            # 计算标签位置
            if TEXT_AVOID_OVERLAP:
                bg_x, bg_y, text_x, text_y = place_label_with_avoidance(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    text_w=text_w,
                    text_h=text_h,
                    width_real=width_real,
                    height_real=height_real,
                    used_label_boxes=used_label_boxes,
                    prefer_outside=True,
                    margin=LABEL_PADDING,
                    line_gap=LABEL_LINE_GAP,
                )
            else:
                # 紧贴框体模式
                outside = (y1 - text_h - LABEL_PADDING) >= 0

                if outside:
                    bg_x = x1
                    bg_y = y1 - text_h - LABEL_PADDING
                    text_x = x1 + 1
                    text_y = y1 - LABEL_PADDING - 2
                else:
                    bg_x = x1
                    bg_y = y1
                    text_x = x1 + 1
                    text_y = y1 + text_h - 2

            # 1) 绘制框体
            svg_parts.append(
                f'  <rect x="{x1:.2f}" y="{y1:.2f}" '
                f'width="{(x2 - x1):.2f}" height="{(y2 - y1):.2f}" '
                f'fill="none" stroke="{box_color}" '
                f'stroke-width="{BOX_STROKE_WIDTH}" '
                f'shape-rendering="geometricPrecision"/>'
            )

            # 2) 绘制标签背景
            svg_parts.append(
                f'  <rect x="{bg_x:.2f}" y="{bg_y:.2f}" '
                f'width="{text_w + 4:.2f}" height="{text_h:.2f}" '
                f'fill="{box_color}" fill-opacity="{LABEL_BG_OPACITY}"/>'
            )

            # 3) 绘制标签文字
            svg_parts.append(
                f'  <text x="{text_x:.2f}" y="{text_y:.2f}" '
                f'fill="{TEXT_COLOR}" '
                f'font-family="{FONT_FAMILY}" '
                f'font-size="{FONT_SIZE}" '
                # f'font-weight="normal">'
                f'font-weight="bold">'
                f'{label_esc}</text>'
            )

            draw_count += 1

    svg_parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

    print(f"✅ 已保存: {out_path.name}，画了 {draw_count} 个框")


# =========================================================
# 主函数
# =========================================================
def main():
    """
    主流程：
    1. 检查图像目录
    2. 扫描所有图片
    3. 遍历每个模型
    4. 为每张图生成对应的 SVG
    """
    if not image_dir.exists():
        print(f"❌ 图像目录不存在: {image_dir}")
        return

    # 生成类别颜色映射
    class_colors = generate_class_colors(NUM_CLASSES)

    # 收集图片
    img_files = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix in img_exts:
            img_files.append(p)
    img_files.sort()

    if not img_files:
        print(f"❌ 未找到图片: {image_dir}")
        return

    # 遍历模型
    for model_name, label_dir in model_dirs.items():
        if not label_dir.exists():
            print(f"⚠️ 模型目录不存在，跳过: {label_dir}")
            continue

        print(f"\n=== 正在处理模型: {model_name} ===")

        # 遍历图片
        for img_path in img_files:
            basename = img_path.stem
            txt_path = label_dir / f"{basename}.txt"

            out_name = f"{basename}_{model_name}.svg"
            out_path = save_dir / out_name

            build_svg(img_path, txt_path, out_path, class_colors)

    print("\n🎉 三个模型的 SVG 矢量标注图已全部生成")


# =========================================================
# 程序入口
# =========================================================
if __name__ == "__main__":
    main()