
"""
Draw GT and Prediction Boxes for Robustness Visualization
Vector + Crop + Class Names + Class Colors Version
=========================================================

生成五张图：
1. Original + GT
2. YOLOv11 on Original
3. YOLOv11 on Rain
4. ECAFA-YOLO on Original
5. ECAFA-YOLO on Rain

仅生成五张单独图，不再生成 1x5 拼接图。

重要说明：
- 底图仍然是位图，这是正常的。
- 检测框、标签文字、标题是矢量对象。
- 推荐论文中使用 PDF 文件。
- 不同类别使用不同颜色。
- 同一类别在不同图中保持同一种颜色。
- 当前裁剪区域：
    左上角坐标: (705, 740)
    右下角坐标: (1500, 880)
"""

import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =====================================================
# 需要绘制的图像编号
# =====================================================

IMAGE_ID = "12435"   # 改这里，例如 "28701", "12435"


# =====================================================
# 路径配置
# =====================================================

DATA_ROOT = Path(
    r"F:\DataSets\tt100k\yolojack"
)

ORIGINAL_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    "val"
)

RAIN_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    "val_rain"
)

GT_LABEL_DIR = (
    DATA_ROOT /
    "labels" /
    "val"
)

YOLO_ORIGINAL_LABEL_DIR = Path(
    r"../valsRobust/robust_tt100k130/YOLOv11_Original/labels"
)

YOLO_RAIN_LABEL_DIR = Path(
    r"../valsRobust/robust_tt100k130/YOLOv11_Rain/labels"
)

ECAFA_ORIGINAL_LABEL_DIR = Path(
    r"../valsRobust/robust_tt100k130/ECAFA_YOLO_Original/labels"
)

ECAFA_RAIN_LABEL_DIR = Path(
    r"../valsRobust/robust_tt100k130/ECAFA_YOLO_Rain/labels"
)

OUTPUT_DIR = Path(
    r"../valsRobust/robust_tt100k130/visual_cases"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 绘图参数
# =====================================================

CONF_THRESH = 0.25          # 预测框置信度阈值
DRAW_LABEL = True           # 是否显示类别和置信度

BOX_LINEWIDTH = 1.5         # 矢量框线宽，单位是 pt
LABEL_FONT_SIZE = 5.8      # 类别文字大小，太挤可以改成 4.8 或 5.0
LABEL_BBOX_PAD = 0.05       # 标签背景框贴边程度，越小越贴文字
LABEL_LINESPACING = 0.95    # 两行文字行距，避免字母尾部压到置信度

SAVE_PNG_PREVIEW = True     # 是否额外保存 PNG 预览图
SAVE_PDF = True             # 是否保存 PDF
SAVE_SVG = True             # 是否保存 SVG

# 备用颜色。如果类别 id 超出范围，就使用这个颜色
FALLBACK_COLOR = "#FF0000"


# =====================================================
# 裁剪参数
# =====================================================
# 固定像素裁剪区域：
# 左上角: (705, 730)
# 右下角: (1500, 870)
#
# 注意：
# 这里不真正裁剪原图像素，只控制 Matplotlib 的显示范围。
# 因此 YOLO 标签坐标不需要重新换算。

CROP_X_MIN = 705
CROP_Y_MIN = 730
CROP_X_MAX = 1500
CROP_Y_MAX = 870


# =====================================================
# TT100K 类别名称
# =====================================================

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


# =====================================================
# 为每个类别生成固定颜色
# =====================================================

def generate_class_colors(num_classes: int):
    """
    为每个类别生成固定颜色。
    相同类别在不同图中颜色一致。
    """
    base_colors = [
        "#FF0000", "#00AA00", "#0000FF",
        "#FFFF00", "#FF00FF", "#00BFFF",
        "#FFA500", "#FF4D4F", "#0066FF",
        "#FF1493", "#7FFF00", "#FFD700",
        "#FF4500", "#00CED1", "#9400D3",
    ]

    color_map = {}

    for i in range(num_classes):
        color_map[i] = base_colors[i % len(base_colors)]

    # 你指定的重点类别颜色
    # 注意：只有当类别 id 98 存在时才有意义
    if num_classes > 98:
        color_map[98] = "#E63946"

    return color_map


CLASS_NAME_MAP = {
    i: name
    for i, name in enumerate(CLASS_NAMES)
}

CLASS_COLOR_MAP = generate_class_colors(
    len(CLASS_NAMES)
)
# =====================================================
# 手动调整部分类别颜色
# =====================================================

PH45_ID = CLASS_NAMES.index("ph4.5")
PB_ID = CLASS_NAMES.index("pb")

# ph4.5 改成亮绿色：
# 1. 和蓝底区别明显
# 2. 和旁边红色框区别明显
# 3. 不使用黄色，避免白字看不清
CLASS_COLOR_MAP[PH45_ID] = "#39FF14"

# pb 原来的青色/蓝绿色容易和图像底部背景混在一起，
# 这里单独改成醒目的橙色，和青色背景对比更明显，黑色加粗文字也更清楚。
CLASS_COLOR_MAP[PB_ID] = "#FF8C00"

# =====================================================
# Matplotlib 全局参数
# =====================================================
# 让 PDF / SVG 中的文字尽量保持为可编辑文本

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"


# =====================================================
# 工具函数
# =====================================================

def find_image(image_dir, stem):
    """
    根据图像编号寻找图片。
    """
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def get_class_name(cls_id):
    """
    根据类别 id 获取类别名称。
    如果找不到，就退回显示类别 id。
    """
    return CLASS_NAME_MAP.get(cls_id, str(cls_id))


def get_class_color(cls_id):
    """
    根据类别 id 获取类别颜色。
    如果找不到，就使用备用颜色。
    """
    return CLASS_COLOR_MAP.get(cls_id, FALLBACK_COLOR)

def get_label_text_color(hex_color):
    """
    根据标签背景色自动选择文字颜色。
    浅色背景用黑字，深色背景用白字。
    """
    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance > 150:
        return "black"
    else:
        return "white"

def read_gt_labels(label_path):
    """
    读取 GT 标签。

    GT YOLO 格式:
    cls x_center y_center width height
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            boxes.append(
                {
                    "cls": cls_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": None,
                }
            )

    return boxes


def read_pred_labels(label_path, conf_thresh=0.25):
    """
    读取预测标签。

    预测 YOLO 格式:
    cls x_center y_center width height conf
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 6:
                continue

            cls_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5])

            if conf < conf_thresh:
                continue

            boxes.append(
                {
                    "cls": cls_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                }
            )

    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    """
    YOLO 归一化格式转 xyxy 像素坐标。
    """
    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def bgr_to_rgb(img_bgr):
    """
    OpenCV BGR 转 Matplotlib RGB。
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_crop_region(img_bgr):
    """
    根据固定像素坐标计算裁剪显示区域。

    当前区域：
        左上角: (705, 740)
        右下角: (1500, 880)

    注意：
    这里不真正裁剪原图像素，只控制 Matplotlib 的显示范围。
    因此 YOLO 标签坐标不需要重新换算。
    """
    h, w = img_bgr.shape[:2]

    x_min = max(0, min(CROP_X_MIN, w - 1))
    x_max = max(0, min(CROP_X_MAX, w))

    y_min = max(0, min(CROP_Y_MIN, h - 1))
    y_max = max(0, min(CROP_Y_MAX, h))

    if x_max <= x_min:
        raise ValueError(
            f"Invalid crop x range: x_min={x_min}, x_max={x_max}, image width={w}"
        )

    if y_max <= y_min:
        raise ValueError(
            f"Invalid crop y range: y_min={y_min}, y_max={y_max}, image height={h}"
        )

    return x_min, x_max, y_min, y_max


def setup_image_axis(ax, img_bgr, use_crop=True):
    """
    显示图像并关闭坐标轴。

    use_crop=True:
        只显示裁剪后的区域。
    """
    h, w = img_bgr.shape[:2]

    ax.imshow(
        bgr_to_rgb(img_bgr),
        interpolation="nearest"
    )

    if use_crop:
        x_min, x_max, y_min, y_max = get_crop_region(img_bgr)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    ax.set_aspect("equal")
    ax.axis("off")


def box_intersects_crop(x1, y1, x2, y2, crop_region):
    """
    判断框是否与裁剪区域有交集。
    完全不在裁剪区域内的框不绘制。
    """
    x_min, x_max, y_min, y_max = crop_region

    if x2 < x_min:
        return False

    if x1 > x_max:
        return False

    if y2 < y_min:
        return False

    if y1 > y_max:
        return False

    return True


def draw_vector_boxes(ax, img_bgr, boxes, mode="pred"):
    """
    用 Matplotlib 的 Rectangle 画矢量框。

    不同类别使用不同颜色。

    mode:
        "gt"   -> 画 GT 框
        "pred" -> 画预测框
    """
    h, w = img_bgr.shape[:2]

    crop_region = get_crop_region(img_bgr)
    x_min, x_max, y_min, y_max = crop_region

    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(
            box,
            img_w=w,
            img_h=h
        )

        # 如果框完全不在裁剪区域内，就不绘制
        if not box_intersects_crop(
            x1,
            y1,
            x2,
            y2,
            crop_region
        ):
            continue

        cls_id = box["cls"]
        cls_name = get_class_name(cls_id)
        box_color = get_class_color(cls_id)

        box_w = x2 - x1
        box_h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            box_w,
            box_h,
            linewidth=BOX_LINEWIDTH,
            edgecolor=box_color,
            facecolor="none",
            clip_on=True
        )

        ax.add_patch(rect)

        if DRAW_LABEL:
            if mode == "gt":
                label = cls_name
            else:
                label = f"{cls_name}\n{box['conf']:.2f}"

            # 标签位置尽量放在裁剪区域内
            text_x = max(x1, x_min + 2)

            # 如果框上沿太靠近裁剪区域顶部，就把标签放到框内
            if y1 < y_min + 16:
                text_y = max(y1 + 2, y_min + 2)
                va = "top"
            else:
                text_y = y1 - 2
                va = "bottom"

            # ax.text(
            #     text_x,
            #     text_y,
            #     label,
            #     fontsize=LABEL_FONT_SIZE,
            #     color="white",
            #     verticalalignment=va,
            #     horizontalalignment="left",
            #     clip_on=True,
            #     bbox=dict(
            #         facecolor=box_color,
            #         edgecolor=box_color,
            #         boxstyle="square,pad=0.18",
            #         linewidth=0.0
            #     )
            # )
            # 标签文本统一使用黑色，并加粗，避免不同背景色导致可读性不一致。
            label_text_color = "black"

            ax.text(
                text_x,
                text_y,
                label,
                fontsize=LABEL_FONT_SIZE,
                color=label_text_color,
                fontweight="bold",
                verticalalignment=va,
                horizontalalignment="left",
                multialignment="left",
                linespacing=LABEL_LINESPACING,
                clip_on=True,
                bbox=dict(
                    facecolor=box_color,
                    edgecolor=box_color,
                    boxstyle=f"square,pad={LABEL_BBOX_PAD}",
                    linewidth=0.0
                )
            )


def save_vector_single(
    img_bgr,
    boxes,
    mode,
    save_base,
    title=None
):
    """
    保存单张矢量图。
    """
    x_min, x_max, y_min, y_max = get_crop_region(img_bgr)

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    dpi = 300

    fig_w = crop_w / dpi
    fig_h = crop_h / dpi

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h),
        dpi=dpi
    )

    setup_image_axis(
        ax,
        img_bgr,
        use_crop=True
    )

    draw_vector_boxes(
        ax=ax,
        img_bgr=img_bgr,
        boxes=boxes,
        mode=mode
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=10,
            pad=4
        )

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1
    )

    if SAVE_PDF:
        pdf_path = save_base.with_suffix(".pdf")
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0
        )
        print(f"[OK] Saved PDF: {pdf_path}")

    if SAVE_SVG:
        svg_path = save_base.with_suffix(".svg")
        fig.savefig(
            svg_path,
            bbox_inches="tight",
            pad_inches=0
        )
        print(f"[OK] Saved SVG: {svg_path}")

    if SAVE_PNG_PREVIEW:
        png_path = save_base.with_suffix(".png")
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )
        print(f"[OK] Saved PNG preview: {png_path}")

    plt.close(fig)


def save_vector_montage(panel_items, save_base):
    """
    保存 1x5 矢量拼接图。

    panel_items:
        [
            {
                "img": img_bgr,
                "boxes": boxes,
                "mode": "gt" or "pred",
                "title": title,
            },
            ...
        ]
    """
    n = len(panel_items)

    # 横向 5 图展示，适合论文
    fig_w = 15
    fig_h = 3.2

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_w, fig_h),
        dpi=300
    )

    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, panel_items):
        img_bgr = item["img"]
        boxes = item["boxes"]
        mode = item["mode"]
        title = item["title"]

        setup_image_axis(
            ax,
            img_bgr,
            use_crop=True
        )

        draw_vector_boxes(
            ax=ax,
            img_bgr=img_bgr,
            boxes=boxes,
            mode=mode
        )

        ax.set_title(
            title,
            fontsize=10,
            pad=5
        )

    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.01,
        top=0.88,
        wspace=0.02
    )

    if SAVE_PDF:
        pdf_path = save_base.with_suffix(".pdf")
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.02
        )
        print(f"[OK] Saved montage PDF: {pdf_path}")

    if SAVE_SVG:
        svg_path = save_base.with_suffix(".svg")
        fig.savefig(
            svg_path,
            bbox_inches="tight",
            pad_inches=0.02
        )
        print(f"[OK] Saved montage SVG: {svg_path}")

    if SAVE_PNG_PREVIEW:
        png_path = save_base.with_suffix(".png")
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02
        )
        print(f"[OK] Saved montage PNG preview: {png_path}")

    plt.close(fig)


# =====================================================
# 主程序
# =====================================================

original_img_path = find_image(
    ORIGINAL_IMAGE_DIR,
    IMAGE_ID
)

rain_img_path = find_image(
    RAIN_IMAGE_DIR,
    IMAGE_ID
)

if original_img_path is None:
    raise FileNotFoundError(
        f"Original image not found: {IMAGE_ID}"
    )

if rain_img_path is None:
    raise FileNotFoundError(
        f"Rain image not found: {IMAGE_ID}"
    )

original_img = cv2.imread(
    str(original_img_path)
)

rain_img = cv2.imread(
    str(rain_img_path)
)

if original_img is None:
    raise RuntimeError(
        f"Failed to read original image: {original_img_path}"
    )

if rain_img is None:
    raise RuntimeError(
        f"Failed to read rain image: {rain_img_path}"
    )
print(f"Original image shape before resize: {original_img.shape}")
print(f"Rain image shape before resize    : {rain_img.shape}")

# =====================================================
# 统一 rain 图尺寸到 original 图尺寸
# =====================================================
# 原因：
# rain 图是 1024x1024，original 图是 2048x2048。
# 如果不统一尺寸，Matplotlib 保存矢量图时，rain 图画布更小，
# 但框线宽和字体大小仍然是固定 pt 单位，所以看起来会特别大。
#
# YOLO 标签是归一化坐标，因此 resize 后框的位置仍然正确。

if rain_img.shape[:2] != original_img.shape[:2]:
    print(
        f"[Info] Resize rain image from {rain_img.shape[:2]} "
        f"to {original_img.shape[:2]}"
    )

    rain_img = cv2.resize(
        rain_img,
        (
            original_img.shape[1],  # width
            original_img.shape[0]   # height
        ),
        interpolation=cv2.INTER_LINEAR
    )

print(f"Rain image shape after resize     : {rain_img.shape}")
# =====================================================
# 读取标签
# =====================================================

gt_boxes = read_gt_labels(
    GT_LABEL_DIR /
    f"{IMAGE_ID}.txt"
)

yolo_original_boxes = read_pred_labels(
    YOLO_ORIGINAL_LABEL_DIR /
    f"{IMAGE_ID}.txt",
    conf_thresh=CONF_THRESH
)

yolo_rain_boxes = read_pred_labels(
    YOLO_RAIN_LABEL_DIR /
    f"{IMAGE_ID}.txt",
    conf_thresh=CONF_THRESH
)

ecafa_original_boxes = read_pred_labels(
    ECAFA_ORIGINAL_LABEL_DIR /
    f"{IMAGE_ID}.txt",
    conf_thresh=CONF_THRESH
)

ecafa_rain_boxes = read_pred_labels(
    ECAFA_RAIN_LABEL_DIR /
    f"{IMAGE_ID}.txt",
    conf_thresh=CONF_THRESH
)


print("=" * 60)
print(f"Image ID: {IMAGE_ID}")
print(f"Original image path  : {original_img_path}")
print(f"Rain image path      : {rain_img_path}")
print(f"Number of classes    : {len(CLASS_NAMES)}")
print(f"GT boxes             : {len(gt_boxes)}")
print(f"YOLO Original boxes  : {len(yolo_original_boxes)}")
print(f"YOLO Rain boxes      : {len(yolo_rain_boxes)}")
print(f"ECAFA Original boxes : {len(ecafa_original_boxes)}")
print(f"ECAFA Rain boxes     : {len(ecafa_rain_boxes)}")
print("=" * 60)


# =====================================================
# 保存五张单图
# =====================================================

single_items = [
    {
        "name": "01_original_GT",
        "img": original_img,
        "boxes": gt_boxes,
        "mode": "gt",
        "title": None,
    },
    {
        "name": "02_YOLOv11_original",
        "img": original_img,
        "boxes": yolo_original_boxes,
        "mode": "pred",
        "title": None,
    },
    {
        "name": "03_YOLOv11_rain",
        "img": rain_img,
        "boxes": yolo_rain_boxes,
        "mode": "pred",
        "title": None,
    },
    {
        "name": "04_ECAFA_original",
        "img": original_img,
        "boxes": ecafa_original_boxes,
        "mode": "pred",
        "title": None,
    },
    {
        "name": "05_ECAFA_rain",
        "img": rain_img,
        "boxes": ecafa_rain_boxes,
        "mode": "pred",
        "title": None,
    },
]

for item in single_items:
    save_base = OUTPUT_DIR / f"{IMAGE_ID}_{item['name']}_pixelcrop_up30_blackbold_vector_class_color"

    save_vector_single(
        img_bgr=item["img"],
        boxes=item["boxes"],
        mode=item["mode"],
        save_base=save_base,
        title=item["title"]
    )


# =====================================================
# 不再保存 1x5 拼接图
# =====================================================
# 说明：
# 你现在只需要五张单独图片，因此这里不再调用 save_vector_montage。


print()
print("=" * 60)
print("Finished")
print("=" * 60)