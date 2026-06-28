"""
Draw GT and Prediction Boxes for Occlusion Robustness Visualization
Vector + Full Image + Class Names + Class Colors Version
====================================================================

生成五张图：
1. Original + GT
2. YOLOv11 on Original
3. YOLOv11 on Occlusion
4. ECAFA-YOLO on Original
5. ECAFA-YOLO on Occlusion

同时生成一张 1x5 拼接图，方便论文展示。

说明：
- 底图仍然是位图，这是正常的；
- 检测框、标签文字、标题是矢量对象；
- 推荐论文中使用 PDF 文件；
- 不同类别使用不同颜色；
- 同一类别在不同图中保持同一种颜色；
- 当前不裁剪，直接在完整原图上绘制；
- 裁剪参数保留，后面需要裁剪时改 USE_CROP = True 即可。
"""

import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =====================================================
# 项目根目录
# =====================================================
# 当前脚本假设位于：
# D:\UKMJIAO\AlgorithmCodes\yolov11_jiao\回应审稿人2的鲁棒性测试\xxx.py
#
# parents[1] 回到：
# D:\UKMJIAO\AlgorithmCodes\yolov11_jiao

ROOT = Path(__file__).resolve().parents[1]


# =====================================================
# 需要绘制的图像编号
# =====================================================

IMAGE_ID = "91888"


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

OCCLUSION_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    "val_occlusion"
)

GT_LABEL_DIR = (
    DATA_ROOT /
    "labels" /
    "val"
)

YOLO_ORIGINAL_LABEL_DIR = (
    ROOT /
    "valsRobust" /
    "robust_tt100k130" /
    "YOLOv11_Original" /
    "labels"
)

YOLO_OCCLUSION_LABEL_DIR = (
    ROOT /
    "valsRobust" /
    "robust_tt100k130" /
    "YOLOv11_Occlusion" /
    "labels"
)

ECAFA_ORIGINAL_LABEL_DIR = (
    ROOT /
    "valsRobust" /
    "robust_tt100k130" /
    "ECAFA_YOLO_Original" /
    "labels"
)

ECAFA_OCCLUSION_LABEL_DIR = (
    ROOT /
    "valsRobust" /
    "robust_tt100k130" /
    "ECAFA_YOLO_Occlusion" /
    "labels"
)

OUTPUT_DIR = (
    ROOT /
    "valsRobust" /
    "robust_tt100k130" /
    "visual_cases"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 绘图参数
# =====================================================

CONF_THRESH = 0.25

DRAW_LABEL = True

BOX_LINEWIDTH = 0.5
LABEL_FONT_SIZE = 1.9
LABEL_BBOX_PAD = 0.05
LABEL_LINESPACING = 1.0

SAVE_PNG_PREVIEW = True
SAVE_PDF = True
SAVE_SVG = True

FALLBACK_COLOR = "#FF0000"


# =====================================================
# 可视化用 NMS 去除严重重叠框
# =====================================================
# 只用于论文画图展示，不影响正式验证结果。
# True  开启去重
# False 关闭去重
ENABLE_VIS_NMS = True

# IoU 超过这个阈值就认为是严重重叠，只保留置信度最高的框
VIS_NMS_IOU_THRESH = 0.70


# =====================================================
# 裁剪参数
# =====================================================
# 当前不裁剪，直接显示完整图像。
# 后续如果需要裁剪，只需要改：
# USE_CROP = True
#
# 注意：
# 这里的裁剪不是裁掉图像文件本身，
# 而是控制 Matplotlib 显示区域。
# 因此 YOLO 标签坐标不需要重新换算。

USE_CROP = True

CROP_LEFT_RATIO = 0.55322265625
CROP_RIGHT_RATIO = 0.326171875

CROP_TOP_RATIO = 0.390625
CROP_BOTTOM_RATIO = 0.48876953125

VIEW_TAG = "crop" if USE_CROP else "full"


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
    "pl120", "ph2.8", "w32", "pm15", "ph5", "pw3.2", "pl10", "il60", "w57",
    "pl100", "p16", "pl110", "w59", "w20", "ph2", "p9", "il100", "p19",
    "ph3.5", "pa10", "pcl", "pl35", "p15", "phcs", "w3", "pl25", "il110",
    "p1", "w46", "pn-2", "w63", "pm20", "i5", "il90", "w21", "p27", "pl50",
    "ph2.2", "pm2", "pw4"
]


# =====================================================
# 类别颜色
# =====================================================

def generate_class_colors(num_classes: int):
    """
    为每个类别生成固定颜色。
    相同类别在不同图中颜色一致。
    """
    base_colors = [
        "#FF0000",
        "#00AA00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00BFFF",
        "#FFA500",
        "#FF4D4F",
        "#0066FF",
        "#FF1493",
        "#7FFF00",
        "#FFD700",
        "#FF4500",
        "#00CED1",
        "#9400D3",
    ]

    color_map = {}

    for i in range(num_classes):
        color_map[i] = base_colors[i % len(base_colors)]

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

# 手动调整 ph4.5 类别颜色，避免黄底白字看不清
PH45_ID = CLASS_NAMES.index("ph4.5")
CLASS_COLOR_MAP[PH45_ID] = "#39FF14"


# =====================================================
# Matplotlib 全局参数
# =====================================================

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"


# =====================================================
# 基础工具函数
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
    """
    return CLASS_NAME_MAP.get(
        cls_id,
        str(cls_id)
    )


def get_class_color(cls_id):
    """
    根据类别 id 获取类别颜色。
    """
    return CLASS_COLOR_MAP.get(
        cls_id,
        FALLBACK_COLOR
    )


def get_label_text_color(hex_color):
    """
    根据标签背景色自动选择文字颜色。
    浅色背景用黑字，深色背景用白字。
    """
    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    luminance = (
        0.299 * r +
        0.587 * g +
        0.114 * b
    )

    if luminance > 150:
        return "black"

    return "white"


def bgr_to_rgb(img_bgr):
    """
    OpenCV BGR 转 Matplotlib RGB。
    """
    return cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )


# =====================================================
# 标签读取函数
# =====================================================

def read_gt_labels(label_path):
    """
    读取 GT 标签。

    GT YOLO 格式：
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

    预测 YOLO 格式：
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


# =====================================================
# 坐标与显示区域
# =====================================================

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


def compute_iou_xyxy(box_a, box_b):
    """
    计算两个 xyxy 框的 IoU。

    box_a, box_b:
        (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def remove_heavily_overlapped_boxes(
        boxes,
        img_bgr,
        iou_thresh=0.70):
    """
    去除严重重叠的预测框。

    说明：
    - 只用于论文可视化；
    - 不区分类别，class-agnostic；
    - 两个框 IoU 大于阈值时，只保留置信度更高的框；
    - GT 框不要调用这个函数。
    """
    if not ENABLE_VIS_NMS:
        return boxes

    if len(boxes) == 0:
        return boxes

    h, w = img_bgr.shape[:2]

    sorted_boxes = sorted(
        boxes,
        key=lambda b: b["conf"] if b["conf"] is not None else 1.0,
        reverse=True
    )

    kept_boxes = []

    for box in sorted_boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(
            box,
            img_w=w,
            img_h=h
        )

        current_xyxy = (
            x1,
            y1,
            x2,
            y2
        )

        keep = True

        for kept_box in kept_boxes:
            kx1, ky1, kx2, ky2 = yolo_to_xyxy(
                kept_box,
                img_w=w,
                img_h=h
            )

            kept_xyxy = (
                kx1,
                ky1,
                kx2,
                ky2
            )

            iou = compute_iou_xyxy(
                current_xyxy,
                kept_xyxy
            )

            if iou > iou_thresh:
                keep = False
                break

        if keep:
            kept_boxes.append(box)

    return kept_boxes


def print_nms_info(name, before_boxes, after_boxes):
    """
    打印可视化 NMS 前后的框数量。
    """
    print(
        f"{name:<24}: "
        f"{len(before_boxes)} -> {len(after_boxes)} "
        f"(removed {len(before_boxes) - len(after_boxes)})"
    )


def get_crop_region(img_bgr):
    """
    根据比例计算裁剪显示区域。

    注意：
    这里不真正裁剪原图像素，只控制 Matplotlib 显示范围。
    因此 YOLO 标签坐标不需要重新换算。
    """
    h, w = img_bgr.shape[:2]

    x_min = w * CROP_LEFT_RATIO
    x_max = w * (1 - CROP_RIGHT_RATIO)

    y_min = h * CROP_TOP_RATIO
    y_max = h * (1 - CROP_BOTTOM_RATIO)

    return x_min, x_max, y_min, y_max


def get_display_region(img_bgr, use_crop=False):
    """
    获取当前显示区域。

    use_crop=False:
        显示完整图像。

    use_crop=True:
        按 CROP_*_RATIO 显示裁剪区域。
    """
    h, w = img_bgr.shape[:2]

    if use_crop:
        return get_crop_region(img_bgr)

    return 0, w, 0, h


def setup_image_axis(ax, img_bgr, use_crop=USE_CROP):
    """
    显示图像并关闭坐标轴。
    """
    ax.imshow(
        bgr_to_rgb(img_bgr),
        interpolation="nearest"
    )

    x_min, x_max, y_min, y_max = get_display_region(
        img_bgr,
        use_crop=use_crop
    )

    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_ylim(
        y_max,
        y_min
    )

    ax.set_aspect("equal")
    ax.axis("off")


def box_intersects_region(x1, y1, x2, y2, region):
    """
    判断框是否与当前显示区域有交集。
    完全不在显示区域内的框不绘制。
    """
    x_min, x_max, y_min, y_max = region

    if x2 < x_min:
        return False

    if x1 > x_max:
        return False

    if y2 < y_min:
        return False

    if y1 > y_max:
        return False

    return True


# =====================================================
# 绘制函数
# =====================================================

def draw_vector_boxes(ax, img_bgr, boxes, mode="pred"):
    """
    用 Matplotlib 的 Rectangle 画矢量框。

    mode:
        gt   -> 画 GT 框
        pred -> 画预测框
    """
    h, w = img_bgr.shape[:2]

    display_region = get_display_region(
        img_bgr,
        use_crop=USE_CROP
    )

    x_min, x_max, y_min, y_max = display_region

    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(
            box,
            img_w=w,
            img_h=h
        )

        if not box_intersects_region(
            x1,
            y1,
            x2,
            y2,
            display_region
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

        if not DRAW_LABEL:
            continue

        if mode == "gt":
            label = cls_name
        else:
            label = f"{cls_name}\n{box['conf']:.2f}"

        text_x = max(
            x1,
            x_min + 2
        )

        if y1 < y_min + 16:
            text_y = max(
                y1 + 2,
                y_min + 2
            )
            va = "top"
        else:
            text_y = y1 - 2
            va = "bottom"

        label_text_color = get_label_text_color(
            box_color
        )

        ax.text(
            text_x,
            text_y,
            label,
            fontsize=LABEL_FONT_SIZE,
            color=label_text_color,
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


# =====================================================
# 保存函数
# =====================================================

def save_vector_single(
        img_bgr,
        boxes,
        mode,
        save_base,
        title=None):
    """
    保存单张矢量图。
    """
    x_min, x_max, y_min, y_max = get_display_region(
        img_bgr,
        use_crop=USE_CROP
    )

    display_w = x_max - x_min
    display_h = y_max - y_min

    dpi = 300

    fig_w = display_w / dpi
    fig_h = display_h / dpi

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h),
        dpi=dpi
    )

    setup_image_axis(
        ax,
        img_bgr,
        use_crop=USE_CROP
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
    """
    n = len(panel_items)

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
            use_crop=USE_CROP
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

if __name__ == "__main__":

    original_img_path = find_image(
        ORIGINAL_IMAGE_DIR,
        IMAGE_ID
    )

    occlusion_img_path = find_image(
        OCCLUSION_IMAGE_DIR,
        IMAGE_ID
    )

    if original_img_path is None:
        raise FileNotFoundError(
            f"Original image not found: {IMAGE_ID}"
        )

    if occlusion_img_path is None:
        raise FileNotFoundError(
            f"Occlusion image not found: {IMAGE_ID}"
        )

    original_img = cv2.imread(
        str(original_img_path)
    )

    occlusion_img = cv2.imread(
        str(occlusion_img_path)
    )

    if original_img is None:
        raise RuntimeError(
            f"Failed to read original image: {original_img_path}"
        )

    if occlusion_img is None:
        raise RuntimeError(
            f"Failed to read occlusion image: {occlusion_img_path}"
        )

    print(f"Original image shape before resize : {original_img.shape}")
    print(f"Occlusion image shape before resize: {occlusion_img.shape}")

    # =====================================================
    # 统一 Occlusion 图尺寸到 Original 图尺寸
    # =====================================================
    # 一般 val_occlusion 和 original 尺寸应该一致。
    # 这里保留 resize 检查，防止尺寸不一致导致框和字体比例异常。
    # YOLO 标签是归一化坐标，因此 resize 后框的位置仍然正确。

    if occlusion_img.shape[:2] != original_img.shape[:2]:

        print(
            f"[Info] Resize occlusion image from {occlusion_img.shape[:2]} "
            f"to {original_img.shape[:2]}"
        )

        occlusion_img = cv2.resize(
            occlusion_img,
            (
                original_img.shape[1],
                original_img.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    print(f"Occlusion image shape after resize : {occlusion_img.shape}")

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

    yolo_occlusion_boxes = read_pred_labels(
        YOLO_OCCLUSION_LABEL_DIR /
        f"{IMAGE_ID}.txt",
        conf_thresh=CONF_THRESH
    )

    ecafa_original_boxes = read_pred_labels(
        ECAFA_ORIGINAL_LABEL_DIR /
        f"{IMAGE_ID}.txt",
        conf_thresh=CONF_THRESH
    )

    ecafa_occlusion_boxes = read_pred_labels(
        ECAFA_OCCLUSION_LABEL_DIR /
        f"{IMAGE_ID}.txt",
        conf_thresh=CONF_THRESH
    )

    # =====================================================
    # 可视化用 NMS：去除严重重叠预测框
    # =====================================================
    # 注意：
    # 这里只处理预测框，不处理 GT。
    # 这一步只影响论文画图，不影响正式验证指标。

    yolo_original_boxes_before = yolo_original_boxes
    yolo_occlusion_boxes_before = yolo_occlusion_boxes
    ecafa_original_boxes_before = ecafa_original_boxes
    ecafa_occlusion_boxes_before = ecafa_occlusion_boxes

    yolo_original_boxes = remove_heavily_overlapped_boxes(
        yolo_original_boxes,
        original_img,
        iou_thresh=VIS_NMS_IOU_THRESH
    )

    yolo_occlusion_boxes = remove_heavily_overlapped_boxes(
        yolo_occlusion_boxes,
        occlusion_img,
        iou_thresh=VIS_NMS_IOU_THRESH
    )

    ecafa_original_boxes = remove_heavily_overlapped_boxes(
        ecafa_original_boxes,
        original_img,
        iou_thresh=VIS_NMS_IOU_THRESH
    )

    ecafa_occlusion_boxes = remove_heavily_overlapped_boxes(
        ecafa_occlusion_boxes,
        occlusion_img,
        iou_thresh=VIS_NMS_IOU_THRESH
    )

    print("=" * 60)
    print("Visualization NMS")
    print(f"Enable NMS                : {ENABLE_VIS_NMS}")
    print(f"NMS IoU threshold         : {VIS_NMS_IOU_THRESH}")

    print_nms_info(
        "YOLO Original",
        yolo_original_boxes_before,
        yolo_original_boxes
    )

    print_nms_info(
        "YOLO Occlusion",
        yolo_occlusion_boxes_before,
        yolo_occlusion_boxes
    )

    print_nms_info(
        "ECAFA Original",
        ecafa_original_boxes_before,
        ecafa_original_boxes
    )

    print_nms_info(
        "ECAFA Occlusion",
        ecafa_occlusion_boxes_before,
        ecafa_occlusion_boxes
    )

    print("=" * 60)

    print("=" * 60)
    print(f"Image ID                  : {IMAGE_ID}")
    print(f"Original image path       : {original_img_path}")
    print(f"Occlusion image path      : {occlusion_img_path}")
    print(f"Number of classes         : {len(CLASS_NAMES)}")
    print(f"GT boxes                  : {len(gt_boxes)}")
    print(f"YOLO Original boxes       : {len(yolo_original_boxes)}")
    print(f"YOLO Occlusion boxes      : {len(yolo_occlusion_boxes)}")
    print(f"ECAFA Original boxes      : {len(ecafa_original_boxes)}")
    print(f"ECAFA Occlusion boxes     : {len(ecafa_occlusion_boxes)}")
    print(f"Use crop                  : {USE_CROP}")
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
            "name": "03_YOLOv11_occlusion",
            "img": occlusion_img,
            "boxes": yolo_occlusion_boxes,
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
            "name": "05_ECAFA_occlusion",
            "img": occlusion_img,
            "boxes": ecafa_occlusion_boxes,
            "mode": "pred",
            "title": None,
        },
    ]

    for item in single_items:

        save_base = (
            OUTPUT_DIR /
            f"{IMAGE_ID}_{item['name']}_{VIEW_TAG}_vector_class_color"
        )

        save_vector_single(
            img_bgr=item["img"],
            boxes=item["boxes"],
            mode=item["mode"],
            save_base=save_base,
            title=item["title"]
        )

    # =====================================================
    # 保存 1x5 矢量拼接图
    # =====================================================

    panel_items = [
        {
            "img": original_img,
            "boxes": gt_boxes,
            "mode": "gt",
            "title": "Original + GT",
        },
        {
            "img": original_img,
            "boxes": yolo_original_boxes,
            "mode": "pred",
            "title": "YOLOv11 Original",
        },
        {
            "img": occlusion_img,
            "boxes": yolo_occlusion_boxes,
            "mode": "pred",
            "title": "YOLOv11 Occlusion",
        },
        {
            "img": original_img,
            "boxes": ecafa_original_boxes,
            "mode": "pred",
            "title": "ECAFA-YOLO Original",
        },
        {
            "img": occlusion_img,
            "boxes": ecafa_occlusion_boxes,
            "mode": "pred",
            "title": "ECAFA-YOLO Occlusion",
        },
    ]

    montage_base = (
        OUTPUT_DIR /
        f"{IMAGE_ID}_5cols_montage_{VIEW_TAG}_vector_class_color"
    )

    save_vector_montage(
        panel_items=panel_items,
        save_base=montage_base
    )

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)
    print(f"Output dir: {OUTPUT_DIR}")