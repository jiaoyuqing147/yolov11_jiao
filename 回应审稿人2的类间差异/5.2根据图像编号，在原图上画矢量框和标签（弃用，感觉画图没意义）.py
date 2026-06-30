import os

# =====================================================
# Windows 下防止 OpenMP 冲突
# =====================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")

# =====================================================
# 重要：尽量保证 PDF/SVG 中文字和框体为矢量
# =====================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# 尽量不压缩 PDF 中的嵌入图像
matplotlib.rcParams["pdf.compression"] = 0

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =====================================================
# 1. 路径配置
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_YAML = PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "tt100k_desk_130.yaml"

GT_LABEL_DIR = Path(r"F:\DataSets\tt100k\yolojack\labels\val")
IMG_DIR = Path(r"F:\DataSets\tt100k\yolojack\images\val")

CONFUSION_ROOT = PROJECT_ROOT / "vals_error_analysis" / "tt100k_confusion_full"

YOLO_PRED_DIR = CONFUSION_ROOT / "YOLOv11_baseline" / "exp" / "labels"
KD_PRED_DIR = CONFUSION_ROOT / "ECAFA_YOLO_KD" / "exp" / "labels"

OUT_DIR = SCRIPT_DIR / "vector_original_case_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. 你从代码 1 的 CSV 里挑选编号，填在这里
# =====================================================

TARGET_STEMS = [
    # 示例：
    "15613",
    # "000123",
]

DRAW_CONF_THRES = 0.25
# =====================================================
# 3. 类别与显示设置
# =====================================================

CLASS_GROUP = ["pl40", "pl50", "pl60", "pl80", "pl100"]

# 是否只画 CLASS_GROUP 里的类别
DRAW_SELECTED_CLASSES_ONLY = True

# 是否显示置信度
SHOW_CONF = True

# 是否输出 PNG 预览图
# 论文优先用 PDF/SVG，PNG 只是检查用
SAVE_PNG_PREVIEW = False

# 导出时的 DPI。
# 为了不缩小原图，figure size 会按 原图像素 / EXPORT_DPI 来设置。
EXPORT_DPI = 100


# =====================================================
# 4. 裁剪开关：默认关闭，即在完整原图上画
# =====================================================

USE_CROP = False

# 如果后面你想裁剪，再打开 USE_CROP = True
# 裁剪坐标为像素坐标：left, top, right, bottom
CROP_LEFT = 0
CROP_TOP = 0
CROP_RIGHT = None
CROP_BOTTOM = None

# 每张图单独裁剪，key 是 stem
CASE_SPECIFIC_CROP_PIXELS = {
    # "12435": (860, 800, 1410, 1350),
}


# =====================================================
# 5. 样式设置
# =====================================================

GT_COLOR = "#00A000"
YOLO_COLOR = "#D62728"
OURS_COLOR = "#1F77B4"

BOX_LINEWIDTH = 2.2
LABEL_FONT_SIZE = 14

# 是否在图左上角加模型名
ADD_MODEL_TITLE = True
MODEL_TITLE_FONT_SIZE = 20


# =====================================================
# 6. 读取类别名称
# =====================================================

def load_names_from_yaml(yaml_path):
    import yaml

    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到 yaml: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id_to_name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError("yaml 中 names 格式不支持")

    name_to_id = {v: k for k, v in id_to_name.items()}

    return id_to_name, name_to_id


# =====================================================
# 7. 读取标签
# =====================================================

def read_gt_file(path):
    items = []

    if not path.exists():
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            items.append(
                {
                    "cls": int(float(parts[0])),
                    "box": list(map(float, parts[1:5])),
                    "conf": 1.0,
                }
            )

    return items


def read_pred_file(path):
    items = []

    if not path.exists():
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            box = list(map(float, parts[1:5]))

            if len(parts) >= 6:
                conf = float(parts[5])
            else:
                conf = 1.0

            items.append(
                {
                    "cls": cls_id,
                    "box": box,
                    "conf": conf,
                }
            )

    return items


# =====================================================
# 8. 图片与坐标工具
# =====================================================

def find_image_by_stem(img_dir, stem):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]

    for ext in exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def yolo_box_to_pixel_float(box, w, h):
    x, y, bw, bh = box

    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h

    return x1, y1, x2, y2


def get_crop_box(stem, image):
    w, h = image.size

    if not USE_CROP:
        return 0, 0, w, h

    if stem in CASE_SPECIFIC_CROP_PIXELS:
        left, top, right, bottom = CASE_SPECIFIC_CROP_PIXELS[stem]
    else:
        left = CROP_LEFT
        top = CROP_TOP
        right = CROP_RIGHT if CROP_RIGHT is not None else w
        bottom = CROP_BOTTOM if CROP_BOTTOM is not None else h

    left = int(max(0, min(w - 1, left)))
    top = int(max(0, min(h - 1, top)))
    right = int(max(left + 1, min(w, right)))
    bottom = int(max(top + 1, min(h, bottom)))

    return left, top, right, bottom


def filter_items(items, selected_ids):
    if not DRAW_SELECTED_CLASSES_ONLY:
        return items

    return [x for x in items if x["cls"] in selected_ids]


# =====================================================
# 9. 矢量标注绘制
# =====================================================

def draw_vector_annotations_on_original(
    image_path,
    items,
    id_to_name,
    selected_ids,
    model_title,
    box_color,
    save_base,
    stem
):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    crop_box = get_crop_box(stem, image)
    left, top, right, bottom = crop_box

    crop_img = image.crop((left, top, right, bottom)).convert("RGB")
    crop_arr = np.asarray(crop_img)

    crop_w = right - left
    crop_h = bottom - top

    # 核心：画布尺寸按像素比例设置，不把图压成小图
    fig_w = crop_w / EXPORT_DPI
    fig_h = crop_h / EXPORT_DPI

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=EXPORT_DPI)

    # 让坐标轴完全铺满画布，不留白边
    ax = fig.add_axes([0, 0, 1, 1])

    # interpolation='none' + resample=False：避免显示时重新插值
    ax.imshow(
        crop_arr,
        interpolation="none",
        resample=False
    )

    ax.set_xlim(0, crop_w)
    ax.set_ylim(crop_h, 0)
    ax.axis("off")

    draw_items = filter_items(items, selected_ids)

    for item in draw_items:
        cls_id = item["cls"]
        box = item["box"]
        conf = item.get("conf", None)

        x1, y1, x2, y2 = yolo_box_to_pixel_float(box, orig_w, orig_h)

        # 转换到 crop 坐标
        x1c = x1 - left
        y1c = y1 - top
        x2c = x2 - left
        y2c = y2 - top

        # 如果框完全不在裁剪区域内，跳过
        if x2c <= 0 or y2c <= 0 or x1c >= crop_w or y1c >= crop_h:
            continue

        # 裁剪到显示区域
        x1v = max(0, min(crop_w, x1c))
        y1v = max(0, min(crop_h, y1c))
        x2v = max(0, min(crop_w, x2c))
        y2v = max(0, min(crop_h, y2c))

        rect = patches.Rectangle(
            (x1v, y1v),
            x2v - x1v,
            y2v - y1v,
            linewidth=BOX_LINEWIDTH,
            edgecolor=box_color,
            facecolor="none",
            joinstyle="miter"
        )
        ax.add_patch(rect)

        cls_name = id_to_name.get(cls_id, str(cls_id))

        if SHOW_CONF and conf is not None and conf < 0.999:
            label = f"{cls_name} {conf:.2f}"
        else:
            label = cls_name

        # 标签位置：优先放框上方，不够就放框内/下方
        label_x = x1v
        label_y = y1v - 3

        if label_y < LABEL_FONT_SIZE + 2:
            label_y = y1v + LABEL_FONT_SIZE + 2

        ax.text(
            label_x,
            label_y,
            label,
            fontsize=LABEL_FONT_SIZE,
            color="white",
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="square,pad=0.18",
                facecolor=box_color,
                edgecolor=box_color,
                linewidth=0.6
            )
        )

    if ADD_MODEL_TITLE:
        ax.text(
            12,
            28,
            model_title,
            fontsize=MODEL_TITLE_FONT_SIZE,
            color="black",
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="square,pad=0.25",
                facecolor="white",
                edgecolor="white",
                alpha=0.90
            )
        )

    pdf_path = save_base.with_suffix(".pdf")
    svg_path = save_base.with_suffix(".svg")

    # 不使用 bbox_inches='tight'，避免重新裁切和缩放
    fig.savefig(
        pdf_path,
        format="pdf",
        dpi=EXPORT_DPI,
        pad_inches=0
    )

    fig.savefig(
        svg_path,
        format="svg",
        dpi=EXPORT_DPI,
        pad_inches=0
    )

    png_path = None

    if SAVE_PNG_PREVIEW:
        png_path = save_base.with_suffix(".png")

        fig.savefig(
            png_path,
            format="png",
            dpi=EXPORT_DPI,
            pad_inches=0
        )

    plt.close(fig)

    return pdf_path, svg_path, png_path, crop_box


# =====================================================
# 10. 主程序
# =====================================================

if __name__ == "__main__":

    print("\n=====================================================")
    print("Generate Original-size Vector Annotation Figures")
    print("=====================================================")
    print(f"IMG_DIR: {IMG_DIR}")
    print(f"GT_LABEL_DIR: {GT_LABEL_DIR}")
    print(f"YOLO_PRED_DIR: {YOLO_PRED_DIR}")
    print(f"KD_PRED_DIR: {KD_PRED_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"USE_CROP: {USE_CROP}")
    print("=====================================================\n")

    if len(TARGET_STEMS) == 0:
        raise ValueError(
            "TARGET_STEMS 为空。请先从 selected_case_ids.csv 中挑选图像编号，填入 TARGET_STEMS。"
        )

    for p in [DATA_YAML, IMG_DIR, GT_LABEL_DIR, YOLO_PRED_DIR, KD_PRED_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {p}")

    id_to_name, name_to_id = load_names_from_yaml(DATA_YAML)

    selected_ids = []

    for name in CLASS_GROUP:
        if name not in name_to_id:
            raise ValueError(f"类别 {name} 不在 yaml 中")

        selected_ids.append(name_to_id[name])

    print("Selected classes:")
    for name, cls_id in zip(CLASS_GROUP, selected_ids):
        print(f"{name}: {cls_id}")

    for stem in TARGET_STEMS:
        img_path = find_image_by_stem(IMG_DIR, stem)

        if img_path is None:
            print(f"[WARNING] 找不到图片: {stem}")
            continue

        gt_path = GT_LABEL_DIR / f"{stem}.txt"
        yolo_path = YOLO_PRED_DIR / f"{stem}.txt"
        kd_path = KD_PRED_DIR / f"{stem}.txt"

        gt_items = read_gt_file(gt_path)

        yolo_items = [
            item for item in read_pred_file(yolo_path)
            if item["conf"] >= DRAW_CONF_THRES
        ]

        kd_items = [
            item for item in read_pred_file(kd_path)
            if item["conf"] >= DRAW_CONF_THRES
        ]


        stem_out_dir = OUT_DIR / stem
        stem_out_dir.mkdir(parents=True, exist_ok=True)

        jobs = [
            {
                "name": "GT",
                "title": "GT",
                "items": gt_items,
                "color": GT_COLOR,
            },
            {
                "name": "YOLOv11",
                "title": "YOLOv11",
                "items": yolo_items,
                "color": YOLO_COLOR,
            },
            {
                "name": "ECAFA_YOLO_KD",
                "title": "ECAFA-YOLO+KD",
                "items": kd_items,
                "color": OURS_COLOR,
            },
        ]

        print("\n-----------------------------------------------------")
        print(f"Processing stem: {stem}")
        print(f"Image: {img_path}")
        print("-----------------------------------------------------")

        for job in jobs:
            save_base = stem_out_dir / f"{stem}_{job['name']}"

            pdf_path, svg_path, png_path, crop_box = draw_vector_annotations_on_original(
                image_path=img_path,
                items=job["items"],
                id_to_name=id_to_name,
                selected_ids=selected_ids,
                model_title=job["title"],
                box_color=job["color"],
                save_base=save_base,
                stem=stem
            )

            print(f"{job['title']}:")
            print(f"  PDF: {pdf_path}")
            print(f"  SVG: {svg_path}")
            print(f"  Crop box: {crop_box}")

            if png_path is not None:
                print(f"  PNG preview: {png_path}")

    print("\n=====================================================")
    print("All done.")
    print("=====================================================")