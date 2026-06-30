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

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =====================================================
# 1. 路径配置
# =====================================================

# 当前脚本所在目录：
# D:\UKMJIAO\AlgorithmCodes\yolov11_jiao\回应审稿人2的类间差异
SCRIPT_DIR = Path(__file__).resolve().parent

# 项目根目录：
# D:\UKMJIAO\AlgorithmCodes\yolov11_jiao
PROJECT_ROOT = SCRIPT_DIR.parent

# 数据集 yaml，用来读取类别名称
DATA_YAML = PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "tt100k_desk_130.yaml"

# 真实 GT 标签目录
# 注意：你之前日志里虽然 split 写的是 test，但实际扫描的是 labels/val
GT_LABEL_DIR = Path(r"F:\DataSets\tt100k\yolojack\labels\val")

# YOLO 验证结果目录
CONFUSION_ROOT = PROJECT_ROOT / "vals_error_analysis" / "tt100k_confusion_full"

# 预测 labels 目录
PRED_DIRS = {
    "YOLOv11": CONFUSION_ROOT / "YOLOv11_baseline" / "exp" / "labels",
    "ECAFA-YOLO+KD": CONFUSION_ROOT / "ECAFA_YOLO_KD" / "exp" / "labels",
}

# 如果你想把 ECAFA-YOLO 也放进图里，就改用下面这个 PRED_DIRS
# PRED_DIRS = {
#     "YOLOv11": CONFUSION_ROOT / "YOLOv11_baseline" / "exp" / "labels",
#     "ECAFA-YOLO": CONFUSION_ROOT / "ECAFA_YOLO" / "exp" / "labels",
#     "ECAFA-YOLO+KD": CONFUSION_ROOT / "ECAFA_YOLO_KD" / "exp" / "labels",
# }

# 输出目录
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. 选择要分析的相似类别
# =====================================================

# 高度限制类，视觉相似，适合回应审稿人关于 visually similar signs 的意见
CLASS_GROUP = ["pl40", "pl50", "pl60", "pl80", "pl100"]

# 如果你想换成限速类，可以改成：
# CLASS_GROUP = ["pl40", "pl50", "pl60", "pl70", "pl80", "pl90", "pl100"]

# IoU 匹配阈值
IOU_THRES = 0.5

# 预测置信度阈值
# 这些预测 txt 已经是 YOLO val 后保存的结果，一般不需要再过滤太高
CONF_THRES = 0.0


# =====================================================
# 3. 从 yaml 读取类别名称
# =====================================================

def load_names_from_yaml(yaml_path):
    """
    从数据集 yaml 中读取类别名称。
    支持：
    names: [a, b, c]
    或：
    names:
      0: a
      1: b
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到数据集 yaml: {yaml_path}")

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "缺少 pyyaml，请先安装：\n"
            "pip install pyyaml"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "names" not in data:
        raise ValueError(f"yaml 中没有 names 字段: {yaml_path}")

    names = data["names"]

    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id_to_name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError("yaml 中 names 格式不支持。")

    name_to_id = {v: k for k, v in id_to_name.items()}

    return id_to_name, name_to_id


# =====================================================
# 4. 基础工具函数
# =====================================================

def xywh_to_xyxy(box):
    """
    YOLO 格式：
    x_center, y_center, width, height

    转为：
    x1, y1, x2, y2
    """
    x, y, w, h = box
    return np.array(
        [
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2,
        ],
        dtype=np.float64
    )


def box_iou(box1, box2):
    """
    计算两个 YOLO 归一化框的 IoU。
    """
    b1 = xywh_to_xyxy(box1)
    b2 = xywh_to_xyxy(box2)

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def read_gt_file(path):
    """
    读取真实标签：
    class_id x_center y_center width height
    """
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

            items.append((cls_id, box))

    return items


def read_pred_file(path):
    """
    读取预测标签：
    class_id x_center y_center width height confidence

    如果没有 confidence，则默认 conf = 1.0。
    """
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

            if conf >= CONF_THRES:
                items.append((cls_id, box, conf))

    return items


# =====================================================
# 5. 构建局部混淆矩阵
# =====================================================

def build_local_confusion_matrix(gt_dir, pred_dir, selected_class_ids):
    """
    构建局部混淆矩阵。

    行：True class
    列：Predicted class

    最后一行/列是 background：
    - True class -> background：漏检 FN
    - background -> Predicted class：误检 FP
    """
    n = len(selected_class_ids)
    background_idx = n

    cm = np.zeros((n + 1, n + 1), dtype=np.float64)

    class_id_to_local = {
        cls_id: i for i, cls_id in enumerate(selected_class_ids)
    }

    txt_files = sorted(gt_dir.glob("*.txt"))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"GT 目录中没有 txt 文件: {gt_dir}")

    for gt_path in txt_files:
        pred_path = pred_dir / gt_path.name

        gt_items = read_gt_file(gt_path)
        pred_items = read_pred_file(pred_path)

        # 只保留当前关注的类别
        gt_items = [
            item for item in gt_items
            if item[0] in selected_class_ids
        ]

        pred_items = [
            item for item in pred_items
            if item[0] in selected_class_ids
        ]

        used_pred = set()

        # 先匹配每个 GT
        for gt_cls, gt_box in gt_items:
            best_iou = 0.0
            best_j = -1

            for j, (pred_cls, pred_box, conf) in enumerate(pred_items):
                if j in used_pred:
                    continue

                iou = box_iou(gt_box, pred_box)

                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            gt_local = class_id_to_local[gt_cls]

            if best_iou >= IOU_THRES and best_j >= 0:
                pred_cls = pred_items[best_j][0]
                pred_local = class_id_to_local[pred_cls]

                cm[gt_local, pred_local] += 1
                used_pred.add(best_j)
            else:
                # 漏检：真实有目标，但没有匹配到预测
                cm[gt_local, background_idx] += 1

        # 剩下没有匹配到 GT 的预测，记为误检
        for j, (pred_cls, pred_box, conf) in enumerate(pred_items):
            if j not in used_pred:
                pred_local = class_id_to_local[pred_cls]
                cm[background_idx, pred_local] += 1

    return cm


def normalize_rows(cm):
    """
    按行归一化。
    每一行加起来为 1。
    """
    row_sum = cm.sum(axis=1, keepdims=True)

    cm_norm = np.divide(
        cm,
        row_sum,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sum != 0
    )

    return cm_norm


# =====================================================
# 6. 绘图函数
# =====================================================

def plot_single_matrix(cm, labels, title, save_path):
    """
    单独绘制一个模型的局部混淆矩阵。
    """
    cm_norm = normalize_rows(cm)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6.5, 5.8),
        constrained_layout=True
    )

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=10
    )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(
        labels,
        fontsize=10
    )

    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j]

            if value > 0:
                text_color = "white" if value > 0.5 else "black"

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color
                )

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        pad=0.02
    )
    cbar.set_label("Normalized value", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_compare_models(cm_dict, labels, save_path):
    """
    将多个模型的局部混淆矩阵画到同一张图中。
    已修复 tight_layout 与 colorbar 不兼容导致的排版问题。
    """
    model_names = list(cm_dict.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(7.2 * n_models, 6.2),
        constrained_layout=True
    )

    if n_models == 1:
        axes = [axes]

    im = None

    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        cm_norm = normalize_rows(cm_dict[model_name])

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
            fontsize=10
        )

        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(
            labels,
            fontsize=10
        )

        ax.set_xlabel("Predicted class", fontsize=11)

        # 只在第一张子图显示 y 轴标题，避免重复拥挤
        if idx == 0:
            ax.set_ylabel("True class", fontsize=11)
        else:
            ax.set_ylabel("")

        ax.set_title(model_name, fontsize=12, pad=10)

        # 在格子里写数值
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                value = cm_norm[i, j]

                if value > 0:
                    text_color = "white" if value >= 0.6 else "black"

                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color
                    )

    # 统一 colorbar
    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.82,
        pad=0.02
    )
    cbar.set_label("Normalized value", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")

    # 不再使用 plt.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =====================================================
# 7. 主程序
# =====================================================

if __name__ == "__main__":

    print("\n=====================================================")
    print("Localized Normalized Confusion Matrix")
    print("=====================================================")
    print(f"SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_YAML: {DATA_YAML}")
    print(f"GT_LABEL_DIR: {GT_LABEL_DIR}")
    print(f"CONFUSION_ROOT: {CONFUSION_ROOT}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("=====================================================\n")

    if not GT_LABEL_DIR.exists():
        raise FileNotFoundError(f"找不到 GT 标签目录: {GT_LABEL_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到数据集 yaml: {DATA_YAML}")

    id_to_name, name_to_id = load_names_from_yaml(DATA_YAML)

    selected_ids = []

    for name in CLASS_GROUP:
        if name not in name_to_id:
            raise ValueError(
                f"类别 {name} 不在 yaml 的 names 中。\n"
                f"请检查 CLASS_GROUP 是否写错。\n"
                f"当前 yaml 中前 30 个类别为: {list(name_to_id.keys())[:30]}"
            )

        selected_ids.append(name_to_id[name])

    print("Selected classes:")
    for class_name, class_id in zip(CLASS_GROUP, selected_ids):
        print(f"{class_name}: {class_id}")

    labels = CLASS_GROUP + ["background"]

    cm_dict = {}

    for model_name, pred_dir in PRED_DIRS.items():
        print("\n-----------------------------------------------------")
        print(f"Processing model: {model_name}")
        print(f"PRED_DIR: {pred_dir}")
        print("-----------------------------------------------------")

        if not pred_dir.exists():
            raise FileNotFoundError(
                f"找不到预测 labels 目录:\n{pred_dir}\n\n"
                "请确认你已经运行过验证脚本，并且 model.val 中包含：\n"
                "save_txt=True, save_conf=True"
            )

        cm = build_local_confusion_matrix(
            gt_dir=GT_LABEL_DIR,
            pred_dir=pred_dir,
            selected_class_ids=selected_ids
        )

        cm_dict[model_name] = cm

        safe_model_name = (
            model_name.replace("+", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )

        class_tag = "_".join(CLASS_GROUP).replace(".", "_")

        single_save_path = OUT_DIR / f"local_confusion_{safe_model_name}_{class_tag}"

        plot_single_matrix(
            cm=cm,
            labels=labels,
            title=model_name,
            save_path=single_save_path
        )

    # 生成一张模型对比图，论文中优先使用这张
    class_tag = "_".join(CLASS_GROUP).replace(".", "_")
    compare_save_path = OUT_DIR / f"local_confusion_compare_{class_tag}"

    plot_compare_models(
        cm_dict=cm_dict,
        labels=labels,
        save_path=compare_save_path
    )

    print("\nAll done.")