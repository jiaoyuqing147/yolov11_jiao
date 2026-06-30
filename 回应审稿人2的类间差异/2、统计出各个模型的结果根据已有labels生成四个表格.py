import os

# =====================================================
# Windows 下防止 OpenMP 冲突
# =====================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd


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

# 真实标签目录
# 注意：你之前日志里虽然 split 写 test，但实际扫描的是 labels/val
# 所以这里先用 val。如果你确认真实测试集是 labels/test，再改成 test。
GT_LABEL_DIR = Path(r"F:\DataSets\tt100k\yolojack\labels\val")

# 你已经跑过验证后保存 labels 的目录
CONFUSION_ROOT = PROJECT_ROOT / "vals_error_analysis" / "tt100k_confusion_full"

PRED_DIRS = {
    "YOLOv11_baseline": CONFUSION_ROOT / "YOLOv11_baseline" / "exp" / "labels",
    "ECAFA_YOLO": CONFUSION_ROOT / "ECAFA_YOLO" / "exp" / "labels",
    "ECAFA_YOLO_KD": CONFUSION_ROOT / "ECAFA_YOLO_KD" / "exp" / "labels",
}

# 四个最终表格输出到当前文件夹
OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IoU 阈值，AP50-95
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

# 是否过滤很低置信度预测
# YOLO val 保存的 labels 通常已经经过 NMS；这里一般不需要再强过滤。
# 如果你想更严格，可以改成 0.001 或 0.25。
CONF_THRES = 0.0


# =====================================================
# 2. 读取 yaml 类别名
# =====================================================

def load_names_from_yaml(yaml_path):
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到数据集 yaml: {yaml_path}")

    try:
        import yaml
    except ImportError:
        raise ImportError("缺少 pyyaml，请先安装：pip install pyyaml")

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

    return id_to_name


# =====================================================
# 3. box 与 IoU
# =====================================================

def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ], dtype=np.float64)


def box_iou(box1, box2):
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


# =====================================================
# 4. 读取 GT 和预测 labels
# =====================================================

def read_gt_file(path):
    """
    GT txt 格式：
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

            items.append({
                "cls": cls_id,
                "box": box
            })

    return items


def read_pred_file(path):
    """
    预测 txt 格式：
    class_id x_center y_center width height confidence

    如果没有 confidence，则 conf = 1.0。
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
                items.append({
                    "cls": cls_id,
                    "box": box,
                    "conf": conf
                })

    return items


def load_all_gt(gt_dir):
    """
    返回：
    gt_by_class_image[class_id][image_id] = [box1, box2, ...]
    num_gt[class_id] = gt 数量
    image_ids = 所有图片 id
    """
    if not gt_dir.exists():
        raise FileNotFoundError(f"找不到 GT 标签目录: {gt_dir}")

    txt_files = sorted(gt_dir.glob("*.txt"))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"GT 标签目录中没有 txt 文件: {gt_dir}")

    gt_by_class_image = {}
    num_gt = {}
    image_ids = []

    for txt_path in txt_files:
        image_id = txt_path.stem
        image_ids.append(image_id)

        items = read_gt_file(txt_path)

        for item in items:
            cls_id = item["cls"]
            box = item["box"]

            gt_by_class_image.setdefault(cls_id, {})
            gt_by_class_image[cls_id].setdefault(image_id, [])
            gt_by_class_image[cls_id][image_id].append(box)

            num_gt[cls_id] = num_gt.get(cls_id, 0) + 1

    return gt_by_class_image, num_gt, image_ids


def load_all_predictions(pred_dir):
    """
    返回：
    preds_by_class[class_id] = [
        {"image_id": ..., "box": ..., "conf": ...},
        ...
    ]
    """
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"找不到预测 labels 目录:\n{pred_dir}\n\n"
            "请确认你已经用 save_txt=True, save_conf=True 跑过 val。"
        )

    txt_files = sorted(pred_dir.glob("*.txt"))

    preds_by_class = {}

    for txt_path in txt_files:
        image_id = txt_path.stem
        items = read_pred_file(txt_path)

        for item in items:
            cls_id = item["cls"]

            preds_by_class.setdefault(cls_id, [])
            preds_by_class[cls_id].append({
                "image_id": image_id,
                "box": item["box"],
                "conf": item["conf"]
            })

    return preds_by_class


# =====================================================
# 5. AP 计算
# =====================================================

def compute_ap(recall, precision):
    """
    计算 AP，使用 101-point interpolation。
    """
    if recall.size == 0 or precision.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return float(ap)


def evaluate_one_class(cls_id, gt_by_class_image, num_gt, preds_by_class, iou_thresholds):
    """
    计算单个类别的 Precision、Recall、AP50、AP50-95。
    """
    n_gt = int(num_gt.get(cls_id, 0))
    preds = preds_by_class.get(cls_id, [])

    n_pred = len(preds)

    if n_gt == 0:
        return {
            "Instances": 0,
            "Predictions": n_pred,
            "Precision": np.nan,
            "Recall": np.nan,
            "AP50": np.nan,
            "AP50-95": np.nan,
        }

    if n_pred == 0:
        return {
            "Instances": n_gt,
            "Predictions": 0,
            "Precision": 0.0,
            "Recall": 0.0,
            "AP50": 0.0,
            "AP50-95": 0.0,
        }

    # 按置信度从高到低排序
    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

    ap_list = []

    final_precision_at_50 = 0.0
    final_recall_at_50 = 0.0

    gt_for_cls = gt_by_class_image.get(cls_id, {})

    for t_i, iou_thres in enumerate(iou_thresholds):
        tp = np.zeros(n_pred, dtype=np.float64)
        fp = np.zeros(n_pred, dtype=np.float64)

        # 每个阈值下，GT 匹配状态要重新初始化
        matched = {
            img_id: np.zeros(len(boxes), dtype=bool)
            for img_id, boxes in gt_for_cls.items()
        }

        for pred_i, pred in enumerate(preds):
            image_id = pred["image_id"]
            pred_box = pred["box"]

            gt_boxes = gt_for_cls.get(image_id, [])

            if len(gt_boxes) == 0:
                fp[pred_i] = 1.0
                continue

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if matched[image_id][gt_idx]:
                    continue

                iou = box_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thres and best_gt_idx >= 0:
                tp[pred_i] = 1.0
                matched[image_id][best_gt_idx] = True
            else:
                fp[pred_i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall_curve = tp_cum / (n_gt + 1e-16)
        precision_curve = tp_cum / (tp_cum + fp_cum + 1e-16)

        ap = compute_ap(recall_curve, precision_curve)
        ap_list.append(ap)

        if abs(iou_thres - 0.5) < 1e-9:
            final_precision_at_50 = float(precision_curve[-1]) if precision_curve.size else 0.0
            final_recall_at_50 = float(recall_curve[-1]) if recall_curve.size else 0.0

    ap50 = ap_list[0]
    ap5095 = float(np.mean(ap_list))

    return {
        "Instances": n_gt,
        "Predictions": n_pred,
        "Precision": final_precision_at_50,
        "Recall": final_recall_at_50,
        "AP50": ap50,
        "AP50-95": ap5095,
    }


def evaluate_model(model_name, pred_dir, id_to_name, gt_by_class_image, num_gt):
    """
    评估一个模型，生成 per-class dataframe。
    """
    print("\n-----------------------------------------------------")
    print(f"Evaluating existing labels: {model_name}")
    print(f"PRED_DIR: {pred_dir}")
    print("-----------------------------------------------------")

    preds_by_class = load_all_predictions(pred_dir)

    rows = []

    for cls_id in sorted(id_to_name.keys()):
        class_name = id_to_name[cls_id]

        metrics = evaluate_one_class(
            cls_id=cls_id,
            gt_by_class_image=gt_by_class_image,
            num_gt=num_gt,
            preds_by_class=preds_by_class,
            iou_thresholds=IOU_THRESHOLDS
        )

        rows.append({
            "class_id": cls_id,
            "class_name": class_name,
            "Instances": metrics["Instances"],
            f"{model_name}_Predictions": metrics["Predictions"],
            f"{model_name}_Precision": metrics["Precision"],
            f"{model_name}_Recall": metrics["Recall"],
            f"{model_name}_AP50": metrics["AP50"],
            f"{model_name}_AP50-95": metrics["AP50-95"],
        })

    df = pd.DataFrame(rows)

    # 保存每个模型自己的 per-class 表，方便检查
    save_dir = pred_dir.parent
    out_csv = save_dir / f"per_class_{model_name}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved model per-class table: {out_csv}")

    return df


# =====================================================
# 6. 主程序
# =====================================================

if __name__ == "__main__":

    print("\n=====================================================")
    print("Generate four CSV tables from existing labels")
    print("No model validation will be run.")
    print("=====================================================")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_YAML: {DATA_YAML}")
    print(f"GT_LABEL_DIR: {GT_LABEL_DIR}")
    print(f"CONFUSION_ROOT: {CONFUSION_ROOT}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("=====================================================\n")

    id_to_name = load_names_from_yaml(DATA_YAML)

    gt_by_class_image, num_gt, image_ids = load_all_gt(GT_LABEL_DIR)

    print(f"Loaded GT images: {len(image_ids)}")
    print(f"Loaded classes from yaml: {len(id_to_name)}")

    all_dfs = []

    for model_name, pred_dir in PRED_DIRS.items():
        df_model = evaluate_model(
            model_name=model_name,
            pred_dir=pred_dir,
            id_to_name=id_to_name,
            gt_by_class_image=gt_by_class_image,
            num_gt=num_gt
        )

        all_dfs.append(df_model)

    # =====================================================
    # 7. 合并三个模型
    # =====================================================

    merged = all_dfs[0]

    for df in all_dfs[1:]:
        # Instances 是同一份 GT，合并时避免重复
        if "Instances" in df.columns:
            df = df.drop(columns=["Instances"])

        merged = pd.merge(
            merged,
            df,
            on=["class_id", "class_name"],
            how="outer"
        )

    # =====================================================
    # 8. 计算提升量
    # =====================================================

    merged["AP50-95_improvement_ECAFA_vs_baseline"] = (
        merged["ECAFA_YOLO_AP50-95"] - merged["YOLOv11_baseline_AP50-95"]
    )

    merged["AP50-95_improvement_KD_vs_baseline"] = (
        merged["ECAFA_YOLO_KD_AP50-95"] - merged["YOLOv11_baseline_AP50-95"]
    )

    merged["AP50-95_improvement_KD_vs_ECAFA"] = (
        merged["ECAFA_YOLO_KD_AP50-95"] - merged["ECAFA_YOLO_AP50-95"]
    )

    merged["AP50_improvement_KD_vs_baseline"] = (
        merged["ECAFA_YOLO_KD_AP50"] - merged["YOLOv11_baseline_AP50"]
    )

    merged["AP50_improvement_KD_vs_ECAFA"] = (
        merged["ECAFA_YOLO_KD_AP50"] - merged["ECAFA_YOLO_AP50"]
    )

    # =====================================================
    # 9. 保存四个表格
    # =====================================================

    out_all = OUT_DIR / "per_class_comparison_all_models.csv"
    merged.to_csv(out_all, index=False, encoding="utf-8-sig")

    top10_kd_vs_baseline = merged.sort_values(
        "AP50-95_improvement_KD_vs_baseline",
        ascending=False
    ).head(10)

    out_top10_baseline = OUT_DIR / "top10_improved_classes_KD_vs_baseline.csv"
    top10_kd_vs_baseline.to_csv(out_top10_baseline, index=False, encoding="utf-8-sig")

    top10_kd_vs_ecafa = merged.sort_values(
        "AP50-95_improvement_KD_vs_ECAFA",
        ascending=False
    ).head(10)

    out_top10_ecafa = OUT_DIR / "top10_KD_gain_classes_vs_ECAFA.csv"
    top10_kd_vs_ecafa.to_csv(out_top10_ecafa, index=False, encoding="utf-8-sig")

    top10_low_ap = merged.sort_values(
        "ECAFA_YOLO_KD_AP50-95",
        ascending=True
    ).head(10)

    out_low_ap = OUT_DIR / "top10_low_AP_classes_final_model.csv"
    top10_low_ap.to_csv(out_low_ap, index=False, encoding="utf-8-sig")

    print("\n=====================================================")
    print("Saved four final tables:")
    print(out_all)
    print(out_top10_baseline)
    print(out_top10_ecafa)
    print(out_low_ap)
    print("=====================================================")
    print("All done.")