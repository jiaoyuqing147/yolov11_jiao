import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import csv
import numpy as np


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

OUT_DIR = SCRIPT_DIR / "fp_fn_case_ids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "selected_case_ids.csv"


# =====================================================
# 2. 类别与筛选配置
# =====================================================

# 和混淆矩阵配套的限速类
CLASS_GROUP = ["pl40", "pl50", "pl60", "pl80", "pl100"]

IOU_THRES = 0.5
CONF_THRES = 0.25

# 每类 case 输出前多少个候选编号
TOP_K_PER_CASE = 30


# =====================================================
# 3. 读取类别名称
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
# 4. 读取标签
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

            if conf >= CONF_THRES:
                items.append(
                    {
                        "cls": cls_id,
                        "box": box,
                        "conf": conf,
                    }
                )

    return items


# =====================================================
# 5. IoU 与匹配
# =====================================================

def xywh_to_xyxy(box):
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


def match_gt_to_preds(gt_items, pred_items):
    used_pred = set()
    matches = []

    for gi, gt in enumerate(gt_items):
        best_iou = 0.0
        best_pi = -1

        for pi, pred in enumerate(pred_items):
            if pi in used_pred:
                continue

            iou = box_iou(gt["box"], pred["box"])

            if iou > best_iou:
                best_iou = iou
                best_pi = pi

        if best_iou >= IOU_THRES and best_pi >= 0:
            used_pred.add(best_pi)
            pred = pred_items[best_pi]

            status = "correct" if gt["cls"] == pred["cls"] else "wrong_class"

            matches.append(
                {
                    "gt_index": gi,
                    "pred_index": best_pi,
                    "gt_cls": gt["cls"],
                    "pred_cls": pred["cls"],
                    "iou": best_iou,
                    "conf": pred["conf"],
                    "status": status,
                }
            )
        else:
            matches.append(
                {
                    "gt_index": gi,
                    "pred_index": -1,
                    "gt_cls": gt["cls"],
                    "pred_cls": -1,
                    "iou": best_iou,
                    "conf": 0.0,
                    "status": "missed",
                }
            )

    unmatched_pred_indices = [
        pi for pi in range(len(pred_items))
        if pi not in used_pred
    ]

    return matches, unmatched_pred_indices


# =====================================================
# 6. 图片查找
# =====================================================

def find_image_by_stem(img_dir, stem):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]

    for ext in exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


# =====================================================
# 7. case 判断
# =====================================================

def count_status(matches, selected_ids):
    correct = 0
    missed = 0
    wrong_class = 0

    for m in matches:
        if m["gt_cls"] not in selected_ids:
            continue

        if m["status"] == "correct":
            correct += 1
        elif m["status"] == "missed":
            missed += 1
        elif m["status"] == "wrong_class":
            wrong_class += 1

    return correct, missed, wrong_class


def count_fp(pred_items, unmatched_indices, selected_ids):
    fp = 0
    fp_classes = []

    for pi in unmatched_indices:
        pred = pred_items[pi]

        if pred["cls"] in selected_ids:
            fp += 1
            fp_classes.append(pred["cls"])

    return fp, fp_classes


def cls_name(cls_id, id_to_name):
    if cls_id == -1:
        return "background"

    return id_to_name.get(cls_id, str(cls_id))


def detail_to_string(details, id_to_name):
    out = []

    for d in details:
        if d["type"] == "miss_fixed":
            out.append(
                f'{cls_name(d["gt"], id_to_name)}: background -> {cls_name(d["kd_pred"], id_to_name)}'
            )

        elif d["type"] == "wrong_fixed":
            out.append(
                f'{cls_name(d["gt"], id_to_name)}: {cls_name(d["yolo_pred"], id_to_name)} -> {cls_name(d["kd_pred"], id_to_name)}'
            )

        elif d["type"] == "remaining_failure":
            out.append(
                f'{cls_name(d["gt"], id_to_name)}: KD {d["status"]} as {cls_name(d["kd_pred"], id_to_name)}'
            )

    return "; ".join(out)


def find_fixed_miss(yolo_matches, kd_matches, selected_ids):
    count = 0
    details = []

    for ym, km in zip(yolo_matches, kd_matches):
        if ym["gt_cls"] not in selected_ids:
            continue

        if ym["status"] == "missed" and km["status"] == "correct":
            count += 1
            details.append(
                {
                    "type": "miss_fixed",
                    "gt": ym["gt_cls"],
                    "kd_pred": km["pred_cls"],
                }
            )

    return count, details


def find_fixed_misclassification(yolo_matches, kd_matches, selected_ids):
    count = 0
    details = []

    for ym, km in zip(yolo_matches, kd_matches):
        if ym["gt_cls"] not in selected_ids:
            continue

        if ym["status"] == "wrong_class" and km["status"] == "correct":
            count += 1
            details.append(
                {
                    "type": "wrong_fixed",
                    "gt": ym["gt_cls"],
                    "yolo_pred": ym["pred_cls"],
                    "kd_pred": km["pred_cls"],
                }
            )

    return count, details


def find_remaining_failure(kd_matches, selected_ids):
    count = 0
    details = []

    for km in kd_matches:
        if km["gt_cls"] not in selected_ids:
            continue

        if km["status"] in ["missed", "wrong_class"]:
            count += 1
            details.append(
                {
                    "type": "remaining_failure",
                    "gt": km["gt_cls"],
                    "kd_pred": km["pred_cls"],
                    "status": km["status"],
                }
            )

    return count, details


# =====================================================
# 8. 主程序
# =====================================================

if __name__ == "__main__":

    print("\n=====================================================")
    print("Select Case Image IDs Only")
    print("=====================================================")
    print(f"GT_LABEL_DIR: {GT_LABEL_DIR}")
    print(f"IMG_DIR: {IMG_DIR}")
    print(f"YOLO_PRED_DIR: {YOLO_PRED_DIR}")
    print(f"KD_PRED_DIR: {KD_PRED_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("=====================================================\n")

    for p in [DATA_YAML, GT_LABEL_DIR, IMG_DIR, YOLO_PRED_DIR, KD_PRED_DIR]:
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

    candidate_cases = {
        "Case1_FN_fixed": [],
        "Case2_Misclassification_fixed": [],
        "Case3_FP_reduced": [],
        "Case4_Remaining_failure": [],
    }

    gt_files = sorted(GT_LABEL_DIR.glob("*.txt"))

    for gt_path in gt_files:
        stem = gt_path.stem

        img_path = find_image_by_stem(IMG_DIR, stem)

        if img_path is None:
            continue

        yolo_path = YOLO_PRED_DIR / gt_path.name
        kd_path = KD_PRED_DIR / gt_path.name

        gt_items_all = read_gt_file(gt_path)
        yolo_preds_all = read_pred_file(yolo_path)
        kd_preds_all = read_pred_file(kd_path)

        gt_items = [x for x in gt_items_all if x["cls"] in selected_ids]
        yolo_preds = [x for x in yolo_preds_all if x["cls"] in selected_ids]
        kd_preds = [x for x in kd_preds_all if x["cls"] in selected_ids]

        if len(gt_items) == 0 and len(yolo_preds) == 0 and len(kd_preds) == 0:
            continue

        yolo_matches, yolo_unmatched = match_gt_to_preds(gt_items, yolo_preds)
        kd_matches, kd_unmatched = match_gt_to_preds(gt_items, kd_preds)

        yolo_correct, yolo_missed, yolo_wrong = count_status(yolo_matches, selected_ids)
        kd_correct, kd_missed, kd_wrong = count_status(kd_matches, selected_ids)

        yolo_fp, yolo_fp_classes = count_fp(yolo_preds, yolo_unmatched, selected_ids)
        kd_fp, kd_fp_classes = count_fp(kd_preds, kd_unmatched, selected_ids)

        # Case 1: YOLO 漏检，KD 检出
        fixed_miss_count, fixed_miss_details = find_fixed_miss(
            yolo_matches,
            kd_matches,
            selected_ids
        )

        if fixed_miss_count > 0:
            score = fixed_miss_count * 10 + kd_correct - yolo_correct

            candidate_cases["Case1_FN_fixed"].append(
                {
                    "case": "Case1_FN_fixed",
                    "stem": stem,
                    "score": score,
                    "details": detail_to_string(fixed_miss_details, id_to_name),
                    "gt_count": len(gt_items),
                    "yolo_correct": yolo_correct,
                    "yolo_missed": yolo_missed,
                    "yolo_wrong": yolo_wrong,
                    "yolo_fp": yolo_fp,
                    "kd_correct": kd_correct,
                    "kd_missed": kd_missed,
                    "kd_wrong": kd_wrong,
                    "kd_fp": kd_fp,
                    "image_path": str(img_path),
                }
            )

        # Case 2: YOLO 错分类，KD 正确
        fixed_wrong_count, fixed_wrong_details = find_fixed_misclassification(
            yolo_matches,
            kd_matches,
            selected_ids
        )

        if fixed_wrong_count > 0:
            score = fixed_wrong_count * 10 + kd_correct - yolo_correct

            candidate_cases["Case2_Misclassification_fixed"].append(
                {
                    "case": "Case2_Misclassification_fixed",
                    "stem": stem,
                    "score": score,
                    "details": detail_to_string(fixed_wrong_details, id_to_name),
                    "gt_count": len(gt_items),
                    "yolo_correct": yolo_correct,
                    "yolo_missed": yolo_missed,
                    "yolo_wrong": yolo_wrong,
                    "yolo_fp": yolo_fp,
                    "kd_correct": kd_correct,
                    "kd_missed": kd_missed,
                    "kd_wrong": kd_wrong,
                    "kd_fp": kd_fp,
                    "image_path": str(img_path),
                }
            )

        # Case 3: YOLO 误检更多，KD 误检更少
        if yolo_fp > kd_fp:
            score = (yolo_fp - kd_fp) * 10 + kd_correct - yolo_correct
            yolo_fp_names = [cls_name(x, id_to_name) for x in yolo_fp_classes]
            kd_fp_names = [cls_name(x, id_to_name) for x in kd_fp_classes]

            candidate_cases["Case3_FP_reduced"].append(
                {
                    "case": "Case3_FP_reduced",
                    "stem": stem,
                    "score": score,
                    "details": f"FP reduced: YOLOv11 {yolo_fp} {yolo_fp_names} -> KD {kd_fp} {kd_fp_names}",
                    "gt_count": len(gt_items),
                    "yolo_correct": yolo_correct,
                    "yolo_missed": yolo_missed,
                    "yolo_wrong": yolo_wrong,
                    "yolo_fp": yolo_fp,
                    "kd_correct": kd_correct,
                    "kd_missed": kd_missed,
                    "kd_wrong": kd_wrong,
                    "kd_fp": kd_fp,
                    "image_path": str(img_path),
                }
            )

        # Case 4: KD 仍然失败
        remaining_count, remaining_details = find_remaining_failure(
            kd_matches,
            selected_ids
        )

        if remaining_count > 0:
            score = remaining_count * 10 + kd_missed + kd_wrong

            candidate_cases["Case4_Remaining_failure"].append(
                {
                    "case": "Case4_Remaining_failure",
                    "stem": stem,
                    "score": score,
                    "details": detail_to_string(remaining_details, id_to_name),
                    "gt_count": len(gt_items),
                    "yolo_correct": yolo_correct,
                    "yolo_missed": yolo_missed,
                    "yolo_wrong": yolo_wrong,
                    "yolo_fp": yolo_fp,
                    "kd_correct": kd_correct,
                    "kd_missed": kd_missed,
                    "kd_wrong": kd_wrong,
                    "kd_fp": kd_fp,
                    "image_path": str(img_path),
                }
            )

    final_rows = []

    for case_name, rows in candidate_cases.items():
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)

        print(f"\n{case_name}: {len(rows)} candidates")

        for rank, row in enumerate(rows[:TOP_K_PER_CASE], start=1):
            row_out = dict(row)
            row_out["rank_in_case"] = rank
            final_rows.append(row_out)

            print(
                f'  Rank {rank:02d} | stem={row["stem"]} | score={row["score"]} | {row["details"]}'
            )

    fieldnames = [
        "case",
        "rank_in_case",
        "stem",
        "score",
        "details",
        "gt_count",
        "yolo_correct",
        "yolo_missed",
        "yolo_wrong",
        "yolo_fp",
        "kd_correct",
        "kd_missed",
        "kd_wrong",
        "kd_fp",
        "image_path",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in final_rows:
            writer.writerow(row)

    print("\n=====================================================")
    print("Done.")
    print(f"CSV saved to: {CSV_PATH}")
    print("=====================================================")