import os
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
from tqdm import tqdm

# ================== 配置 ==================
IMAGES_DIR = Path(r"F:\DataSets\tt100k\yolojack\images\val")
GT_LABELS  = Path(r"F:\DataSets\tt100k\yolojack\labels\val")

# 多模型预测目录
# 原来的 MODELS（保留）
# MODELS = {
#     "A": Path(r"E:\DataSets\tt100k_2021result\yolo11-FASFFHead_P234_RCSOSA_wiou_bce_distillation"),
#     "B": Path(r"E:\DataSets\tt100k_2021result\yolo11-FASFFHead_P234_RCSOSA_wiou_bce_train"),
#     "C": Path(r"E:\DataSets\tt100k_2021result\yolo11-FASFFHead_P234_RCSOSA_ciou_bce_train"),
#     "D": Path(r"E:\DataSets\tt100k_2021result\yolo11-FASFFHead_P234_train"),
#     "E": Path(r"E:\DataSets\tt100k_2021result\yolo11_train"),
# }

MODELS = {
    # "A": Path(r"F:\DataSets\resultTT100k130train\yolo11x-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train300(batch16worker16)"),
    "B": Path(r"F:\DataSets\resultTT100k130val\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation"),
    "C": Path(r"F:\DataSets\resultTT100k130val\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200"),
    "D": Path(r"F:\DataSets\resultTT100k130val\yolo11-FASFFHead_P234_train200"),
    "E": Path(r"F:\DataSets\resultTT100k130val\yolo11_train200"),
}

ORDER  = [ "B", "C", "D", "E"]   # 模型顺序
METRIC = "f1"                        # 或 "ap50"
STRICT = True                        # True: 严格 >；False: 允许 ≥
EPS    = 1e-6                        # 容差

IOU_THR      = 0.50
CONF_THR_PRF = 0.25
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp")

OUT_DIR   = Path(r"F:\DataSets\resultTT100k130val\multi_model_comparenew")
OUT_CSV   = OUT_DIR / f"per_image_{METRIC}_order_vis.csv"

COPY_TOPK = 5
COPY_BOTK = 5

COPY_DIR_TOP = OUT_DIR / "TopK_vis"
COPY_DIR_BOT = OUT_DIR / "BottomK_vis"

# ============== 可视化筛图参数（新增） ==============
# A 作为你的主模型，E 作为 baseline
MAIN_MODEL = "B"
BASE_MODEL = "E"

# 目标数量过滤：太少没内容，太多太乱
MIN_GT = 2
MAX_GT = 18

# 提升阈值：你的模型必须比 baseline 明显更好
MIN_DELTA_F1 = 0.10

# score_vis = delta_f1 + RECALL_WEIGHT * main_recall
RECALL_WEIGHT = 0.5


# ================== 工具函数 ==================
def yolo_to_xyxy(norm_box, w, h):
    cx, cy, bw, bh = norm_box
    return (
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h
    )


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def load_gt_txt(txt_path):
    out = []
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    out.append((int(float(p[0])), *map(float, p[1:5])))
    return out


def load_pred_txt(txt_path):
    out = []
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 6:
                    out.append((int(float(p[0])), *map(float, p[1:6])))
    return out


def match_by_iou(gts_xyxy, preds_xyxy, thr):
    """
    gts_xyxy:  [(cls, box), ...]
    preds_xyxy:[(cls, box, conf), ...]

    返回：
        tp, fp, fn
    """
    order = sorted(range(len(preds_xyxy)), key=lambda i: preds_xyxy[i][2], reverse=True)
    gt_used = [False] * len(gts_xyxy)

    tp = 0
    fp = 0

    for pi in order:
        p_cls, p_box, _ = preds_xyxy[pi]
        best_iou = 0.0
        best_gt = -1

        for gi, (g_cls, g_box) in enumerate(gts_xyxy):
            if gt_used[gi]:
                continue
            if g_cls != p_cls:
                continue

            iou = iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi

        if best_iou >= thr and best_gt >= 0:
            gt_used[best_gt] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn


def prf_from_match(tp, fp, fn):
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F1


def ap50_from_preds(gts_xyxy, preds_xyxy, thr):
    """
    简单 AP50 计算（按单图像）
    """
    preds = sorted(preds_xyxy, key=lambda x: x[2], reverse=True)
    gt_flags = [False] * len(gts_xyxy)

    tps = 0
    fps = 0
    precisions = []
    recalls = []
    total_gt = len(gts_xyxy)

    for p_cls, p_box, _ in preds:
        best_iou = 0.0
        best_gt = -1

        for gi, (g_cls, g_box) in enumerate(gts_xyxy):
            if gt_flags[gi]:
                continue
            if g_cls != p_cls:
                continue

            iou = iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi

        if best_iou >= thr and best_gt >= 0:
            gt_flags[best_gt] = True
            tps += 1
        else:
            fps += 1

        precisions.append(tps / (tps + fps) if (tps + fps) > 0 else 0.0)
        recalls.append(tps / total_gt if total_gt > 0 else 0.0)

    ap = 0.0
    prev_r, prev_p = 0.0, 1.0
    for r, p in sorted(zip(recalls, precisions)):
        ap += (r - prev_r) * (p + prev_p) / 2.0
        prev_r, prev_p = r, p

    return ap


def evaluate_one_image(img_path, gt_dir, pred_dir):
    """
    原来返回：
    {
        "n_gt": len(gts),
        "n_pred": len(preds_keep),
        "precision": P,
        "recall": R,
        "f1": F1,
        "ap50": ...
    }

    现在新增返回 tp / fp / fn
    """
    with Image.open(img_path) as im:
        W, H = im.size

    gts = [
        (c, yolo_to_xyxy((cx, cy, bw, bh), W, H))
        for c, cx, cy, bw, bh in load_gt_txt(gt_dir / f"{img_path.stem}.txt")
    ]

    preds_all = [
        (c, yolo_to_xyxy((cx, cy, bw, bh), W, H), s)
        for c, cx, cy, bw, bh, s in load_pred_txt(pred_dir / f"{img_path.stem}.txt")
    ]

    # 原来的 PR/F1 用的是过滤后的预测
    preds_keep = [p for p in preds_all if p[2] >= CONF_THR_PRF]

    tp, fp, fn = match_by_iou(gts, preds_keep, IOU_THR)
    P, R, F1 = prf_from_match(tp, fp, fn)

    return {
        "n_gt": len(gts),
        "n_pred": len(preds_keep),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": P,
        "recall": R,
        "f1": F1,
        "ap50": ap50_from_preds(gts, preds_all, IOU_THR),
    }


def check_order(values, strict=True, eps=0.0):
    for i in range(len(values) - 1):
        if strict:
            if not (values[i] > values[i + 1] + eps):
                return False
        else:
            if not (values[i] + eps >= values[i + 1]):
                return False
    return True


def safe_copy(src: Path, dst: Path):
    """
    安全拷贝，避免源文件不存在时报错
    """
    if src.exists():
        shutil.copy(src, dst)
    else:
        print(f"[WARN] Source image not found: {src}")


# ================== 主程序 ==================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_list = sorted([p for ext in IMG_EXTS for p in IMAGES_DIR.rglob(f"*{ext}")])
    if not img_list:
        print(f"[WARN] No images found in {IMAGES_DIR}")
        return

    rows = []

    for img_path in tqdm(img_list, desc="Evaluating", ncols=100):
        per_model = {
            name: evaluate_one_image(img_path, GT_LABELS, pred_dir)
            for name, pred_dir in MODELS.items()
        }

        # 原来：按 METRIC 检查排序
        metric_values = [per_model[name][METRIC] for name in ORDER]
        ok = check_order(metric_values, STRICT, EPS)

        row = {"image": img_path.name}

        # 原来只保存 F1 和 AP50（保留注释）
        # for name in MODELS:
        #     m = per_model[name]
        #     row.update({
        #         f"{name}_F1": m["f1"],
        #         f"{name}_AP50": m["ap50"]
        #     })

        # 现在：额外保存 GT/TP/Recall 等，方便后面筛“特征图更好看”的图
        for name in MODELS:
            m = per_model[name]
            row.update({
                f"{name}_GT": m["n_gt"],
                f"{name}_Pred": m["n_pred"],
                f"{name}_TP": m["tp"],
                f"{name}_FP": m["fp"],
                f"{name}_FN": m["fn"],
                f"{name}_Precision": m["precision"],
                f"{name}_Recall": m["recall"],
                f"{name}_F1": m["f1"],
                f"{name}_AP50": m["ap50"],
            })

        row["match"] = int(ok)

        # ========== 新增：可视化筛图分数 ==========
        # 你的主模型 vs baseline
        row["delta_f1_main_base"] = row[f"{MAIN_MODEL}_F1"] - row[f"{BASE_MODEL}_F1"]
        row["delta_recall_main_base"] = row[f"{MAIN_MODEL}_Recall"] - row[f"{BASE_MODEL}_Recall"]

        # 论文筛图分数：提升明显 + 主模型 Recall 高
        row["score_vis"] = row["delta_f1_main_base"] + RECALL_WEIGHT * row[f"{MAIN_MODEL}_Recall"]

        # 也可以额外留一个“整体F1总和”，方便你后续对比
        row["sum_f1"] = sum(row[f"{name}_F1"] for name in ORDER)

        rows.append(row)

    # ================== 保存 CSV ==================
    # 原来：
    # with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    #     w = csv.writer(f)
    #     w.writerow(row.keys())
    #     for r in rows:
    #         w.writerow(r.values())

    # 现在改成更稳妥写法：用 rows[0].keys()
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0].keys())
        for r in rows:
            w.writerow(r.values())

    print(f"[OK] CSV saved -> {OUT_CSV}")

    # ================== 选 TopK / BottomK ==================

    # 原来：
    # rows_match = [r for r in rows if r["match"] == 1]
    # rows_match.sort(key=lambda r: sum(r[f"{name}_F1"] for name in ORDER), reverse=True)

    # 现在：
    # TopK 选“更适合画特征图”的图：
    # 1) 排序成立
    # 2) GT 数量适中
    # 3) 主模型相对 baseline 提升明显
    rows_match = [
        r for r in rows
        if r["match"] == 1
        and r[f"{MAIN_MODEL}_GT"] >= MIN_GT
        and r[f"{MAIN_MODEL}_GT"] <= MAX_GT
        and r["delta_f1_main_base"] > MIN_DELTA_F1
    ]

    # 按可视化分数排序，而不是按 F1 总和排序
    rows_match.sort(key=lambda r: r["score_vis"], reverse=True)

    COPY_DIR_TOP.mkdir(parents=True, exist_ok=True)
    for r in rows_match[:COPY_TOPK]:
        safe_copy(IMAGES_DIR / r["image"], COPY_DIR_TOP / r["image"])

    # 原来 BottomK：
    # for r in rows_match[-COPY_BOTK:]:
    #     shutil.copy(IMAGES_DIR / r["image"], COPY_DIR_BOT / r["image"])

    # 现在 BottomK 改成：
    # 排序仍成立，GT 也适中，但与 baseline 差距很小
    rows_bottom = [
        r for r in rows
        if r["match"] == 1
        and r[f"{MAIN_MODEL}_GT"] >= MIN_GT
        and r[f"{MAIN_MODEL}_GT"] <= MAX_GT
    ]
    rows_bottom.sort(key=lambda r: abs(r["delta_f1_main_base"]))

    COPY_DIR_BOT.mkdir(parents=True, exist_ok=True)
    for r in rows_bottom[:COPY_BOTK]:
        safe_copy(IMAGES_DIR / r["image"], COPY_DIR_BOT / r["image"])

    # ================== 控制台输出一些统计信息 ==================
    print(f"[OK] Copied top {min(COPY_TOPK, len(rows_match))} images to -> {COPY_DIR_TOP}")
    print(f"[OK] Copied bottom {min(COPY_BOTK, len(rows_bottom))} images to -> {COPY_DIR_BOT}")

    print("\n===== Summary =====")
    print(f"Total images        : {len(rows)}")
    print(f"Order-matched images: {sum(r['match'] for r in rows)}")
    print(f"Top candidates      : {len(rows_match)}")
    print(f"Bottom candidates   : {len(rows_bottom)}")
    print(f"CSV path            : {OUT_CSV}")


if __name__ == "__main__":
    main()