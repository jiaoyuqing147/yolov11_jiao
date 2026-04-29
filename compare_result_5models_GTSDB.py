import os
import csv
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ================== 配置 ==================
IMAGES_DIR = Path(r"E:\DataSets\GTSDB\yolo43\images\val")
GT_LABELS  = Path(r"E:\DataSets\GTSDB\yolo43\labels\val")

# 多模型预测目录
MODELS = {
    "A": Path(r"E:\DataSets\resultGTSDBval\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation"),
    "B": Path(r"E:\DataSets\resultGTSDBval\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200"),
    "C": Path(r"E:\DataSets\resultGTSDBval\yolo11-FASFFHead_P234_train200"),
    "D": Path(r"E:\DataSets\resultGTSDBval\yolo11-OECSOSAInterleave_train200"),
    "E": Path(r"E:\DataSets\resultGTSDBval\yolo11_train200"),
}

ORDER = ["A", "B", "C", "D", "E"]

# ================== 评价逻辑 ==================
# 你说“模型本事在于尽可能多找到目标”，所以主指标改成 recall
METRIC = "recall"

STRICT = True
EPS = 1e-6

IOU_THR = 0.50
CONF_THR_PRF = 0.25
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

OUT_DIR = Path(r"E:\DataSets\resultGTSDBval\multi_model_comparenew2")
OUT_CSV = OUT_DIR / "per_image_recall_custom_order_vis.csv"

COPY_TOPK = 10
COPY_BOTK = 10

COPY_DIR_TOP = OUT_DIR / "TopK_vis"
COPY_DIR_BOT = OUT_DIR / "BottomK_vis"

# 主模型 / baseline
MAIN_MODEL = "A"
BASE_MODEL = "E"

# ================== GT数量统一控制 ==================
# 1) 参与 match 判断的最小 GT 数量（例如设成 2，表示 GT >= 2 才参与）
MIN_GT_MATCH = 1

# 2) TopK / BottomK 额外筛选的 GT 范围
MIN_GT_FILTER = 1
MAX_GT_FILTER = 999999

# recall 至少比 baseline 高这么多，才算“明显提升”
MIN_DELTA_RECALL = 0.005

# 可视化打分
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
    gts_xyxy:   [(cls, box), ...]
    preds_xyxy: [(cls, box, conf), ...]

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


def better(a, b, strict=True, eps=1e-6):
    if strict:
        return a > b + eps
    else:
        return a + eps >= b


def check_custom_order(per_model, metric="recall", strict=True, eps=1e-6):
    """
    必须同时满足：
        A > B > C > E
        A > B > D > E

    等价于：
        A > B
        B > C
        C > E
        B > D
        D > E

    注意：
        不比较 C 和 D
    """
    A = per_model["A"][metric]
    B = per_model["B"][metric]
    C = per_model["C"][metric]
    D = per_model["D"][metric]
    E = per_model["E"][metric]

    cond_ab = better(A, B, strict, eps)
    cond_bc = better(B, C, strict, eps)
    cond_ce = better(C, E, strict, eps)
    cond_bd = better(B, D, strict, eps)
    cond_de = better(D, E, strict, eps)

    ok = cond_ab and cond_bc and cond_ce and cond_bd and cond_de
    return ok, cond_ab, cond_bc, cond_ce, cond_bd, cond_de


def safe_copy(src: Path, dst: Path):
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

        n_gt = per_model["A"]["n_gt"]

        ok, ok_ab, ok_bc, ok_ce, ok_bd, ok_de = check_custom_order(
            per_model,
            metric=METRIC,
            strict=STRICT,
            eps=EPS
        )

        row = {"image": img_path.name}

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

        # GT数量是否达到参与match判断的门槛
        row["gt_ok"] = int(n_gt >= MIN_GT_MATCH)

        row["match_AB"] = int(ok_ab)
        row["match_BC"] = int(ok_bc)
        row["match_CE"] = int(ok_ce)
        row["match_BD"] = int(ok_bd)
        row["match_DE"] = int(ok_de)

        # 最终是否通过：GT数量达到要求 且 两条链同时成立
        row["match"] = int(ok and (n_gt >= MIN_GT_MATCH))

        # 用 recall 做主模型和 baseline 的差值
        row["delta_recall_main_base"] = row[f"{MAIN_MODEL}_Recall"] - row[f"{BASE_MODEL}_Recall"]

        # 可视化得分：既要比 baseline 提升，也要主模型本身 recall 高
        row["score_vis"] = row["delta_recall_main_base"] + RECALL_WEIGHT * row[f"{MAIN_MODEL}_Recall"]

        # 额外保留总 TP
        row["sum_tp"] = sum(row[f"{name}_TP"] for name in ORDER)

        rows.append(row)

    # ================== 保存 CSV ==================
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0].keys())
        for r in rows:
            w.writerow(r.values())

    print(f"[OK] CSV saved -> {OUT_CSV}")

    # ================== TopK ==================
    rows_match = [
        r for r in rows
        if r["match"] == 1
        and r["delta_recall_main_base"] > MIN_DELTA_RECALL
        and r[f"{MAIN_MODEL}_GT"] >= MIN_GT_FILTER
        and r[f"{MAIN_MODEL}_GT"] <= MAX_GT_FILTER
    ]

    rows_match.sort(key=lambda r: r["score_vis"], reverse=True)

    COPY_DIR_TOP.mkdir(parents=True, exist_ok=True)
    for r in rows_match[:COPY_TOPK]:
        safe_copy(IMAGES_DIR / r["image"], COPY_DIR_TOP / r["image"])

    # ================== BottomK ==================
    rows_bottom = [
        r for r in rows
        if r["match"] == 1
        and r[f"{MAIN_MODEL}_GT"] >= MIN_GT_FILTER
        and r[f"{MAIN_MODEL}_GT"] <= MAX_GT_FILTER
    ]
    rows_bottom.sort(key=lambda r: abs(r["delta_recall_main_base"]))

    COPY_DIR_BOT.mkdir(parents=True, exist_ok=True)
    for r in rows_bottom[:COPY_BOTK]:
        safe_copy(IMAGES_DIR / r["image"], COPY_DIR_BOT / r["image"])

    # ================== 统计输出 ==================
    print(f"[OK] Copied top {min(COPY_TOPK, len(rows_match))} images to -> {COPY_DIR_TOP}")
    print(f"[OK] Copied bottom {min(COPY_BOTK, len(rows_bottom))} images to -> {COPY_DIR_BOT}")

    print("\n===== Summary =====")
    print(f"Total images                    : {len(rows)}")
    print(f"GT >= {MIN_GT_MATCH} and order matched : {sum(r['match'] for r in rows)}")
    print(f"A > B                           : {sum(r['match_AB'] for r in rows)}")
    print(f"B > C                           : {sum(r['match_BC'] for r in rows)}")
    print(f"C > E                           : {sum(r['match_CE'] for r in rows)}")
    print(f"B > D                           : {sum(r['match_BD'] for r in rows)}")
    print(f"D > E                           : {sum(r['match_DE'] for r in rows)}")
    print(f"Top candidates                  : {len(rows_match)}")
    print(f"Bottom candidates               : {len(rows_bottom)}")
    print(f"CSV path                        : {OUT_CSV}")


if __name__ == "__main__":
    main()