import csv
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ================== 路径配置 ==================
IMAGES_DIR = Path(r"E:\DataSets\tt100k_2021\size_split_test\tiny\images")
GT_LABELS = Path(r"E:\DataSets\tt100k_2021\size_split_test\tiny\labels")

MODELS = {
    "baseline": Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\yolo11_train200"),
    "ours": Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200"),
    "ours_kd": Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation"),
}

# ================== 参数配置 ==================
IOU_THR = 0.50
CONF_THR = 0.25
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
EPS = 1e-6

# small 子集已经提前筛好了，所以这里只要求目标数够多
MIN_GT_BUCKET = 5

# 严格提升阈值
MIN_GAIN_BASE_TO_OURS = 0.10
MIN_GAIN_OURS_TO_KD = 0.03

# 输出目录
OUT_DIR = Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult\compare_strict_small")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "small_compare_strict.csv"
COPY_DIR = OUT_DIR / "TopK_vis"
COPY_DIR.mkdir(parents=True, exist_ok=True)

TOPK = 20


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
    返回:
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

    preds_keep = [p for p in preds_all if p[2] >= CONF_THR]

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
    }


def safe_copy(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
    else:
        print(f"[WARN] Source image not found: {src}")


def strictly_better(a, b, eps=1e-6):
    return a > b + eps


# ================== 主程序 ==================
def main():
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

        n_gt = per_model["baseline"]["n_gt"]
        base_r = per_model["baseline"]["recall"]
        ours_r = per_model["ours"]["recall"]
        kd_r = per_model["ours_kd"]["recall"]

        gain1 = ours_r - base_r
        gain2 = kd_r - ours_r

        # 严格满足 baseline < ours < ours_kd
        strict_order_ok = (
            strictly_better(ours_r, base_r, EPS)
            and strictly_better(kd_r, ours_r, EPS)
        )

        # 幅度阈值
        gain_ok = (
            gain1 >= MIN_GAIN_BASE_TO_OURS
            and gain2 >= MIN_GAIN_OURS_TO_KD
        )

        gt_ok = n_gt >= MIN_GT_BUCKET

        # 最终通过条件
        match_ok = strict_order_ok and gain_ok and gt_ok

        # 排序分数：优先 baseline→ours 提升，再看 ours→kd，再看最终 kd recall
        score = 2.0 * gain1 + 1.0 * gain2 + 0.5 * kd_r

        row = {
            "image": img_path.name,

            "GT": n_gt,

            "baseline_P": per_model["baseline"]["precision"],
            "baseline_R": base_r,
            "baseline_F1": per_model["baseline"]["f1"],
            "baseline_TP": per_model["baseline"]["tp"],
            "baseline_FP": per_model["baseline"]["fp"],
            "baseline_FN": per_model["baseline"]["fn"],

            "ours_P": per_model["ours"]["precision"],
            "ours_R": ours_r,
            "ours_F1": per_model["ours"]["f1"],
            "ours_TP": per_model["ours"]["tp"],
            "ours_FP": per_model["ours"]["fp"],
            "ours_FN": per_model["ours"]["fn"],

            "ours_kd_P": per_model["ours_kd"]["precision"],
            "ours_kd_R": kd_r,
            "ours_kd_F1": per_model["ours_kd"]["f1"],
            "ours_kd_TP": per_model["ours_kd"]["tp"],
            "ours_kd_FP": per_model["ours_kd"]["fp"],
            "ours_kd_FN": per_model["ours_kd"]["fn"],

            "gain_ours_vs_base": gain1,
            "gain_kd_vs_ours": gain2,

            "gt_ok": int(gt_ok),
            "strict_order_ok": int(strict_order_ok),
            "gain_ok": int(gain_ok),
            "match_ok": int(match_ok),
            "score": score,
        }

        rows.append(row)

    # 保存全部 CSV
    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] CSV saved -> {OUT_CSV}")
    else:
        print("[WARN] No rows generated.")
        return

    # 只保留严格满足条件的图
    rows_match = [r for r in rows if r["match_ok"] == 1]
    rows_match.sort(key=lambda r: r["score"], reverse=True)

    # 复制 TopK 图片
    for r in rows_match[:TOPK]:
        safe_copy(IMAGES_DIR / r["image"], COPY_DIR / r["image"])

    print(f"[OK] Copied top {min(TOPK, len(rows_match))} images to -> {COPY_DIR}")

    # 统计输出
    print("\n===== Summary =====")
    print(f"Total images         : {len(rows)}")
    print(f"GT >= {MIN_GT_BUCKET}          : {sum(r['gt_ok'] for r in rows)}")
    print(f"Strict order matched : {sum(r['strict_order_ok'] for r in rows)}")
    print(f"Gain threshold match : {sum(r['gain_ok'] for r in rows)}")
    print(f"Final matched images : {len(rows_match)}")
    print(f"CSV path             : {OUT_CSV}")
    print(f"Copy dir             : {COPY_DIR}")


if __name__ == "__main__":
    main()