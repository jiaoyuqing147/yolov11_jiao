import os
import csv
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
#--------使用说明 & 可调参数
'''
IOU_THR：默认 0.5（AP@0.5 / PRF@0.5）。你可以改为 0.5:0.95 多阈值做平均（需要把 ap50_from_preds 循环不同阈值求均值）。
CONF_THR_PRF：用于计算 P/R/F1 的置信度阈值（不影响 AP 的计算）。
COPY_TOPK / COPY_BOTTOMK：设置为 0 则不拷贝；否则会把 A>B 最明显 和 B>A 最明显 的图片复制出来，便于肉眼对比。
标签格式：GT 为 class cx cy w h，预测为 class cx cy w h conf，都是归一化。
'''


# ========== 配置 ==========
IMAGES_DIR = Path(r"E:\DataSets\tt100k_2021\yolojack\images\val")  # 图像目录（用于拿到宽高）
GT_LABELS = Path(r"E:\DataSets\tt100k_2021\yolojack\labels\val")  # GT 标签目录（YOLO txt，无 conf）
PRED_A = Path(r"E:\DataSets\tt100k_2021result\yolo11_train")  # 模型A 预测标签目录（有 conf）
PRED_B = Path(r"E:\DataSets\tt100k_2021result\yolo11_WIOU+BCELoss_train")  # 模型B 预测标签目录（有 conf）

OUT_CSV = Path(r"E:\DataSets\tt100k_2021result\AB_compare_per_image.csv")
COPY_TOPK = 10  # A 明显优于 B 的前 K 张（按 ΔF1 排序），0 表示不拷贝
COPY_BOTTOMK = 10  # B 明显优于 A 的前 K 张（按 -ΔF1 排序）
COPY_DIR_TOP = Path(r"E:\DataSets\tt100k_2021result\TOP_A_over_B")
COPY_DIR_BOT = Path(r"E:\DataSets\tt100k_2021result\TOP_B_over_A")

IOU_THR = 0.50  # IoU 阈值
CONF_THR_PRF = 0.25  # 计算 Precision/Recall/F1 时的置信度阈值
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# =========================


def yolo_to_xyxy(norm_box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    """(cx, cy, w, h) normalized -> (x1, y1, x2, y2) absolute pixels"""
    cx, cy, bw, bh = norm_box
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1);
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_txt(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    GT: class cx cy w h  (no conf)
    return list of (cls, cx, cy, w, h)
    """
    out = []
    if not txt_path.exists():
        return out
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            out.append((cls, cx, cy, bw, bh))
    return out


def load_pred_txt(txt_path: Path) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Pred: class cx cy w h conf
    return list of (cls, cx, cy, w, h, conf)
    """
    out = []
    if not txt_path.exists():
        return out
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh, conf = map(float, parts[1:6])
            out.append((cls, cx, cy, bw, bh, conf))
    return out


def match_by_iou(
        gts_xyxy: List[Tuple[int, Tuple[float, float, float, float]]],
        preds_xyxy: List[Tuple[int, Tuple[float, float, float, float], float]],
        iou_thr: float
) -> Tuple[List[int], List[int], List[int]]:
    """
    逐预测（按 conf 降序）与 GT 做一对一匹配（类别必须一致，IoU≥阈值）。
    返回：tp_idx, fp_idx, fn_idx（分别是预测索引/GT索引）
    """
    # 按置信度排序（高->低）
    order = sorted(range(len(preds_xyxy)), key=lambda i: preds_xyxy[i][2], reverse=True)
    gt_used = [False] * len(gts_xyxy)

    tp_idx, fp_idx = [], []
    for pi in order:
        p_cls, p_box, _ = preds_xyxy[pi]
        best_iou, best_gt = 0.0, -1
        for gi, (g_cls, g_box) in enumerate(gts_xyxy):
            if gt_used[gi]:  # 已匹配过
                continue
            if g_cls != p_cls:
                continue
            iou = iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou, best_gt = iou, gi
        if best_iou >= iou_thr and best_gt >= 0:
            gt_used[best_gt] = True
            tp_idx.append(pi)
        else:
            fp_idx.append(pi)

    fn_idx = [gi for gi, used in enumerate(gt_used) if not used]  # 未匹配上的 GT
    return tp_idx, fp_idx, fn_idx


def prf_from_match(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def ap50_from_preds(
        gts_xyxy: List[Tuple[int, Tuple[float, float, float, float]]],
        preds_xyxy: List[Tuple[int, Tuple[float, float, float, float], float]],
        iou_thr: float
) -> float:
    """
    单图 AP@IoU_thr：
    - 将预测按 conf 降序逐个加入，构造 PR 曲线
    - 用梯形法计算面积
    """
    if len(preds_xyxy) == 0:
        return 0.0
    # 排序
    preds = sorted(preds_xyxy, key=lambda x: x[2], reverse=True)
    gt_flags = [False] * len(gts_xyxy)

    tps, fps = [], []
    tp_cum, fp_cum = 0, 0

    for p_cls, p_box, _conf in preds:
        # 找最佳匹配 GT
        best_iou, best_gt = 0.0, -1
        for gi, (g_cls, g_box) in enumerate(gts_xyxy):
            if gt_flags[gi]:
                continue
            if g_cls != p_cls:
                continue
            iou = iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou, best_gt = iou, gi
        if best_iou >= iou_thr and best_gt >= 0:
            gt_flags[best_gt] = True
            tp_cum += 1
        else:
            fp_cum += 1

        tps.append(tp_cum)
        fps.append(fp_cum)

    # 构造 P-R 曲线
    precisions, recalls = [], []
    total_gt = len(gts_xyxy)
    if total_gt == 0:
        # 若该图无 GT，则 AP 定义为 0（或跳过）；这里给 0
        return 0.0

    for tp_i, fp_i in zip(tps, fps):
        prec = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        rec = tp_i / total_gt
        precisions.append(prec)
        recalls.append(rec)

    # 排序（recall 单调非降）
    pts = sorted(zip(recalls, precisions))
    recalls, precisions = [p[0] for p in pts], [p[1] for p in pts]

    # 插值单调包络（可选，这里直接做简单梯形积分）
    ap = 0.0
    prev_r, prev_p = 0.0, 1.0
    for r, p in zip(recalls, precisions):
        ap += (r - prev_r) * ((p + prev_p) / 2)
        prev_r, prev_p = r, p
    # 若最后 recall < 1，可以补尾巴（可选）；这里保持常见实现：按现有曲线面积
    return ap


def evaluate_one_image(
        img_path: Path, gt_dir: Path, pred_dir: Path, conf_thr: float, iou_thr: float
) -> Dict[str, float]:
    """
    读取单图 GT / Pred，计算 Precision/Recall/F1（按 conf_thr 过滤）与 AP@IoU_thr
    """
    stem = img_path.stem
    # 找 GT/PRED 对应的 txt
    gt_txt = gt_dir / f"{stem}.txt"
    pred_txt = pred_dir / f"{stem}.txt"

    # 图像尺寸
    with Image.open(img_path) as im:
        W, H = im.size

    # 读入
    gts = load_gt_txt(gt_txt)  # (cls,cx,cy,w,h)
    preds = load_pred_txt(pred_txt)  # (cls,cx,cy,w,h,conf)

    # 归一化 -> xyxy 像素坐标
    gts_xyxy = [(cls, yolo_to_xyxy((cx, cy, bw, bh), W, H)) for (cls, cx, cy, bw, bh) in gts]
    preds_all = [(cls, yolo_to_xyxy((cx, cy, bw, bh), W, H), conf) for (cls, cx, cy, bw, bh, conf) in preds]
    # for PR/F1：按阈值过滤
    preds_keep = [(c, b, s) for (c, b, s) in preds_all if s >= conf_thr]

    # 匹配并统计
    tp_idx, fp_idx, fn_idx = match_by_iou(gts_xyxy, preds_keep, iou_thr)
    tp, fp, fn = len(tp_idx), len(fp_idx), len(fn_idx)
    prec, rec, f1 = prf_from_match(tp, fp, fn)

    # AP（使用所有预测，按置信度排序逐点累积）
    ap50 = ap50_from_preds(gts_xyxy, preds_all, iou_thr)

    return {
        "n_gt": len(gts_xyxy),
        "n_pred": len(preds_keep),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "ap50": ap50
    }


def main():
    # 收集所有图片
    img_list = []
    for ext in IMG_EXTS:
        img_list += list(IMAGES_DIR.rglob(f"*{ext}"))
    img_list.sort()

    rows = []
    for img_path in img_list:
        mA = evaluate_one_image(img_path, GT_LABELS, PRED_A, CONF_THR_PRF, IOU_THR)
        mB = evaluate_one_image(img_path, GT_LABELS, PRED_B, CONF_THR_PRF, IOU_THR)

        row = {
            "image": img_path.name,
            "n_gt": mA["n_gt"],

            "A_npred": mA["n_pred"],
            "A_P": mA["precision"],
            "A_R": mA["recall"],
            "A_F1": mA["f1"],
            "A_AP50": mA["ap50"],

            "B_npred": mB["n_pred"],
            "B_P": mB["precision"],
            "B_R": mB["recall"],
            "B_F1": mB["f1"],
            "B_AP50": mB["ap50"],
        }
        row["dF1"] = row["A_F1"] - row["B_F1"]
        row["dAP50"] = row["A_AP50"] - row["B_AP50"]
        rows.append(row)

    # 写 CSV（按 ΔF1 从大到小排序）
    rows_sorted = sorted(rows, key=lambda r: r["dF1"], reverse=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "n_gt",
            "A_npred", "A_P", "A_R", "A_F1", "A_AP50",
            "B_npred", "B_P", "B_R", "B_F1", "B_AP50",
            "dF1", "dAP50"
        ])
        for r in rows_sorted:
            w.writerow([
                r["image"], r["n_gt"],
                r["A_npred"], f"{r['A_P']:.6f}", f"{r['A_R']:.6f}", f"{r['A_F1']:.6f}", f"{r['A_AP50']:.6f}",
                r["B_npred"], f"{r['B_P']:.6f}", f"{r['B_R']:.6f}", f"{r['B_F1']:.6f}", f"{r['B_AP50']:.6f}",
                f"{r['dF1']:.6f}", f"{r['dAP50']:.6f}",
            ])

    print(f"[OK] Saved per-image comparison CSV -> {OUT_CSV}")

    # 可选：拷贝 Top-K / Bottom-K
    if COPY_TOPK > 0:
        COPY_DIR_TOP.mkdir(parents=True, exist_ok=True)
        for r in rows_sorted[:COPY_TOPK]:
            src = next((p for p in (IMAGES_DIR / r["image"]).parent.rglob(r["image"])), None)
            if src and src.exists():
                shutil.copy(src, COPY_DIR_TOP / r["image"])
        print(f"[OK] Copied TOP {COPY_TOPK} images (A > B) -> {COPY_DIR_TOP}")

    if COPY_BOTTOMK > 0:
        COPY_DIR_BOT.mkdir(parents=True, exist_ok=True)
        for r in rows_sorted[-COPY_BOTTOMK:]:
            src = next((p for p in (IMAGES_DIR / r["image"]).parent.rglob(r["image"])), None)
            if src and src.exists():
                shutil.copy(src, COPY_DIR_BOT / r["image"])
        print(f"[OK] Copied B over A TOP {COPY_BOTTOMK} images -> {COPY_DIR_BOT}")


if __name__ == "__main__":
    main()
