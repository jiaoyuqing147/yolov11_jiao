from pathlib import Path
import pandas as pd


# =====================================================
# 自动定位项目根目录
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 当前脚本位于：
# 项目根目录 / 回应审稿人2的鲁棒性测试 / xxx.py
# 所以项目根目录 = 当前脚本目录的上一级
PROJECT_ROOT = SCRIPT_DIR.parent


# =====================================================
# 配置
# =====================================================

CONDITION = "Fog"

RESULT_ROOT = PROJECT_ROOT / "valsRobust" / "robust_tt100k130"

# Original 预测结果
YOLO_ORIGINAL_LABEL_DIR = (
    RESULT_ROOT /
    "YOLOv11_Original" /
    "labels"
)

ECAFA_ORIGINAL_LABEL_DIR = (
    RESULT_ROOT /
    "ECAFA_YOLO_Original" /
    "labels"
)

# Fog 预测结果
YOLO_FOG_LABEL_DIR = (
    RESULT_ROOT /
    "YOLOv11_Fog" /
    "labels"
)

ECAFA_FOG_LABEL_DIR = (
    RESULT_ROOT /
    "ECAFA_YOLO_Fog" /
    "labels"
)

# 数据集根目录
DATA_ROOT = Path(
    r"F:\DataSets\tt100k\yolojack"
)

# 原始图像目录
ORIGINAL_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    "val"
)

# 雾气图像目录
FOG_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    "val_fog"
)

# GT标签目录
GT_LABEL_DIR = (
    DATA_ROOT /
    "labels" /
    "val"
)

# 输出CSV
OUT_CSV = (
    RESULT_ROOT /
    "select_Fog_cases_both_degraded_tp_match.csv"
)


# =====================================================
# 筛选参数
# =====================================================

# 最终画图显示框的置信度阈值
DISPLAY_CONF_THRESH = 0.25

# 高置信度框阈值，仅作为辅助参考
HIGH_CONF_THRESH = 0.40

# NMS去重IoU阈值
NMS_IOU_THRESH = 0.50

# True：不管类别，只要两个框高度重叠，就认为是重复框
# 用于可视化筛图更合适
CLASS_AGNOSTIC_NMS = True

# TP匹配IoU阈值
MATCH_IOU_THRESH = 0.50

# TP匹配是否要求类别一致
MATCH_CLASS_AWARE = True


# =====================================================
# 核心筛选条件
# =====================================================

MIN_GT_BOXES = 4

# Original 下两个模型都应该能检测到多个目标
MIN_YOLO_ORIGINAL_TP = 2
MIN_ECAFA_ORIGINAL_TP = 3

# Fog 下两个模型都要出现衰退
MIN_YOLO_DROP = 1
MIN_ECAFA_DROP = 1

# Fog 下 ECAFA 至少还要保留一定检测能力
MIN_ECAFA_FOG_TP = 2

# Fog 下 ECAFA 要比 YOLO 多检出至少几个真实目标
MIN_FOG_TP_GAIN = 1

# 避免图上框太乱，可按需要调整
MAX_YOLO_FOG_SHOW_BOXES = 30
MAX_ECAFA_FOG_SHOW_BOXES = 30


# =====================================================
# 读取txt
# =====================================================

def read_pred_txt(txt_path):
    """
    YOLO predict txt格式:
    cls x y w h conf
    """
    if not txt_path.exists():
        return []

    preds = []

    with open(txt_path, "r", encoding="utf-8") as f:
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

            preds.append(
                {
                    "cls": cls_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                }
            )

    return preds


def read_gt_txt(txt_path):
    """
    GT YOLO txt格式:
    cls x y w h
    """
    if not txt_path.exists():
        return []

    gts = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            gts.append(
                {
                    "cls": cls_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    return gts


# =====================================================
# 基础工具函数
# =====================================================

def filter_by_conf(preds, conf_thresh):
    return [
        p for p in preds
        if p["conf"] >= conf_thresh
    ]


def mean_conf(preds):
    if len(preds) == 0:
        return 0.0

    return sum(p["conf"] for p in preds) / len(preds)


def max_conf(preds):
    if len(preds) == 0:
        return 0.0

    return max(p["conf"] for p in preds)


def get_image_path(image_dir, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)

    return ""


# =====================================================
# IoU 与 NMS
# =====================================================

def xywh_to_xyxy(box):
    """
    YOLO normalized xywh -> normalized xyxy
    """
    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return x1, y1, x2, y2


def box_iou(box1, box2):
    """
    计算两个 normalized bbox 的 IoU
    """
    x1_1, y1_1, x2_1, y2_1 = xywh_to_xyxy(box1)
    x1_2, y1_2, x2_2, y2_2 = xywh_to_xyxy(box2)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def nms_deduplicate_preds(
        preds,
        iou_thresh=0.5,
        class_agnostic=True):
    """
    对预测框做 NMS 去重。

    class_agnostic=True:
        不区分类别，只要高度重叠，就认为是重复框。
        用于论文可视化筛图更合适。
    """
    if len(preds) == 0:
        return []

    preds = sorted(
        preds,
        key=lambda p: p["conf"],
        reverse=True
    )

    kept = []

    while len(preds) > 0:

        best = preds.pop(0)
        kept.append(best)

        remain = []

        for p in preds:

            if not class_agnostic:
                if p["cls"] != best["cls"]:
                    remain.append(p)
                    continue

            iou = box_iou(best, p)

            if iou < iou_thresh:
                remain.append(p)

        preds = remain

    return kept


def prepare_preds_for_match(
        preds_raw,
        conf_thresh=0.25):
    """
    用于匹配和统计的预测框：
    1. conf筛选
    2. NMS去重
    """
    preds = filter_by_conf(
        preds_raw,
        conf_thresh
    )

    preds = nms_deduplicate_preds(
        preds,
        iou_thresh=NMS_IOU_THRESH,
        class_agnostic=CLASS_AGNOSTIC_NMS
    )

    return preds


# =====================================================
# 预测框与GT匹配，统计TP / FP / FN
# =====================================================

def match_predictions_to_gt(
        preds,
        gts,
        iou_thresh=0.5,
        class_aware=True):
    """
    预测框与GT一对一匹配。

    匹配规则：
    - 预测框按conf从高到低排序；
    - 每个GT最多只能被匹配一次；
    - IoU >= iou_thresh；
    - class_aware=True 时要求类别一致。

    返回：
    - tp: 正确检测数量
    - fp: 未匹配预测框数量
    - fn: 未被检测GT数量
    - matched_gt_indices: 被匹配到的GT索引
    """
    if len(gts) == 0:
        return {
            "tp": 0,
            "fp": len(preds),
            "fn": 0,
            "matched_gt_indices": [],
            "matched_pred_indices": [],
            "matched_ious": [],
            "matched_confs": [],
        }

    if len(preds) == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(gts),
            "matched_gt_indices": [],
            "matched_pred_indices": [],
            "matched_ious": [],
            "matched_confs": [],
        }

    preds_sorted = sorted(
        list(enumerate(preds)),
        key=lambda item: item[1]["conf"],
        reverse=True
    )

    matched_gt_indices = set()
    matched_pred_indices = []
    matched_ious = []
    matched_confs = []

    tp = 0

    for pred_idx, pred in preds_sorted:

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gts):

            if gt_idx in matched_gt_indices:
                continue

            if class_aware:
                if pred["cls"] != gt["cls"]:
                    continue

            iou = box_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_thresh:

            matched_gt_indices.add(best_gt_idx)
            matched_pred_indices.append(pred_idx)
            matched_ious.append(best_iou)
            matched_confs.append(pred["conf"])
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_gt_indices": sorted(list(matched_gt_indices)),
        "matched_pred_indices": matched_pred_indices,
        "matched_ious": matched_ious,
        "matched_confs": matched_confs,
    }


def calc_stats(preds, gts):
    """
    对一组预测框和GT计算统计结果。
    """
    match_result = match_predictions_to_gt(
        preds=preds,
        gts=gts,
        iou_thresh=MATCH_IOU_THRESH,
        class_aware=MATCH_CLASS_AWARE
    )

    tp = match_result["tp"]
    fp = match_result["fp"]
    fn = match_result["fn"]

    recall = tp / len(gts) if len(gts) > 0 else 0.0
    precision = tp / len(preds) if len(preds) > 0 else 0.0

    matched_confs = match_result["matched_confs"]
    matched_ious = match_result["matched_ious"]

    matched_mean_conf = (
        sum(matched_confs) / len(matched_confs)
        if len(matched_confs) > 0 else 0.0
    )

    matched_mean_iou = (
        sum(matched_ious) / len(matched_ious)
        if len(matched_ious) > 0 else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "show_boxes": len(preds),
        "mean_conf": mean_conf(preds),
        "max_conf": max_conf(preds),
        "matched_mean_conf": matched_mean_conf,
        "matched_mean_iou": matched_mean_iou,
        "matched_gt_indices": match_result["matched_gt_indices"],
    }


# =====================================================
# 检查路径
# =====================================================

required_dirs = [
    YOLO_ORIGINAL_LABEL_DIR,
    ECAFA_ORIGINAL_LABEL_DIR,
    YOLO_FOG_LABEL_DIR,
    ECAFA_FOG_LABEL_DIR,
    GT_LABEL_DIR,
    ORIGINAL_IMAGE_DIR,
    FOG_IMAGE_DIR,
]

for d in required_dirs:
    if not d.exists():
        print(f"[WARNING] Path not found: {d}")


# =====================================================
# 收集所有图像ID
# =====================================================

all_stems = set()

for p in GT_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)

for p in YOLO_ORIGINAL_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)

for p in ECAFA_ORIGINAL_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)

for p in YOLO_FOG_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)

for p in ECAFA_FOG_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)


# =====================================================
# 主循环
# =====================================================

rows = []

for stem in sorted(all_stems):

    gt_txt = GT_LABEL_DIR / f"{stem}.txt"

    yolo_original_txt = YOLO_ORIGINAL_LABEL_DIR / f"{stem}.txt"
    ecafa_original_txt = ECAFA_ORIGINAL_LABEL_DIR / f"{stem}.txt"

    yolo_fog_txt = YOLO_FOG_LABEL_DIR / f"{stem}.txt"
    ecafa_fog_txt = ECAFA_FOG_LABEL_DIR / f"{stem}.txt"

    gt_boxes = read_gt_txt(gt_txt)
    gt_n = len(gt_boxes)

    if gt_n < MIN_GT_BOXES:
        continue

    # 读取原始预测
    yolo_original_raw = read_pred_txt(yolo_original_txt)
    ecafa_original_raw = read_pred_txt(ecafa_original_txt)

    yolo_fog_raw = read_pred_txt(yolo_fog_txt)
    ecafa_fog_raw = read_pred_txt(ecafa_fog_txt)

    # conf >= 0.25 + NMS 去重
    yolo_original_preds = prepare_preds_for_match(
        yolo_original_raw,
        DISPLAY_CONF_THRESH
    )

    ecafa_original_preds = prepare_preds_for_match(
        ecafa_original_raw,
        DISPLAY_CONF_THRESH
    )

    yolo_fog_preds = prepare_preds_for_match(
        yolo_fog_raw,
        DISPLAY_CONF_THRESH
    )

    ecafa_fog_preds = prepare_preds_for_match(
        ecafa_fog_raw,
        DISPLAY_CONF_THRESH
    )

    # 高置信度框，辅助观察
    yolo_original_high_preds = prepare_preds_for_match(
        yolo_original_raw,
        HIGH_CONF_THRESH
    )

    ecafa_original_high_preds = prepare_preds_for_match(
        ecafa_original_raw,
        HIGH_CONF_THRESH
    )

    yolo_fog_high_preds = prepare_preds_for_match(
        yolo_fog_raw,
        HIGH_CONF_THRESH
    )

    ecafa_fog_high_preds = prepare_preds_for_match(
        ecafa_fog_raw,
        HIGH_CONF_THRESH
    )

    # TP匹配统计
    yolo_original_stats = calc_stats(
        yolo_original_preds,
        gt_boxes
    )

    ecafa_original_stats = calc_stats(
        ecafa_original_preds,
        gt_boxes
    )

    yolo_fog_stats = calc_stats(
        yolo_fog_preds,
        gt_boxes
    )

    ecafa_fog_stats = calc_stats(
        ecafa_fog_preds,
        gt_boxes
    )

    # 高置信度TP统计
    yolo_original_high_stats = calc_stats(
        yolo_original_high_preds,
        gt_boxes
    )

    ecafa_original_high_stats = calc_stats(
        ecafa_original_high_preds,
        gt_boxes
    )

    yolo_fog_high_stats = calc_stats(
        yolo_fog_high_preds,
        gt_boxes
    )

    ecafa_fog_high_stats = calc_stats(
        ecafa_fog_high_preds,
        gt_boxes
    )

    # =================================================
    # 计算 Original -> Fog 衰退
    # =================================================

    yolo_original_tp = yolo_original_stats["tp"]
    yolo_fog_tp = yolo_fog_stats["tp"]
    yolo_drop = yolo_original_tp - yolo_fog_tp

    ecafa_original_tp = ecafa_original_stats["tp"]
    ecafa_fog_tp = ecafa_fog_stats["tp"]
    ecafa_drop = ecafa_original_tp - ecafa_fog_tp

    fog_tp_gain = ecafa_fog_tp - yolo_fog_tp

    yolo_retention = (
        yolo_fog_tp / yolo_original_tp
        if yolo_original_tp > 0 else 0.0
    )

    ecafa_retention = (
        ecafa_fog_tp / ecafa_original_tp
        if ecafa_original_tp > 0 else 0.0
    )

    retention_gain = ecafa_retention - yolo_retention

    # =================================================
    # 核心筛选：
    # Original两个模型都能检测；
    # Fog两个模型都衰退；
    # Fog下ECAFA保留更多TP。
    # =================================================

    if yolo_original_tp < MIN_YOLO_ORIGINAL_TP:
        continue

    if ecafa_original_tp < MIN_ECAFA_ORIGINAL_TP:
        continue

    if yolo_drop < MIN_YOLO_DROP:
        continue

    if ecafa_drop < MIN_ECAFA_DROP:
        continue

    if ecafa_fog_tp < MIN_ECAFA_FOG_TP:
        continue

    if fog_tp_gain < MIN_FOG_TP_GAIN:
        continue

    if yolo_fog_stats["show_boxes"] > MAX_YOLO_FOG_SHOW_BOXES:
        continue

    if ecafa_fog_stats["show_boxes"] > MAX_ECAFA_FOG_SHOW_BOXES:
        continue

    # =================================================
    # 综合分数
    # 目标：
    # 1. Fog下ECAFA比YOLO多保留TP；
    # 2. YOLO在Fog下有明显漏检；
    # 3. ECAFA也衰退，但仍保留较多TP；
    # 4. 图中GT数量适中；
    # 5. 避免ECAFA误检过多。
    # =================================================

    ecafa_fog_fp_penalty = max(
        0,
        ecafa_fog_stats["fp"] - yolo_fog_stats["fp"]
    )

    score = (
        fog_tp_gain * 100
        +
        ecafa_fog_tp * 35
        +
        yolo_drop * 35
        +
        ecafa_drop * 15
        +
        gt_n * 5
        +
        retention_gain * 50
        -
        ecafa_fog_fp_penalty * 8
    )

    original_image_path = get_image_path(
        ORIGINAL_IMAGE_DIR,
        stem
    )

    fog_image_path = get_image_path(
        FOG_IMAGE_DIR,
        stem
    )

    rows.append(
        {
            "image": stem,
            "gt_boxes": gt_n,

            "original_image_path": original_image_path,
            "fog_image_path": fog_image_path,

            # YOLO Original
            "yolo_original_tp": yolo_original_stats["tp"],
            "yolo_original_fp": yolo_original_stats["fp"],
            "yolo_original_fn": yolo_original_stats["fn"],
            "yolo_original_recall": yolo_original_stats["recall"],
            "yolo_original_precision": yolo_original_stats["precision"],
            "yolo_original_show_boxes": yolo_original_stats["show_boxes"],
            "yolo_original_mean_conf": yolo_original_stats["mean_conf"],
            "yolo_original_matched_mean_conf": yolo_original_stats["matched_mean_conf"],
            "yolo_original_matched_mean_iou": yolo_original_stats["matched_mean_iou"],

            # YOLO Fog
            "yolo_fog_tp": yolo_fog_stats["tp"],
            "yolo_fog_fp": yolo_fog_stats["fp"],
            "yolo_fog_fn": yolo_fog_stats["fn"],
            "yolo_fog_recall": yolo_fog_stats["recall"],
            "yolo_fog_precision": yolo_fog_stats["precision"],
            "yolo_fog_show_boxes": yolo_fog_stats["show_boxes"],
            "yolo_fog_mean_conf": yolo_fog_stats["mean_conf"],
            "yolo_fog_matched_mean_conf": yolo_fog_stats["matched_mean_conf"],
            "yolo_fog_matched_mean_iou": yolo_fog_stats["matched_mean_iou"],
            "yolo_drop": yolo_drop,
            "yolo_retention": yolo_retention,

            # ECAFA Original
            "ecafa_original_tp": ecafa_original_stats["tp"],
            "ecafa_original_fp": ecafa_original_stats["fp"],
            "ecafa_original_fn": ecafa_original_stats["fn"],
            "ecafa_original_recall": ecafa_original_stats["recall"],
            "ecafa_original_precision": ecafa_original_stats["precision"],
            "ecafa_original_show_boxes": ecafa_original_stats["show_boxes"],
            "ecafa_original_mean_conf": ecafa_original_stats["mean_conf"],
            "ecafa_original_matched_mean_conf": ecafa_original_stats["matched_mean_conf"],
            "ecafa_original_matched_mean_iou": ecafa_original_stats["matched_mean_iou"],

            # ECAFA Fog
            "ecafa_fog_tp": ecafa_fog_stats["tp"],
            "ecafa_fog_fp": ecafa_fog_stats["fp"],
            "ecafa_fog_fn": ecafa_fog_stats["fn"],
            "ecafa_fog_recall": ecafa_fog_stats["recall"],
            "ecafa_fog_precision": ecafa_fog_stats["precision"],
            "ecafa_fog_show_boxes": ecafa_fog_stats["show_boxes"],
            "ecafa_fog_mean_conf": ecafa_fog_stats["mean_conf"],
            "ecafa_fog_matched_mean_conf": ecafa_fog_stats["matched_mean_conf"],
            "ecafa_fog_matched_mean_iou": ecafa_fog_stats["matched_mean_iou"],
            "ecafa_drop": ecafa_drop,
            "ecafa_retention": ecafa_retention,

            # 对比指标
            "fog_tp_gain": fog_tp_gain,
            "retention_gain": retention_gain,

            # 高置信度辅助指标
            "high_conf_thresh": HIGH_CONF_THRESH,
            "yolo_original_high_tp": yolo_original_high_stats["tp"],
            "yolo_fog_high_tp": yolo_fog_high_stats["tp"],
            "ecafa_original_high_tp": ecafa_original_high_stats["tp"],
            "ecafa_fog_high_tp": ecafa_fog_high_stats["tp"],
            "high_fog_tp_gain": (
                ecafa_fog_high_stats["tp"] -
                yolo_fog_high_stats["tp"]
            ),

            # 参数记录
            "display_conf_thresh": DISPLAY_CONF_THRESH,
            "match_iou_thresh": MATCH_IOU_THRESH,
            "nms_iou_thresh": NMS_IOU_THRESH,
            "class_agnostic_nms": CLASS_AGNOSTIC_NMS,
            "match_class_aware": MATCH_CLASS_AWARE,

            "score": score,

            # 文件路径
            "gt_txt": str(gt_txt),
            "yolo_original_txt": str(yolo_original_txt),
            "yolo_fog_txt": str(yolo_fog_txt),
            "ecafa_original_txt": str(ecafa_original_txt),
            "ecafa_fog_txt": str(ecafa_fog_txt),
        }
    )


# =====================================================
# 保存结果
# =====================================================

df = pd.DataFrame(rows)

if len(df) == 0:

    print()
    print("[WARNING] No cases found.")
    print("You can try lowering these thresholds:")
    print("  MIN_YOLO_ORIGINAL_TP")
    print("  MIN_ECAFA_ORIGINAL_TP")
    print("  MIN_YOLO_DROP")
    print("  MIN_ECAFA_DROP")
    print("  MIN_FOG_TP_GAIN")
    print()

else:

    df = df.sort_values(
        by=[
            "score",
            "fog_tp_gain",
            "ecafa_fog_tp",
            "yolo_drop",
            "ecafa_drop",
            "gt_boxes",
            "retention_gain",
        ],
        ascending=False
    )

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 120)
    print("Top 30 cases: Original both good, Fog both degraded, ECAFA retains more detections")
    print("=" * 120)

    show_cols = [
        "image",
        "gt_boxes",

        "yolo_original_tp",
        "yolo_fog_tp",
        "yolo_drop",

        "ecafa_original_tp",
        "ecafa_fog_tp",
        "ecafa_drop",

        "fog_tp_gain",
        "yolo_fog_fp",
        "ecafa_fog_fp",
        "yolo_retention",
        "ecafa_retention",
        "score",
    ]

    print(
        df[show_cols]
        .head(30)
        .to_string(index=False)
    )

    print()
    print(f"Saved to: {OUT_CSV}")
    print()

    print("=" * 120)
    print("Recommended first image:")
    print("=" * 120)

    first = df.iloc[0]

    print(f"Image ID: {first['image']}")
    print(f"Original image: {first['original_image_path']}")
    print(f"Fog image     : {first['fog_image_path']}")
    print()
    print(f"GT boxes: {first['gt_boxes']}")
    print(
        f"YOLOv11: Original TP {first['yolo_original_tp']} "
        f"-> Fog TP {first['yolo_fog_tp']} "
        f"(drop {first['yolo_drop']})"
    )
    print(
        f"ECAFA-YOLO: Original TP {first['ecafa_original_tp']} "
        f"-> Fog TP {first['ecafa_fog_tp']} "
        f"(drop {first['ecafa_drop']})"
    )
    print(f"Fog TP gain of ECAFA over YOLOv11: {first['fog_tp_gain']}")
    print()