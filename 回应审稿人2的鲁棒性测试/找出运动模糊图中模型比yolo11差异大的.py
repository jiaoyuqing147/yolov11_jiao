from pathlib import Path
import pandas as pd


# =====================================================
# 自动定位项目根目录
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 如果脚本就在项目根目录
if (SCRIPT_DIR / "valsRobust").exists():
    PROJECT_ROOT = SCRIPT_DIR
else:
    # 如果脚本在：
    # 项目根目录 / 回应审稿人2的鲁棒性测试 / xxx.py
    # 那么项目根目录就是上一级
    PROJECT_ROOT = SCRIPT_DIR.parent


# =====================================================
# 配置
# =====================================================

# 可选：
# Rain / Fog / LowLight / MotionBlur / JPEG / Scale / Occlusion
CONDITION = "MotionBlur"

DATA_ROOT = Path(
    r"F:\DataSets\tt100k\yolojack"
)

RESULT_ROOT = (
    PROJECT_ROOT /
    "valsRobust" /
    "robust_tt100k130"
)

GT_LABEL_DIR = (
    DATA_ROOT /
    "labels" /
    "val"
)

ORIGINAL_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    "val"
)

DEGRADED_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    f"val_{CONDITION.lower()}"
)

YOLO_ORIGINAL_LABEL_DIR = (
    RESULT_ROOT /
    "YOLOv11_Original" /
    "labels"
)

YOLO_DEGRADED_LABEL_DIR = (
    RESULT_ROOT /
    f"YOLOv11_{CONDITION}" /
    "labels"
)

ECAFA_ORIGINAL_LABEL_DIR = (
    RESULT_ROOT /
    "ECAFA_YOLO_Original" /
    "labels"
)

ECAFA_DEGRADED_LABEL_DIR = (
    RESULT_ROOT /
    f"ECAFA_YOLO_{CONDITION}" /
    "labels"
)

OUT_CSV = (
    RESULT_ROOT /
    f"select_{CONDITION}_original_degraded_recall_compare.csv"
)


# =====================================================
# 筛选参数
# =====================================================

IOU_THRESH = 0.5
CONF_THRESH = 0.25

# GT数量不能太少
MIN_GT_BOXES = 4

# 原图上至少要有一定检测效果
MIN_YOLO_ORIGINAL_RECALL = 0.20
MIN_ECAFA_ORIGINAL_RECALL = 0.30

# 退化图上 ECAFA 相比 YOLO 至少提升多少 recall
MIN_DEGRADED_RECALL_GAIN = 0.15

# 避免预测框太离谱
# 例如 GT=5，最多允许预测框数量为 5*5=25
MAX_PRED_RATIO = 5.0

# 是否允许 YOLO 在退化图上完全检测不到
# 论文正文里建议 False，避免案例太极端
ALLOW_YOLO_DEGRADED_ZERO_TP = False


# =====================================================
# 打印路径，方便检查
# =====================================================

print("=" * 80)
print(f"SCRIPT_DIR              : {SCRIPT_DIR}")
print(f"PROJECT_ROOT            : {PROJECT_ROOT}")
print(f"RESULT_ROOT             : {RESULT_ROOT}")
print(f"CONDITION               : {CONDITION}")
print(f"GT_LABEL_DIR            : {GT_LABEL_DIR}")
print(f"ORIGINAL_IMAGE_DIR      : {ORIGINAL_IMAGE_DIR}")
print(f"DEGRADED_IMAGE_DIR      : {DEGRADED_IMAGE_DIR}")
print(f"YOLO_ORIGINAL_LABEL_DIR : {YOLO_ORIGINAL_LABEL_DIR}")
print(f"YOLO_DEGRADED_LABEL_DIR : {YOLO_DEGRADED_LABEL_DIR}")
print(f"ECAFA_ORIGINAL_LABEL_DIR: {ECAFA_ORIGINAL_LABEL_DIR}")
print(f"ECAFA_DEGRADED_LABEL_DIR: {ECAFA_DEGRADED_LABEL_DIR}")
print(f"OUT_CSV                 : {OUT_CSV}")
print("=" * 80)


# =====================================================
# 工具函数
# =====================================================

def check_dir(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )


def read_gt_txt(txt_path):
    """
    GT YOLO格式:
    cls x y w h
    """
    boxes = []

    if not txt_path.exists():
        return boxes

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            boxes.append(
                {
                    "cls": int(float(parts[0])),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                    "conf": 1.0,
                }
            )

    return boxes


def read_pred_txt(txt_path, conf_thresh=0.25):
    """
    Prediction YOLO格式:
    cls x y w h conf
    """
    boxes = []

    if not txt_path.exists():
        return boxes

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 6:
                continue

            conf = float(parts[5])

            if conf < conf_thresh:
                continue

            boxes.append(
                {
                    "cls": int(float(parts[0])),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                    "conf": conf,
                }
            )

    return boxes


def xywh_to_xyxy(box):
    """
    YOLO归一化 xywh 转 xyxy。
    这里仍然是归一化坐标，不需要图像尺寸。
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
    x11, y11, x12, y12 = xywh_to_xyxy(box1)
    x21, y21, x22, y22 = xywh_to_xyxy(box2)

    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def match_predictions_to_gt(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    按类别 + IoU 贪心匹配：
    - 类别必须一致
    - IoU >= threshold
    - 一个GT最多匹配一个预测框
    - 优先匹配高置信度预测
    """
    matched_gt = set()
    matched_pred = set()

    pred_order = sorted(
        range(len(pred_boxes)),
        key=lambda i: pred_boxes[i]["conf"],
        reverse=True
    )

    for pred_idx in pred_order:
        pred = pred_boxes[pred_idx]

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gt_boxes):

            if gt_idx in matched_gt:
                continue

            if pred["cls"] != gt["cls"]:
                continue

            iou = box_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_thresh:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - tp

    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "pred_boxes": len(pred_boxes),
    }


def get_image_path(image_dir, stem):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return ""


def safe_float(x):
    return round(float(x), 6)


# =====================================================
# 检查路径
# =====================================================

check_dir(GT_LABEL_DIR, "GT_LABEL_DIR")
check_dir(ORIGINAL_IMAGE_DIR, "ORIGINAL_IMAGE_DIR")
check_dir(DEGRADED_IMAGE_DIR, "DEGRADED_IMAGE_DIR")
check_dir(YOLO_ORIGINAL_LABEL_DIR, "YOLO_ORIGINAL_LABEL_DIR")
check_dir(YOLO_DEGRADED_LABEL_DIR, "YOLO_DEGRADED_LABEL_DIR")
check_dir(ECAFA_ORIGINAL_LABEL_DIR, "ECAFA_ORIGINAL_LABEL_DIR")
check_dir(ECAFA_DEGRADED_LABEL_DIR, "ECAFA_DEGRADED_LABEL_DIR")


# =====================================================
# 主程序
# =====================================================

all_stems = set()

for p in GT_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)

rows = []

for stem in sorted(all_stems):

    gt_txt = GT_LABEL_DIR / f"{stem}.txt"

    yolo_original_txt = (
        YOLO_ORIGINAL_LABEL_DIR /
        f"{stem}.txt"
    )

    yolo_degraded_txt = (
        YOLO_DEGRADED_LABEL_DIR /
        f"{stem}.txt"
    )

    ecafa_original_txt = (
        ECAFA_ORIGINAL_LABEL_DIR /
        f"{stem}.txt"
    )

    ecafa_degraded_txt = (
        ECAFA_DEGRADED_LABEL_DIR /
        f"{stem}.txt"
    )

    gt_boxes = read_gt_txt(gt_txt)
    gt_n = len(gt_boxes)

    if gt_n < MIN_GT_BOXES:
        continue

    yolo_original_preds = read_pred_txt(
        yolo_original_txt,
        conf_thresh=CONF_THRESH
    )

    yolo_degraded_preds = read_pred_txt(
        yolo_degraded_txt,
        conf_thresh=CONF_THRESH
    )

    ecafa_original_preds = read_pred_txt(
        ecafa_original_txt,
        conf_thresh=CONF_THRESH
    )

    ecafa_degraded_preds = read_pred_txt(
        ecafa_degraded_txt,
        conf_thresh=CONF_THRESH
    )

    yolo_original = match_predictions_to_gt(
        gt_boxes,
        yolo_original_preds,
        iou_thresh=IOU_THRESH
    )

    yolo_degraded = match_predictions_to_gt(
        gt_boxes,
        yolo_degraded_preds,
        iou_thresh=IOU_THRESH
    )

    ecafa_original = match_predictions_to_gt(
        gt_boxes,
        ecafa_original_preds,
        iou_thresh=IOU_THRESH
    )

    ecafa_degraded = match_predictions_to_gt(
        gt_boxes,
        ecafa_degraded_preds,
        iou_thresh=IOU_THRESH
    )

    # =================================================
    # 关键指标
    # =================================================

    yolo_recall_drop = (
        yolo_original["recall"]
        -
        yolo_degraded["recall"]
    )

    ecafa_recall_drop = (
        ecafa_original["recall"]
        -
        ecafa_degraded["recall"]
    )

    degraded_recall_gain = (
        ecafa_degraded["recall"]
        -
        yolo_degraded["recall"]
    )

    original_recall_gain = (
        ecafa_original["recall"]
        -
        yolo_original["recall"]
    )

    degraded_tp_gain = (
        ecafa_degraded["tp"]
        -
        yolo_degraded["tp"]
    )

    # =================================================
    # 筛选逻辑
    # =================================================

    # 原图上两个模型至少要能检出一些目标
    if yolo_original["recall"] < MIN_YOLO_ORIGINAL_RECALL:
        continue

    if ecafa_original["recall"] < MIN_ECAFA_ORIGINAL_RECALL:
        continue

    # 退化图上 ECAFA 要明显优于 YOLO
    if degraded_recall_gain < MIN_DEGRADED_RECALL_GAIN:
        continue

    # 正文图里不建议选择 YOLO 完全失败的极端样例
    if not ALLOW_YOLO_DEGRADED_ZERO_TP:
        if yolo_degraded["tp"] == 0:
            continue

    # 避免预测框过多，看起来像误检
    max_pred_boxes = gt_n * MAX_PRED_RATIO

    if yolo_degraded["pred_boxes"] > max_pred_boxes:
        continue

    if ecafa_degraded["pred_boxes"] > max_pred_boxes:
        continue

    if yolo_original["pred_boxes"] > max_pred_boxes:
        continue

    if ecafa_original["pred_boxes"] > max_pred_boxes:
        continue

    # =================================================
    # 综合分数
    # =================================================
    # 适合5列图的逻辑：
    # 1. 退化图中 ECAFA recall 高于 YOLO，最重要
    # 2. YOLO 从 Original 到 Degraded 有明显下降
    # 3. ECAFA 在 Degraded 下仍能保持较好 recall
    # 4. GT 数量适中偏多
    # 5. FP 较少
    # =================================================

    total_fp = (
        yolo_original["fp"]
        +
        yolo_degraded["fp"]
        +
        ecafa_original["fp"]
        +
        ecafa_degraded["fp"]
    )

    score = (
        degraded_recall_gain * 100
        +
        yolo_recall_drop * 50
        +
        ecafa_degraded["recall"] * 40
        +
        degraded_tp_gain * 5
        +
        gt_n * 2
        -
        total_fp * 0.5
    )

    rows.append(
        {
            "image": stem,
            "condition": CONDITION,
            "gt_boxes": gt_n,

            "yolo_original_tp": yolo_original["tp"],
            "yolo_original_fp": yolo_original["fp"],
            "yolo_original_fn": yolo_original["fn"],
            "yolo_original_recall": safe_float(yolo_original["recall"]),
            "yolo_original_precision": safe_float(yolo_original["precision"]),
            "yolo_original_pred_boxes": yolo_original["pred_boxes"],

            f"yolo_{CONDITION}_tp": yolo_degraded["tp"],
            f"yolo_{CONDITION}_fp": yolo_degraded["fp"],
            f"yolo_{CONDITION}_fn": yolo_degraded["fn"],
            f"yolo_{CONDITION}_recall": safe_float(yolo_degraded["recall"]),
            f"yolo_{CONDITION}_precision": safe_float(yolo_degraded["precision"]),
            f"yolo_{CONDITION}_pred_boxes": yolo_degraded["pred_boxes"],

            "ecafa_original_tp": ecafa_original["tp"],
            "ecafa_original_fp": ecafa_original["fp"],
            "ecafa_original_fn": ecafa_original["fn"],
            "ecafa_original_recall": safe_float(ecafa_original["recall"]),
            "ecafa_original_precision": safe_float(ecafa_original["precision"]),
            "ecafa_original_pred_boxes": ecafa_original["pred_boxes"],

            f"ecafa_{CONDITION}_tp": ecafa_degraded["tp"],
            f"ecafa_{CONDITION}_fp": ecafa_degraded["fp"],
            f"ecafa_{CONDITION}_fn": ecafa_degraded["fn"],
            f"ecafa_{CONDITION}_recall": safe_float(ecafa_degraded["recall"]),
            f"ecafa_{CONDITION}_precision": safe_float(ecafa_degraded["precision"]),
            f"ecafa_{CONDITION}_pred_boxes": ecafa_degraded["pred_boxes"],

            "yolo_recall_drop": safe_float(yolo_recall_drop),
            "ecafa_recall_drop": safe_float(ecafa_recall_drop),
            "degraded_recall_gain": safe_float(degraded_recall_gain),
            "original_recall_gain": safe_float(original_recall_gain),
            "degraded_tp_gain": degraded_tp_gain,
            "total_fp": total_fp,
            "score": safe_float(score),

            "original_image_path": get_image_path(
                ORIGINAL_IMAGE_DIR,
                stem
            ),

            "degraded_image_path": get_image_path(
                DEGRADED_IMAGE_DIR,
                stem
            ),

            "gt_txt": str(gt_txt),

            "yolo_original_txt": str(yolo_original_txt),
            "yolo_degraded_txt": str(yolo_degraded_txt),

            "ecafa_original_txt": str(ecafa_original_txt),
            "ecafa_degraded_txt": str(ecafa_degraded_txt),
        }
    )


df = pd.DataFrame(rows)

OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

if len(df) == 0:
    print()
    print("[WARNING] No cases found.")
    print("You can try:")
    print("1. Lower MIN_DEGRADED_RECALL_GAIN, e.g. 0.10")
    print("2. Lower CONF_THRESH, e.g. 0.20")
    print("3. Set ALLOW_YOLO_DEGRADED_ZERO_TP = True")
    print("4. Increase MAX_PRED_RATIO, e.g. 6.0")
else:
    df = df.sort_values(
        by=[
            "score",
            "degraded_recall_gain",
            "yolo_recall_drop",
            f"ecafa_{CONDITION}_recall",
            "gt_boxes",
        ],
        ascending=False
    )

    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print(f"Found cases: {len(df)}")
    print(f"Saved to   : {OUT_CSV}")
    print("=" * 80)
    print()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print(df.head(50))