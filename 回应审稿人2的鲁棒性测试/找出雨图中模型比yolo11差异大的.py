from pathlib import Path
import pandas as pd


# =====================================================
# 自动定位项目根目录
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 当前脚本位于：
# 项目根目录 / 回应审稿人2的鲁棒性测试 / 找出雨图中模型比yolo11差异大的.py
# 所以项目根目录 = 当前脚本目录的上一级
PROJECT_ROOT = SCRIPT_DIR.parent


# =====================================================
# 配置
# =====================================================

CONDITION = "Rain"   # MotionBlur / Rain / Fog / LowLight / JPEG / Scale / Occlusion

RESULT_ROOT = PROJECT_ROOT / "valsRobust" / "robust_tt100k130"

YOLO_LABEL_DIR = (
    RESULT_ROOT /
    f"YOLOv11_{CONDITION}" /
    "labels"
)

ECAFA_LABEL_DIR = (
    RESULT_ROOT /
    f"ECAFA_YOLO_{CONDITION}" /
    "labels"
)

# 数据集根目录仍然用绝对路径，不用改
DATA_ROOT = Path(
    r"F:\DataSets\tt100k\yolojack"
)

# 退化图像目录
IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    f"val_{CONDITION.lower()}"
)

# 原始图像目录
ORIGINAL_IMAGE_DIR = (
    DATA_ROOT /
    "images" /
    "val"
)

# GT标签目录
GT_LABEL_DIR = (
    DATA_ROOT /
    "labels" /
    "val"
)

OUT_CSV = (
    RESULT_ROOT /
    f"select_{CONDITION}_cases.csv"
)


# =====================================================
# 筛选参数
# =====================================================

MIN_GT_BOXES = 4          # 至少4个真实目标
MIN_YOLO_BOXES = 3        # YOLO至少有一些检测框，避免完全空
MIN_ECAFA_BOXES = 6       # ECAFA检测框数量至少6个
MIN_BOX_GAIN = 3          # ECAFA比YOLO至少多3个框


# =====================================================
# 工具函数
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


def mean_conf(preds):
    if len(preds) == 0:
        return 0.0

    return sum(p["conf"] for p in preds) / len(preds)


def get_image_path(image_dir, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return ""


# =====================================================
# 收集所有txt
# =====================================================

all_stems = set()

for p in YOLO_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)

for p in ECAFA_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)

# 也把GT加入，避免预测为空的图漏掉
for p in GT_LABEL_DIR.glob("*.txt"):
    all_stems.add(p.stem)


rows = []

for stem in sorted(all_stems):

    yolo_txt = YOLO_LABEL_DIR / f"{stem}.txt"
    ecafa_txt = ECAFA_LABEL_DIR / f"{stem}.txt"
    gt_txt = GT_LABEL_DIR / f"{stem}.txt"

    yolo_preds = read_pred_txt(yolo_txt)
    ecafa_preds = read_pred_txt(ecafa_txt)
    gt_boxes = read_gt_txt(gt_txt)

    gt_n = len(gt_boxes)
    yolo_n = len(yolo_preds)
    ecafa_n = len(ecafa_preds)

    yolo_conf = mean_conf(yolo_preds)
    ecafa_conf = mean_conf(ecafa_preds)

    box_gain = ecafa_n - yolo_n
    conf_gain = ecafa_conf - yolo_conf

    # =================================================
    # 筛选：GT数量要够，预测差异要明显
    # =================================================

    if gt_n < MIN_GT_BOXES:
        continue

    if yolo_n < MIN_YOLO_BOXES:
        continue

    if ecafa_n < MIN_ECAFA_BOXES:
        continue

    if box_gain < MIN_BOX_GAIN:
        continue

    # 综合分数：
    # 1. GT多的图更适合展示
    # 2. ECAFA比YOLO多检出的框越多越好
    # 3. 置信度差异作为辅助
    score = (
        gt_n * 5
        +
        box_gain * 10
        +
        conf_gain
    )

    degraded_image_path = get_image_path(
        IMAGE_DIR,
        stem
    )

    original_image_path = get_image_path(
        ORIGINAL_IMAGE_DIR,
        stem
    )

    rows.append(
        {
            "image": stem,
            "gt_boxes": gt_n,

            "original_image_path": original_image_path,
            "degraded_image_path": degraded_image_path,

            "yolo_boxes": yolo_n,
            "ecafa_boxes": ecafa_n,
            "box_gain": box_gain,

            "yolo_mean_conf": yolo_conf,
            "ecafa_mean_conf": ecafa_conf,
            "conf_gain": conf_gain,

            "score": score,

            "gt_txt": str(gt_txt),
            "yolo_txt": str(yolo_txt),
            "ecafa_txt": str(ecafa_txt),
        }
    )


df = pd.DataFrame(rows)

if len(df) == 0:
    print("[WARNING] No cases found. Try lowering thresholds.")
else:
    df = df.sort_values(
        by=[
            "score",
            "gt_boxes",
            "box_gain",
            "conf_gain"
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

    print(df.head(30))
    print()
    print(f"Saved to: {OUT_CSV}")