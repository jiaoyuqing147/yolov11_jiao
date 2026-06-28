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

CONDITION = "Occlusion"   # MotionBlur / Rain / Fog / LowLight / JPEG / Scale / Occlusion

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
    f"select_{CONDITION}_cases_highconf.csv"
)


# =====================================================
# 筛选参数
# =====================================================

# 和你最终画图脚本里的 CONF_THRESH 保持一致
DISPLAY_CONF_THRESH = 0.25

# 用于筛“更漂亮案例”的高置信度阈值
HIGH_CONF_THRESH = 0.40

MIN_GT_BOXES = 4              # 至少4个真实目标

# 下面这些都基于 conf >= DISPLAY_CONF_THRESH 的框
MIN_YOLO_SHOW_BOXES = 3       # YOLO至少有一些可显示框，避免完全空
MIN_ECAFA_SHOW_BOXES = 6      # ECAFA可显示框至少6个
MIN_SHOW_BOX_GAIN = 3         # ECAFA比YOLO至少多3个可显示框

# 下面这些基于 conf >= HIGH_CONF_THRESH 的框
MIN_ECAFA_HIGH_BOXES = 3      # ECAFA至少有3个高置信度框
MIN_HIGH_BOX_GAIN = 1         # ECAFA高置信度框至少比YOLO多1个


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


def filter_by_conf(preds, conf_thresh):
    """
    按置信度筛选预测框。
    """
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

    yolo_preds_all = read_pred_txt(yolo_txt)
    ecafa_preds_all = read_pred_txt(ecafa_txt)
    gt_boxes = read_gt_txt(gt_txt)

    gt_n = len(gt_boxes)

    # =================================================
    # 1. 原始全部预测框数量
    # =================================================

    yolo_all_n = len(yolo_preds_all)
    ecafa_all_n = len(ecafa_preds_all)
    all_box_gain = ecafa_all_n - yolo_all_n

    yolo_all_mean_conf = mean_conf(yolo_preds_all)
    ecafa_all_mean_conf = mean_conf(ecafa_preds_all)
    all_conf_gain = ecafa_all_mean_conf - yolo_all_mean_conf

    # =================================================
    # 2. 最终画图会显示的框：conf >= 0.25
    # =================================================

    yolo_show_preds = filter_by_conf(
        yolo_preds_all,
        DISPLAY_CONF_THRESH
    )

    ecafa_show_preds = filter_by_conf(
        ecafa_preds_all,
        DISPLAY_CONF_THRESH
    )

    yolo_show_n = len(yolo_show_preds)
    ecafa_show_n = len(ecafa_show_preds)
    show_box_gain = ecafa_show_n - yolo_show_n

    yolo_show_mean_conf = mean_conf(yolo_show_preds)
    ecafa_show_mean_conf = mean_conf(ecafa_show_preds)
    show_conf_gain = ecafa_show_mean_conf - yolo_show_mean_conf

    # =================================================
    # 3. 高置信度框：conf >= 0.40
    # =================================================

    yolo_high_preds = filter_by_conf(
        yolo_preds_all,
        HIGH_CONF_THRESH
    )

    ecafa_high_preds = filter_by_conf(
        ecafa_preds_all,
        HIGH_CONF_THRESH
    )

    yolo_high_n = len(yolo_high_preds)
    ecafa_high_n = len(ecafa_high_preds)
    high_box_gain = ecafa_high_n - yolo_high_n

    yolo_high_mean_conf = mean_conf(yolo_high_preds)
    ecafa_high_mean_conf = mean_conf(ecafa_high_preds)
    high_conf_gain = ecafa_high_mean_conf - yolo_high_mean_conf

    yolo_max_conf = max_conf(yolo_preds_all)
    ecafa_max_conf = max_conf(ecafa_preds_all)

    # =================================================
    # 筛选：GT数量要够，且高置信度框差异要明显
    # =================================================

    if gt_n < MIN_GT_BOXES:
        continue

    if yolo_show_n < MIN_YOLO_SHOW_BOXES:
        continue

    if ecafa_show_n < MIN_ECAFA_SHOW_BOXES:
        continue

    if show_box_gain < MIN_SHOW_BOX_GAIN:
        continue

    if ecafa_high_n < MIN_ECAFA_HIGH_BOXES:
        continue

    if high_box_gain < MIN_HIGH_BOX_GAIN:
        continue

    # =================================================
    # 综合分数
    # 重点改为：
    # 1. 高置信度框增益最重要
    # 2. 可显示框增益其次
    # 3. GT数量作为辅助
    # 4. 高置信度平均置信度差异作为辅助
    # =================================================

    score = (
        gt_n * 5
        +
        high_box_gain * 50
        +
        ecafa_high_n * 10
        +
        show_box_gain * 15
        +
        high_conf_gain * 20
        +
        show_conf_gain * 10
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

            # 全部预测框
            "yolo_all_boxes": yolo_all_n,
            "ecafa_all_boxes": ecafa_all_n,
            "all_box_gain": all_box_gain,
            "yolo_all_mean_conf": yolo_all_mean_conf,
            "ecafa_all_mean_conf": ecafa_all_mean_conf,
            "all_conf_gain": all_conf_gain,

            # conf >= 0.25，也就是最终图里会显示的框
            "display_conf_thresh": DISPLAY_CONF_THRESH,
            "yolo_show_boxes": yolo_show_n,
            "ecafa_show_boxes": ecafa_show_n,
            "show_box_gain": show_box_gain,
            "yolo_show_mean_conf": yolo_show_mean_conf,
            "ecafa_show_mean_conf": ecafa_show_mean_conf,
            "show_conf_gain": show_conf_gain,

            # conf >= 0.40，高置信度框
            "high_conf_thresh": HIGH_CONF_THRESH,
            "yolo_high_boxes": yolo_high_n,
            "ecafa_high_boxes": ecafa_high_n,
            "high_box_gain": high_box_gain,
            "yolo_high_mean_conf": yolo_high_mean_conf,
            "ecafa_high_mean_conf": ecafa_high_mean_conf,
            "high_conf_gain": high_conf_gain,

            "yolo_max_conf": yolo_max_conf,
            "ecafa_max_conf": ecafa_max_conf,

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
            "high_box_gain",
            "ecafa_high_boxes",
            "show_box_gain",
            "gt_boxes",
            "high_conf_gain"
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