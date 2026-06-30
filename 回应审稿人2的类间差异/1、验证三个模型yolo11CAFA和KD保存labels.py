import os

# =====================================================
# 必须放在 numpy / pandas / torch / ultralytics 之前
# =====================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 防止 Windows 中文/emoji 输出乱码
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"


import warnings
warnings.filterwarnings("ignore")

# 防止 plots=True 时 matplotlib 后端出问题
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from ultralytics import YOLO


# =====================================================
# 0. 项目根目录
# =====================================================
# 当前脚本位置：
# yolov11_jiao/回应审稿人2的类间差异/1、验证三个模型yolo11CAFA和KD保存labels.py
#
# 所以 parent 是：回应审稿人2的类间差异
# parent.parent 是：yolov11_jiao 项目根目录
ROOT = Path(__file__).resolve().parent.parent

print(f"\nProject ROOT: {ROOT}\n")


# =====================================================
# 1. 检查 seaborn
# =====================================================
try:
    import seaborn  # noqa
except ImportError:
    raise ImportError(
        "\n缺少 seaborn，YOLO 无法生成 confusion_matrix.png。\n"
        "请先运行：\n"
        r"D:\UKMJIAO\my_envs\yolov11_env\python.exe -m pip install seaborn"
    )


# =====================================================
# 2. 数据集配置
# =====================================================
DATA_YAML = ROOT / "ultralytics" / "cfg" / "datasets" / "tt100k_desk_130.yaml"

# 注意：
# 如果你论文主实验用 test，就写 test；
# 如果你想和训练过程默认验证保持一致，就写 val。
SPLIT = "test"

IMGSZ = 640
BATCH = 12


# =====================================================
# 3. 模型配置
# =====================================================
MODELS = [
    {
        "name": "YOLOv11_baseline",
        "weight": ROOT / "runsTT100k130" / "yolo11_train200" / "exp" / "weights" / "best.pt",
    },
    {
        "name": "ECAFA_YOLO",
        "weight": ROOT / "runsTT100k130" / "yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200" / "exp" / "weights" / "best.pt",
    },
    {
        "name": "ECAFA_YOLO_KD",
        "weight": ROOT / "runsTT100k130" / "yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation" / "exp" / "weights" / "best.pt",
    },
]


# =====================================================
# 4. 输出目录
# =====================================================
PROJECT = ROOT / "vals_error_analysis" / "tt100k_confusion_full"


# =====================================================
# 5. 检查路径是否存在
# =====================================================
def check_file_exists(file_path, file_type="file"):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"\n找不到{file_type}：\n{file_path}\n\n"
            f"请检查路径是否正确。"
        )


# =====================================================
# 6. 主函数：逐个模型生成完整混淆矩阵
# =====================================================
if __name__ == "__main__":

    print("\n=====================================================")
    print("Start generating full confusion matrices")
    print("=====================================================")
    print(f"ROOT: {ROOT}")
    print(f"DATA_YAML: {DATA_YAML}")
    print(f"SPLIT: {SPLIT}")
    print(f"IMGSZ: {IMGSZ}")
    print(f"BATCH: {BATCH}")
    print(f"OUTPUT PROJECT: {PROJECT}")
    print("=====================================================\n")

    # 检查数据集 yaml
    check_file_exists(DATA_YAML, "数据集 YAML 文件")

    for item in MODELS:
        model_name = item["name"]
        weight = item["weight"]

        print("\n-----------------------------------------------------")
        print(f"Validating model: {model_name}")
        print(f"Weight: {weight}")
        print("-----------------------------------------------------\n")

        check_file_exists(weight, "权重文件")

        model = YOLO(str(weight))

        results = model.val(
            data=str(DATA_YAML),
            split=SPLIT,
            imgsz=IMGSZ,
            batch=BATCH,

            # =====================================================
            # 关键：生成完整混淆矩阵
            # =====================================================
            plots=True,

            # 下面这些不是混淆矩阵必须的，但保留有利于后续分析
            save_json=True,
            save_txt=True,
            save_conf=True,

            # Windows 下建议加
            workers=0,

            project=str(PROJECT),
            name=f"{model_name}/exp",
            exist_ok=True,
        )

        save_dir = PROJECT / model_name / "exp"

        print("\nSaved results to:")
        print(save_dir)

        print("\nExpected confusion matrix files:")
        print(save_dir / "confusion_matrix.png")
        print(save_dir / "confusion_matrix_normalized.png")

        if (save_dir / "confusion_matrix.png").exists():
            print("confusion_matrix.png generated.")
        else:
            print("confusion_matrix.png not found. Please check whether plots=True worked.")

        if (save_dir / "confusion_matrix_normalized.png").exists():
            print("confusion_matrix_normalized.png generated.")
        else:
            print("confusion_matrix_normalized.png not found. Please check whether seaborn is installed.")

    print("\n=====================================================")
    print("All full confusion matrices have been generated.")
    print("=====================================================")