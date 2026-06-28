import warnings
warnings.filterwarnings("ignore")

import os
import gc
import time
from pathlib import Path

import torch
import pandas as pd
from ultralytics import YOLO


# =====================================================
# 项目根目录
# =====================================================

# 当前脚本位置：
# D:\UKMJIAO\AlgorithmCodes\yolov11_jiao\回应审稿人2的鲁棒性测试\1.弄出8个场景的txt文件val_roubust_8sences.py
#
# parents[1] 回到：
# D:\UKMJIAO\AlgorithmCodes\yolov11_jiao

ROOT = Path(__file__).resolve().parents[1]


# =====================================================
# 环境变量
# =====================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# =====================================================
# 路径工具函数
# =====================================================

def project_path(relative_path):
    """
    返回项目内的相对路径字符串。
    用于写入 CSV，避免出现 D:\\... 这种绝对路径。
    """
    return str(Path(relative_path)).replace("\\", "/")


def absolute_path(relative_path):
    """
    根据项目根目录，得到真实绝对路径。
    用于检查文件是否存在。
    """
    return ROOT / relative_path


# =====================================================
# 模型配置
# =====================================================
# 这里保存为项目内相对路径。
# YOLO 加载时会在主程序中切换工作目录到 ROOT，因此可以直接使用相对路径。

MODELS = {
    "YOLOv11":
        project_path(
            "runsTT100k130/yolo11_train200/exp/weights/best.pt"
        ),

    "ECAFA-YOLO":
        project_path(
            "runsTT100k130/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation/exp/weights/best.pt"
        ),
}


# =====================================================
# 鲁棒性测试集
# =====================================================
# 想跑哪个场景，就取消哪个注释。
# 当前只跑重新生成后的 Occlusion 遮挡数据集。
#
# 新遮挡数据集路径：
# F:\DataSets\tt100k\yolojack\images\val_occlusion
#
# 注意：
# ultralytics/cfg/datasets/tt100k_desk_130_occlusion.yaml
# 里面的 test 必须指向：
# F:\DataSets\tt100k\yolojack\images\val_occlusion

DATASETS = {
    # "Original":
    #     project_path(
    #         "ultralytics/cfg/datasets/tt100k_desk_130_original.yaml"
    #     ),
    #
    # "Rain":
    #     project_path(
    #         "ultralytics/cfg/datasets/tt100k_desk_130_rain.yaml"
    #     ),
    #
    # "Fog":
    #     project_path(
    #         "ultralytics/cfg/datasets/tt100k_desk_130_fog.yaml"
    #     ),
    #
    # "LowLight":
    #     project_path(
    #         "ultralytics/cfg/datasets/tt100k_desk_130_lowlight.yaml"
    #     ),
    #
    # "MotionBlur":
    #     project_path(
    #         "ultralytics/cfg/datasets/tt100k_desk_130_motionblur.yaml"
    #     ),
    #
    # "JPEG":
    #     project_path(
    #         "ultralytics/cfg/datasets/tt100k_desk_130_jpeg.yaml"
    #     ),
    #
    "Scale":
        project_path(
            "ultralytics/cfg/datasets/tt100k_desk_130_scale.yaml"
        ),

    # "Occlusion":
    #     project_path(
    #         "ultralytics/cfg/datasets/tt100k_desk_130_occlusion.yaml"
    #     ),
}


# =====================================================
# 验证参数
# =====================================================

IMG_SIZE = 640
BATCH = 12

PROJECT = project_path(
    "valsRobust/robust_tt100k130"
)

# =====================================================
# CSV 保存路径
# =====================================================
# 如果 DATASETS 里只启用了一个场景，比如 Fog，
# 就自动保存为 robust_results_fog.csv。
#
# 如果 DATASETS 里启用了多个场景，
# 就保存为 robust_results_selected.csv。

active_conditions = list(DATASETS.keys())

if len(active_conditions) == 1:
    csv_condition_name = active_conditions[0].lower()
    CSV_SAVE_PATH = project_path(
        f"valsRobust/robust_tt100k130/robust_results_{csv_condition_name}.csv"
    )
else:
    CSV_SAVE_PATH = project_path(
        "valsRobust/robust_tt100k130/robust_results.csv"
    )


# =====================================================
# 工具函数
# =====================================================

def safe_name(text):
    return (
        text.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def to_project_relative(path):
    """
    把 results.save_dir 这类路径转换成项目内相对路径。
    如果已经是相对路径，就直接规范化。
    """
    p = Path(path)

    try:
        rel = p.resolve().relative_to(ROOT.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


# =====================================================
# 单次验证
# =====================================================

def run_val(
        model_name,
        weight_path,
        condition,
        data_yaml):

    print()
    print("=" * 80)
    print(f"Model     : {model_name}")
    print(f"Condition : {condition}")
    print(f"Weight    : {weight_path}")
    print(f"Dataset   : {data_yaml}")
    print("=" * 80)

    model = None
    results = None

    try:

        model = YOLO(weight_path)

        run_name = (
            f"{safe_name(model_name)}_"
            f"{safe_name(condition)}"
        )

        results = model.val(
            data=data_yaml,
            split="test",
            imgsz=IMG_SIZE,
            batch=BATCH,

            save_json=False,

            save_txt=True,
            save_conf=True,

            plots=False,

            project=PROJECT,
            name=run_name,

            # 不删除旧目录，不新建 rerun 后缀目录。
            # 如果目录已存在，就继续写入同名目录。
            exist_ok=True,

            verbose=True,
        )

        row = {
            "Model": model_name,
            "Condition": condition,

            "Precision":
                float(results.box.mp),

            "Recall":
                float(results.box.mr),

            "mAP50":
                float(results.box.map50),

            "mAP50-95":
                float(results.box.map),

            # 这里强制转成项目内相对路径
            "SaveDir":
                to_project_relative(results.save_dir),

            # 这里本身就是项目内相对路径
            "Weight":
                project_path(weight_path),

            # 这里本身就是项目内相对路径
            "Dataset":
                project_path(data_yaml),
        }

        return row

    finally:

        try:
            del results
        except:
            pass

        try:
            del model
        except:
            pass

        gc.collect()

        if torch.cuda.is_available():

            try:
                torch.cuda.synchronize()
            except:
                pass

            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except:
                pass

        gc.collect()

        time.sleep(3)

        print("[INFO] CPU and CUDA memory released.")


# =====================================================
# 主程序
# =====================================================

if __name__ == "__main__":

    # 切换到项目根目录。
    # 这样 YOLO 加载权重、读取 yaml、保存 project 时都使用项目相对路径。
    os.chdir(ROOT)

    print("=" * 80)
    print("Path Check")
    print("=" * 80)
    print(f"Current working dir : {os.getcwd()}")
    print(f"Project ROOT        : {ROOT}")
    print(f"PROJECT output dir  : {PROJECT}")
    print(f"CSV save path       : {CSV_SAVE_PATH}")
    print("=" * 80)

    all_results = []

    for model_name, weight_path in MODELS.items():

        print()
        print("#" * 80)
        print(f"Running Model: {model_name}")
        print("#" * 80)

        if not absolute_path(weight_path).exists():

            print(
                f"[WARNING] Weight not found:\n"
                f"{weight_path}"
            )

            continue

        for condition, data_yaml in DATASETS.items():

            if not absolute_path(data_yaml).exists():

                print(
                    f"[WARNING] YAML not found:\n"
                    f"{data_yaml}"
                )

                continue

            try:

                row = run_val(
                    model_name=model_name,
                    weight_path=weight_path,
                    condition=condition,
                    data_yaml=data_yaml,
                )

                all_results.append(row)

                os.makedirs(
                    os.path.dirname(CSV_SAVE_PATH),
                    exist_ok=True
                )

                pd.DataFrame(
                    all_results
                ).to_csv(
                    CSV_SAVE_PATH,
                    index=False,
                    encoding="utf-8-sig"
                )

                print("[OK] Current results saved:")
                print(CSV_SAVE_PATH)

            except Exception as e:

                print()
                print(
                    f"[ERROR] "
                    f"{model_name} - {condition}"
                )

                print(e)

    df = pd.DataFrame(all_results)

    if len(df) > 0:

        df.to_csv(
            CSV_SAVE_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print("=" * 80)
        print("Final Results")
        print("=" * 80)

        print(df)

        print()
        print(
            f"CSV saved to:\n"
            f"{CSV_SAVE_PATH}"
        )

    print()
    print("=" * 80)
    print("Robustness validation finished.")
    print("=" * 80)