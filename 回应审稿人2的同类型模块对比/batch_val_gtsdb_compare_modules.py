import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import traceback
import gc
import pandas as pd

from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None


# ============================================================
# 配置区
# ============================================================

# 模型训练结果总目录
RUNS_DIR = Path("../runsCompare_modules")

# 只跑 GTSDB 模型，不跑 TT100K
FOLDER_PREFIX = "GTSDB_chu_"

# GTSDB 数据集配置文件
DATA_YAML = r"ultralytics/cfg/datasets/GTSDB_desktop.yaml"

# 跑测试集；如果你的 yaml 里没有 test，就改成 val
SPLIT = "test"

# 验证参数
IMGSZ = 640
BATCH = 32
DEVICE = 0
WORKERS = 0

# 验证结果保存目录
VAL_PROJECT = Path("../valsCompare_modules/gtsdb_test")

# 汇总结果保存目录
OUT_DIR = Path("../valsCompare_modules")
OUT_EXCEL = OUT_DIR / "gtsdb_test_summary.xlsx"
OUT_CSV = OUT_DIR / "gtsdb_test_summary.csv"
OUT_TXT = OUT_DIR / "gtsdb_test_summary.txt"

# True 表示已经跑过且 status=ok 的模型，下次自动跳过
RESUME = True

# ============================================================


def clean_model_name(folder_name: str):
    """
    例如：
    GTSDB_chu_yolo11-AFPN_train200 -> YOLO11-AFPN
    """
    name = folder_name
    name = name.replace("GTSDB_chu_", "")
    name = name.replace("_train200", "")
    name = name.replace("yolo11", "YOLO11")
    return name


def find_weight(model_dir: Path):
    """
    自动查找权重：
    优先 best.pt，找不到再找 last.pt。
    支持：
    runsCompare_modules/xxx/exp/weights/best.pt
    runsCompare_modules/xxx/weights/best.pt
    runsCompare_modules/xxx/**/weights/best.pt
    """
    candidates = list(model_dir.glob("**/weights/best.pt"))

    if not candidates:
        candidates = list(model_dir.glob("**/weights/last.pt"))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def safe_float(x, ndigits=6):
    try:
        if callable(x):
            x = x()
        return round(float(x), ndigits)
    except Exception:
        return None


def get_model_profile(model):
    """
    获取 Params 和 GFLOPs。
    如果当前 Ultralytics 版本拿不到 GFLOPs，则返回 None。
    """
    profile = {
        "Params": None,
        "Params(M)": None,
        "GFLOPs": None,
    }

    # 统计参数量
    try:
        torch_model = model.model
        params = sum(p.numel() for p in torch_model.parameters())
        profile["Params"] = int(params)
        profile["Params(M)"] = round(params / 1e6, 3)
    except Exception:
        pass

    # 尝试获取 GFLOPs
    try:
        info = model.info(verbose=False)

        if isinstance(info, (list, tuple)):
            # 不同 Ultralytics 版本返回格式可能是：
            # layers, params, gradients, flops
            if len(info) >= 4:
                profile["GFLOPs"] = safe_float(info[3], ndigits=3)

        elif isinstance(info, dict):
            flops = (
                info.get("GFLOPs", None)
                or info.get("gflops", None)
                or info.get("flops", None)
            )
            profile["GFLOPs"] = safe_float(flops, ndigits=3)

    except Exception:
        pass

    # 再尝试从 torch_utils 里获取 GFLOPs
    if profile["GFLOPs"] is None:
        try:
            from ultralytics.utils.torch_utils import get_flops
            flops = get_flops(model.model, imgsz=IMGSZ)
            profile["GFLOPs"] = safe_float(flops, ndigits=3)
        except Exception:
            pass

    return profile


def extract_val_results(metrics):
    """
    提取验证指标：
    mAP@50, mAP@50-95, Precision, Recall
    """
    result = {
        "mAP@50": None,
        "mAP@50-95": None,
        "Precision": None,
        "Recall": None,
    }

    try:
        box = metrics.box

        result["Precision"] = safe_float(getattr(box, "mp", None), ndigits=6)
        result["Recall"] = safe_float(getattr(box, "mr", None), ndigits=6)
        result["mAP@50"] = safe_float(getattr(box, "map50", None), ndigits=6)
        result["mAP@50-95"] = safe_float(getattr(box, "map", None), ndigits=6)

    except Exception:
        pass

    # 兼容 results_dict
    try:
        results_dict = getattr(metrics, "results_dict", {})

        if result["Precision"] is None:
            result["Precision"] = safe_float(
                results_dict.get("metrics/precision(B)", None),
                ndigits=6
            )

        if result["Recall"] is None:
            result["Recall"] = safe_float(
                results_dict.get("metrics/recall(B)", None),
                ndigits=6
            )

        if result["mAP@50"] is None:
            result["mAP@50"] = safe_float(
                results_dict.get("metrics/mAP50(B)", None),
                ndigits=6
            )

        if result["mAP@50-95"] is None:
            result["mAP@50-95"] = safe_float(
                results_dict.get("metrics/mAP50-95(B)", None),
                ndigits=6
            )

    except Exception:
        pass

    return result


def save_outputs(rows):
    """
    保存 Excel、CSV、TXT。
    这里只保留你要的核心指标。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    preferred_cols = [
        "Model",
        "mAP@50",
        "mAP@50-95",
        "Precision",
        "Recall",
        "Params(M)",
        "GFLOPs",
        "Params",
        "Weight",
        "Status",
        "Error",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    try:
        df.to_excel(OUT_EXCEL, index=False)
    except Exception as e:
        print("[WARN] Excel 保存失败，但 CSV 和 TXT 会正常保存。")
        print("[WARN] 如果需要 Excel，请安装：pip install openpyxl")
        print(f"[WARN] 错误信息：{e}")

    try:
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
    except Exception as e:
        print(f"[WARN] TXT 保存失败：{e}")


def load_old_rows():
    """
    读取旧结果，用于断点续跑。
    """
    if not OUT_CSV.exists():
        return []

    try:
        df = pd.read_csv(OUT_CSV)
        return df.to_dict("records")
    except Exception:
        return []


def get_done_models(rows):
    """
    找到已经成功跑完的模型。
    """
    if not RESUME:
        return set()

    done = set()
    for r in rows:
        if str(r.get("Status", "")) == "ok":
            done.add(str(r.get("Model", "")))
    return done


def upsert_row(rows, row):
    """
    同一个模型重新跑时，覆盖旧结果。
    """
    model_name = str(row.get("Model", ""))
    rows = [r for r in rows if str(r.get("Model", "")) != model_name]
    rows.append(row)
    return rows


def clear_memory():
    """
    清理显存。
    """
    gc.collect()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VAL_PROJECT.mkdir(parents=True, exist_ok=True)

    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"没有找到目录：{RUNS_DIR.resolve()}")

    model_dirs = [
        p for p in RUNS_DIR.iterdir()
        if p.is_dir() and p.name.startswith(FOLDER_PREFIX)
    ]

    model_dirs = sorted(model_dirs, key=lambda p: p.name)

    print("=" * 100)
    print(f"模型目录：{RUNS_DIR.resolve()}")
    print(f"只扫描前缀：{FOLDER_PREFIX}")
    print(f"数据集配置：{DATA_YAML}")
    print(f"验证 split：{SPLIT}")
    print(f"共发现 {len(model_dirs)} 个 GTSDB 模型文件夹")
    print("=" * 100)

    if not model_dirs:
        raise RuntimeError(
            f"没有找到以 {FOLDER_PREFIX} 开头的模型文件夹，请检查 RUNS_DIR 和 FOLDER_PREFIX。"
        )

    print("\n即将验证以下模型：")
    for i, p in enumerate(model_dirs, 1):
        print(f"{i:02d}. {p.name}")

    rows = load_old_rows()
    done_models = get_done_models(rows)

    for idx, model_dir in enumerate(model_dirs, start=1):
        folder_name = model_dir.name
        model_name = clean_model_name(folder_name)

        if model_name in done_models:
            print(f"\n[{idx}/{len(model_dirs)}] 跳过已完成模型：{model_name}")
            continue

        weight = find_weight(model_dir)

        if weight is None:
            print(f"\n[{idx}/{len(model_dirs)}] 未找到权重文件：{folder_name}")

            row = {
                "Model": model_name,
                "mAP@50": None,
                "mAP@50-95": None,
                "Precision": None,
                "Recall": None,
                "Params(M)": None,
                "GFLOPs": None,
                "Params": None,
                "Weight": "",
                "Status": "no_weight",
                "Error": "未找到 best.pt 或 last.pt",
            }

            rows = upsert_row(rows, row)
            save_outputs(rows)
            continue

        print("\n" + "=" * 100)
        print(f"[{idx}/{len(model_dirs)}] 开始验证：{model_name}")
        print(f"模型文件夹：{folder_name}")
        print(f"权重路径：{weight}")
        print("=" * 100)

        row = {
            "Model": model_name,
            "mAP@50": None,
            "mAP@50-95": None,
            "Precision": None,
            "Recall": None,
            "Params(M)": None,
            "GFLOPs": None,
            "Params": None,
            "Weight": str(weight),
            "Status": "running",
            "Error": "",
        }

        try:
            model = YOLO(str(weight))

            profile = get_model_profile(model)
            row.update(profile)

            metrics = model.val(
                data=DATA_YAML,
                split=SPLIT,
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                workers=WORKERS,
                save_json=False,
                project=str(VAL_PROJECT),
                name=model_name,
                exist_ok=True,
            )

            val_result = extract_val_results(metrics)
            row.update(val_result)

            row["Status"] = "ok"
            row["Error"] = ""

            print("\n验证完成：")
            print(f"Model      : {row.get('Model')}")
            print(f"mAP@50     : {row.get('mAP@50')}")
            print(f"mAP@50-95  : {row.get('mAP@50-95')}")
            print(f"Precision  : {row.get('Precision')}")
            print(f"Recall     : {row.get('Recall')}")
            print(f"Params(M)  : {row.get('Params(M)')}")
            print(f"GFLOPs     : {row.get('GFLOPs')}")

        except Exception as e:
            row["Status"] = "failed"
            row["Error"] = traceback.format_exc()

            print(f"\n[ERROR] 模型验证失败：{model_name}")
            print(e)
            print(traceback.format_exc())

        finally:
            try:
                del model
            except Exception:
                pass

            clear_memory()

        rows = upsert_row(rows, row)
        save_outputs(rows)

        print("\n当前汇总文件已保存：")
        print(f"CSV  : {OUT_CSV.resolve()}")
        print(f"Excel: {OUT_EXCEL.resolve()}")
        print(f"TXT  : {OUT_TXT.resolve()}")

    print("\n" + "=" * 100)
    print("全部 GTSDB 模型验证完成。")
    print(f"最终 CSV  ：{OUT_CSV.resolve()}")
    print(f"最终 Excel：{OUT_EXCEL.resolve()}")
    print(f"最终 TXT  ：{OUT_TXT.resolve()}")
    print("=" * 100)