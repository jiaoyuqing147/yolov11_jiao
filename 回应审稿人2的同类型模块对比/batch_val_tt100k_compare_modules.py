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
import yaml

from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics.data.utils import IMG_FORMATS, img2label_paths
except Exception:
    IMG_FORMATS = {
        "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"
    }

    def img2label_paths(img_paths):
        sa = f"{os.sep}images{os.sep}"
        sb = f"{os.sep}labels{os.sep}"
        label_paths = []
        for x in img_paths:
            x = str(x)
            if sa in x:
                label_paths.append(sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")
            else:
                label_paths.append(str(Path(x).with_suffix(".txt")))
        return label_paths


# ============================================================
# 配置区
# ============================================================

# 模型训练结果目录
RUNS_DIR = Path("../runsCompare_modules")

# 只跑 TT100K，不跑 GTSDB
FOLDER_PREFIX = "tt100k_chu_"

# 数据集配置文件
DATA_YAML = r"ultralytics/cfg/datasets/tt100k_desk_130.yaml"

# 跑测试集
SPLIT = "test"

# 验证参数
IMGSZ = 640
BATCH = 32
DEVICE = 0
WORKERS = 0

# 验证结果保存目录
VAL_PROJECT = Path("../valsCompare_modules/tt100k_test")

# 汇总结果保存目录
OUT_DIR = Path("../valsCompare_modules")
OUT_EXCEL = OUT_DIR / "tt100k_test_summary.xlsx"
OUT_CSV = OUT_DIR / "tt100k_test_summary.csv"
OUT_TXT = OUT_DIR / "tt100k_test_summary.txt"

# True 表示如果表格里已经有 status=ok 的模型，则下次运行自动跳过
RESUME = True

# ============================================================


def clean_model_name(folder_name: str):
    """
    把文件夹名清理成表格里的模型名。
    例如：
    tt100k_chu_yolo11-AFPN_train200 -> yolo11-AFPN
    """
    name = folder_name
    name = name.replace("tt100k_chu_", "")
    name = name.replace("_train200", "")
    return name


def find_weight(model_dir: Path):
    """
    自动查找权重文件。
    优先找 best.pt，找不到再找 last.pt。
    兼容以下结构：
    runsCompare_modules/xxx/weights/best.pt
    runsCompare_modules/xxx/exp/weights/best.pt
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


def resolve_path(p, yaml_dir: Path, base_path):
    """
    解析 data yaml 里的 path/train/val/test 路径。
    """
    p = Path(str(p))

    if p.is_absolute():
        return p

    if base_path:
        base = Path(str(base_path))
        if not base.is_absolute():
            base = yaml_dir / base
        return (base / p).resolve()

    return (yaml_dir / p).resolve()


def resolve_image_line(line: str, txt_path: Path, yaml_dir: Path, base_path):
    """
    解析 test.txt 里面的图片路径。
    """
    line = line.strip()
    p = Path(line)

    if p.is_absolute():
        return p

    candidates = []

    if base_path:
        try:
            candidates.append(resolve_path(line, yaml_dir, base_path))
        except Exception:
            pass

    candidates.append((txt_path.parent / line).resolve())
    candidates.append((yaml_dir / line).resolve())

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def collect_images_from_entry(entry, yaml_dir: Path, base_path):
    """
    根据 data yaml 的 test 字段收集图片列表。
    支持：
    test: images/test
    test: test.txt
    test: [path1, path2]
    """
    image_paths = []

    if isinstance(entry, (list, tuple)):
        for e in entry:
            image_paths.extend(collect_images_from_entry(e, yaml_dir, base_path))
        return image_paths

    path = resolve_path(entry, yaml_dir, base_path)

    if path.is_file():
        if path.suffix.lower() == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            image_paths = [
                resolve_image_line(x, path, yaml_dir, base_path)
                for x in lines
            ]
        else:
            if path.suffix.lower().replace(".", "") in IMG_FORMATS:
                image_paths = [path]

    elif path.is_dir():
        image_paths = [
            p for p in path.rglob("*")
            if p.suffix.lower().replace(".", "") in IMG_FORMATS
        ]

    return sorted(list(set([Path(x).resolve() for x in image_paths])))


def count_dataset_images_instances(data_yaml, split="test"):
    """
    统计 Images 和 Instances。
    这个值用于填表，和 YOLO 控制台输出里的 Images / Instances 对应。
    """
    data_yaml = Path(data_yaml).resolve()

    if not data_yaml.exists():
        print(f"[WARN] 没找到 data yaml：{data_yaml}")
        return None, None

    with open(data_yaml, "r", encoding="utf-8", errors="ignore") as f:
        data = yaml.safe_load(f)

    yaml_dir = data_yaml.parent
    base_path = data.get("path", None)
    split_entry = data.get(split, None)

    if split_entry is None:
        print(f"[WARN] data yaml 里没有 split={split}")
        return None, None

    images = collect_images_from_entry(split_entry, yaml_dir, base_path)

    if not images:
        print(f"[WARN] 没有统计到 {split} 图片数量")
        return None, None

    label_paths = img2label_paths([str(x) for x in images])

    instances = 0
    for lp in label_paths:
        lp = Path(lp)
        if lp.exists():
            with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                instances += sum(1 for line in f if line.strip())

    return len(images), instances


def get_model_profile(model):
    """
    获取模型 layers / params / gradients / GFLOPs。
    优先使用 Ultralytics 自带 info。
    """
    profile = {
        "layers": None,
        "params": None,
        "gradients": None,
        "GFLOPs": None,
    }

    try:
        info = model.info(verbose=False)

        if isinstance(info, (list, tuple)):
            if len(info) >= 4:
                profile["layers"] = info[0]
                profile["params"] = info[1]
                profile["gradients"] = info[2]
                profile["GFLOPs"] = info[3]

        elif isinstance(info, dict):
            profile["layers"] = info.get("layers", profile["layers"])
            profile["params"] = info.get("parameters", profile["params"])
            profile["gradients"] = info.get("gradients", profile["gradients"])
            profile["GFLOPs"] = info.get("GFLOPs", profile["GFLOPs"])

    except Exception:
        pass

    try:
        torch_model = model.model
        if profile["params"] is None:
            profile["params"] = sum(p.numel() for p in torch_model.parameters())
        if profile["gradients"] is None:
            profile["gradients"] = sum(
                p.numel() for p in torch_model.parameters() if p.requires_grad
            )
        if profile["layers"] is None:
            profile["layers"] = len(list(torch_model.modules()))
    except Exception:
        pass

    profile["GFLOPs"] = safe_float(profile["GFLOPs"])

    return profile


def get_metrics_value(metrics, key, default=None):
    """
    从 metrics.results_dict 里取值。
    """
    try:
        results_dict = getattr(metrics, "results_dict", {})
        return results_dict.get(key, default)
    except Exception:
        return default


def extract_val_results(metrics):
    """
    提取验证指标：
    Box(P), R, mAP50, mAP50-95
    """
    result = {
        "Box_P": None,
        "Box_R": None,
        "mAP50": None,
        "mAP50-95": None,
        "fitness": None,
    }

    try:
        box = metrics.box

        result["Box_P"] = safe_float(getattr(box, "mp", None))
        result["Box_R"] = safe_float(getattr(box, "mr", None))
        result["mAP50"] = safe_float(getattr(box, "map50", None))
        result["mAP50-95"] = safe_float(getattr(box, "map", None))

    except Exception:
        pass

    if result["Box_P"] is None:
        result["Box_P"] = safe_float(get_metrics_value(metrics, "metrics/precision(B)"))

    if result["Box_R"] is None:
        result["Box_R"] = safe_float(get_metrics_value(metrics, "metrics/recall(B)"))

    if result["mAP50"] is None:
        result["mAP50"] = safe_float(get_metrics_value(metrics, "metrics/mAP50(B)"))

    if result["mAP50-95"] is None:
        result["mAP50-95"] = safe_float(get_metrics_value(metrics, "metrics/mAP50-95(B)"))

    result["fitness"] = safe_float(getattr(metrics, "fitness", None))

    return result


def extract_speed(metrics):
    """
    提取速度指标。
    单位一般是 ms/image。
    """
    speed_result = {}

    try:
        speed = getattr(metrics, "speed", None)

        if isinstance(speed, dict):
            for k, v in speed.items():
                speed_result[f"speed_{k}_ms"] = safe_float(v)

    except Exception:
        pass

    return speed_result


def reorder_columns(df: pd.DataFrame):
    """
    调整表格列顺序。
    """
    preferred = [
        "model",
        "folder",
        "weight",
        "Class",
        "Images",
        "Instances",
        "Box_P",
        "Box_R",
        "mAP50",
        "mAP50-95",
        "fitness",
        "layers",
        "params",
        "gradients",
        "GFLOPs",
        "speed_preprocess_ms",
        "speed_inference_ms",
        "speed_loss_ms",
        "speed_postprocess_ms",
        "split",
        "imgsz",
        "batch",
        "device",
        "save_dir",
        "status",
        "error",
    ]

    cols = list(df.columns)
    final_cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]

    return df[final_cols]


def save_outputs(rows):
    """
    同时保存 CSV、Excel、TXT。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df = reorder_columns(df)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    try:
        df.to_excel(OUT_EXCEL, index=False)
    except Exception as e:
        print(f"[WARN] Excel 保存失败，CSV 和 TXT 已保存。")
        print(f"[WARN] 如果需要 Excel，请执行：pip install openpyxl")
        print(f"[WARN] 错误信息：{e}")

    try:
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
    except Exception as e:
        print(f"[WARN] TXT 保存失败：{e}")


def load_old_rows():
    """
    读取已经存在的结果，方便断点续跑。
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
    找到已经成功完成的模型。
    """
    if not RESUME:
        return set()

    done = set()
    for r in rows:
        if str(r.get("status", "")) == "ok":
            done.add(str(r.get("model", "")))
    return done


def upsert_row(rows, row):
    """
    如果同一个模型已经有旧结果，则替换掉旧结果。
    """
    model_name = str(row.get("model", ""))
    rows = [r for r in rows if str(r.get("model", "")) != model_name]
    rows.append(row)
    return rows


def clear_memory():
    """
    清理显存和内存。
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
    print(f"共发现 {len(model_dirs)} 个 TT100K 模型文件夹")
    print("=" * 100)

    if not model_dirs:
        raise RuntimeError(
            f"没有找到以 {FOLDER_PREFIX} 开头的模型文件夹，请检查 RUNS_DIR 和 FOLDER_PREFIX。"
        )

    print("\n即将验证以下模型：")
    for i, p in enumerate(model_dirs, 1):
        print(f"{i:02d}. {p.name}")

    print("\n开始统计测试集 Images / Instances ...")
    images_count, instances_count = count_dataset_images_instances(DATA_YAML, SPLIT)
    print(f"测试集统计：Images={images_count}, Instances={instances_count}")
    print("=" * 100)

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
                "model": model_name,
                "folder": folder_name,
                "weight": "",
                "Class": "all",
                "Images": images_count,
                "Instances": instances_count,
                "split": SPLIT,
                "imgsz": IMGSZ,
                "batch": BATCH,
                "device": DEVICE,
                "status": "no_weight",
                "error": "未找到 best.pt 或 last.pt",
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
            "model": model_name,
            "folder": folder_name,
            "weight": str(weight),
            "Class": "all",
            "Images": images_count,
            "Instances": instances_count,
            "split": SPLIT,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "status": "running",
            "error": "",
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

            speed_result = extract_speed(metrics)
            row.update(speed_result)

            row["save_dir"] = str(getattr(metrics, "save_dir", ""))
            row["status"] = "ok"
            row["error"] = ""

            print("\n验证完成：")
            print(f"Model       : {model_name}")
            print(f"Images      : {row.get('Images')}")
            print(f"Instances   : {row.get('Instances')}")
            print(f"Box_P       : {row.get('Box_P')}")
            print(f"Box_R       : {row.get('Box_R')}")
            print(f"mAP50       : {row.get('mAP50')}")
            print(f"mAP50-95    : {row.get('mAP50-95')}")
            print(f"Params      : {row.get('params')}")
            print(f"GFLOPs      : {row.get('GFLOPs')}")
            print(f"Save dir    : {row.get('save_dir')}")

        except Exception as e:
            row["status"] = "failed"
            row["error"] = traceback.format_exc()

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
    print("全部 TT100K 模型验证完成。")
    print(f"最终 CSV  ：{OUT_CSV.resolve()}")
    print(f"最终 Excel：{OUT_EXCEL.resolve()}")
    print(f"最终 TXT  ：{OUT_TXT.resolve()}")
    print("=" * 100)