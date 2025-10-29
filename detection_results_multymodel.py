import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import torch
# 选设备：优先用 GPU，没有就退回 CPU
DEVICE = 0 if torch.cuda.is_available() else 'cpu'   # 也可写 "cuda:0" / "cpu"
USE_HALF = (DEVICE != 'cpu')  # 只有 GPU 才能开 FP16

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# 可选：固定尺寸时有时更快
torch.backends.cudnn.benchmark = True


from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


def make_colors(num_classes: int, overrides: dict | None = None):
    """返回长度为 num_classes 的 RGB 颜色列表（int, 0~255），可用 overrides 覆盖某些类别颜色。"""
    #cmap = plt.get_cmap("tab20")  # 20色，不够会循环，这个颜色浅了点，后面会加深下
    cmap = plt.get_cmap("Dark2")

    base = [tuple(int(x * 255) for x in cmap(i % 20)[:3]) for i in range(num_classes)]
    if overrides:
        for cls_id, rgb in overrides.items():
            if 0 <= cls_id < num_classes:
                base[cls_id] = rgb
    return base


def draw_boxes(img: Image.Image,
               boxes_xyxy: np.ndarray,
               confs: np.ndarray,
               classes: np.ndarray,
               class_names,
               colors: list[tuple[int, int, int]]):
    """在图上画框+标签（类别+置信度）"""
    draw = ImageDraw.Draw(img)
    # 字体：默认字体通用，如需更美观可换成 truetype 字体文件
    font = ImageFont.load_default()

    W, H = img.size
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, classes):
        cls = int(cls)
        color = colors[cls % len(colors)]
        # 加深：每个通道乘以1.8（上限255），上面的设置，颜色太浅了，使用下方代码加粗
        color = tuple(min(255, int(c * 1.2)) for c in color)

        # 1) 画边框,若要调整粗细，改变数字大小即可
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # 2) 画左上角标签背景 + 文本（类别 名称 + 置信度）
        label = f"{class_names[cls]} {conf:.2f}"

        # Pillow 新旧版本兼容：优先 textbbox，失败则用 textsize
        try:
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = r - l, b - t
        except Exception:
            text_w, text_h = draw.textsize(label, font=font)

        # 让标签框不越界
        tx = max(0, min(int(x1), W - text_w - 1))
        ty = max(0, int(y1) - text_h - 2)

        # 背景与边框同色，文字黑色
        draw.rectangle([tx, ty, tx + text_w, ty + text_h], fill=color)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    return img


if __name__ == "__main__":
    # ===== 路径配置 =====
    models_root = Path(r"runsTT100k130")                      # 所有模型所在的根目录
    model_dirs = [
        "YOLO11-C3K2-RCSOSA_train",
        "yolo11-FASFFHead_P234_RCSOSA_ciou_bce_train",
        "yolo11-FASFFHead_P234_RCSOSA_wiou_bce_distillation",
        "yolo11-FASFFHead_P234_RCSOSA_wiou_bce_train",
        "yolo11-FASFFHead_P234_train",
        "yolo11-FASFFHead_P234_wiou_bce_train",
        "yolo11_train",
        "yolo11_WIOU+BCELoss_train",
        #"yolo11x-FASFFHead_P234_RCSOSA_wiou_bce_train(batch16worker16)",
    ]
    img_folder   = Path(r"E:\DataSets\tt100k_2021\yolojack\images\val")
    base_out_dir = Path(r"E:\DataSets\tt100k_2021result")           # 所有结果的统一根目录
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 收集图片 =====
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths += list(img_folder.rglob(ext))
    img_paths.sort()

    for model_name in model_dirs:
        weights_path = models_root / model_name / "exp" / "weights" / "best.pt"
        if not weights_path.exists():
            print(f"[跳过] 未找到: {weights_path}")
            continue

        # 每个模型单独的输出子目录：统一放在 base_out_dir 下
        out_dir = base_out_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(weights_path))
        model.to(DEVICE)
        print(f"\n=== 推理模型: {model_name}  |  device: {model.device} ===")

        custom_colors = [
            (255, 0, 0), (0, 176, 80), (0, 112, 192),
            (255, 192, 0), (255, 0, 255), (255, 128, 0),
            (128, 0, 255), (0, 255, 255),
        ]
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(model.names))]

        # 一个模型一个进度条
        # 一个模型一个进度条
        for img_path in tqdm(img_paths, desc=f"{model_name}", ncols=100):
            results = model.predict(
                img_path,
                imgsz=640,
                conf=0.25,
                iou=0.50,
                device=DEVICE,
                half=USE_HALF,
                verbose=False  # 关闭Ultralytics自带逐图日志
            )

            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)

            img = Image.open(img_path).convert("RGB")
            img = draw_boxes(img, boxes, confs, clses, model.names, colors)

            save_img = out_dir / f"{img_path.stem}_result.jpg"
            img.save(save_img, quality=95)

            W, H = img.size
            save_txt = out_dir / f"{img_path.stem}.txt"
            with open(save_txt, "w", encoding="utf-8") as f:
                for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clses):
                    x_c = (x1 + x2) / 2 / W
                    y_c = (y1 + y2) / 2 / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        # 可选：每个模型结束后给一行简短汇总
        print(f"[Done] {model_name} -> {out_dir}")
