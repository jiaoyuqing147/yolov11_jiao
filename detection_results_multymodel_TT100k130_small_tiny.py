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


def draw_boxes(img, boxes_xyxy, confs, classes, class_names, colors):
    # 先转 RGBA，后面才能做透明叠加
    img = img.convert("RGBA")
    W, H = img.size

    # ===== 字体大小自适应 =====
    try:
        font_size = max(8, int(min(W, H) * 0.05))
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # ===== 框粗细自适应 =====
    BOX_WIDTH = max(2, int(min(W, H) * 0.004))

    # 先画不透明框
    draw = ImageDraw.Draw(img)

    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, classes):
        cls = int(cls)
        color = colors[cls % len(colors)]
        color = tuple(min(255, int(c * 1.2)) for c in color)

        # ===== 画框（不透明）=====
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

        # ===== 标签 =====
        label = f"{class_names[cls]} {conf:.2f}"

        try:
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = r - l, b - t
        except:
            text_w, text_h = draw.textsize(label, font=font)

        tx = max(0, min(int(x1), W - text_w - 1))
        ty = max(0, int(y1) - text_h - 6)

        # ===== 半透明标签背景 =====
        pad = 4
        alpha = 200  # 0~255，推荐 100~150

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        overlay_color = (color[0], color[1], color[2], alpha)
        overlay_draw.rectangle(
            [tx - pad, ty - pad, tx + text_w + pad, ty + text_h + pad],
            fill=overlay_color
        )

        img = Image.alpha_composite(img, overlay)

        # 重新建立 draw，继续在合成后的图上画字
        draw = ImageDraw.Draw(img)

        # ===== 白字 =====
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    # 保存成 jpg 前最好转回 RGB
    return img.convert("RGB")

if __name__ == "__main__":
    # ===== 路径配置 =====
    models_root = Path(r"runsTT100k130")                      # 所有模型所在的根目录
    model_dirs = [
        # "yolo11-OECSOSAInterleave_train200",
        "yolo11_train200",
        # "yolo11-FASFFHead_P234_train200",
        "yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200",
        "yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation",
        # "yolo11x-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train300(batch16worker16)",
    ]
    img_folder   = Path(r"E:\DataSets\tt100k_2021\size_split_test\tiny\images")
    base_out_dir = Path(r"E:\DataSets\tt100k_2021\size_split_test\tinyresult")           # 所有结果的统一根目录
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

            # img = Image.open(img_path).convert("RGB")
            # img = draw_boxes(img, boxes, confs, clses, model.names, colors)
            #
            # save_img = out_dir / f"{img_path.stem}_result.jpg"
            # img.save(save_img, quality=95)

            # W, H = img.size
            with Image.open(img_path) as im:
                W, H = im.size
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
