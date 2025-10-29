#此脚本负责推理模型结果，指定图像文件夹，输出带框体的图像和txt边框的备注
import warnings
warnings.filterwarnings('ignore')

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
    weights_path = r"runsTT100k130/yolo11_train/exp/weights/best.pt"
    img_folder   = r"E:\DataSets\ceshi"
    output_path  = r"E:\DataSets\ceshiresult"

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 加载模型 =====
    model = YOLO(weights_path)
    model.to(DEVICE)
    print(f"Model device: {model.device}")

    # ===== 准备颜色（可自定义覆盖）=====
    # 覆盖示例：0->红  1->绿  2->蓝；其余按 tab20 自动分配
    # color_overrides = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    # colors = make_colors(len(model.names), overrides=color_overrides)

    #-----------上面的颜色不好看，自定义颜色
    # ===== 自定义颜色表 =====
    # 这里每个元组是 (R, G, B)，你可以随意调整
    custom_colors = [
        (255, 0, 0),  # 红
        (0, 176, 80),  # 深绿
        (0, 112, 192),  # 深蓝
        (255, 192, 0),  # 金黄
        (255, 0, 255),  # 洋红
        (255, 128, 0),  # 橙
        (128, 0, 255),  # 紫
        (0, 255, 255),  # 青
    ]
    # 如果类别数量大于颜色数量，自动循环使用
    colors = [custom_colors[i % len(custom_colors)] for i in range(len(model.names))]
    # -----------自定义颜色


    # ===== 收集图片 =====
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths += list(Path(img_folder).rglob(ext))
    img_paths.sort()

    for img_path in img_paths:
        # 推理（Ultralytics 会把坐标映射回原图尺寸）
        '''
        device=0 等价于 device="cuda:0"；若你的 GPU 不是第 0 块，用 1/2…
        half=True 仅在 GPU 生效；CPU 必须是 False。
        不要设置 dnn=True，那会走 OpenCV CPU 推理。
        '''
        results = model.predict(
            img_path,
            imgsz=640,
            conf=0.25,
            iou=0.50,
            device=DEVICE,  # 关键：用 GPU（0）或 CPU（'cpu'）
            half=USE_HALF,  # GPU 上启用 FP16
            verbose=False
        )

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()  # [N,4] -> x1,y1,x2,y2  绝对像素坐标
        confs = r.boxes.conf.cpu().numpy()  # [N]
        clses = r.boxes.cls.cpu().numpy().astype(int)  # [N]

        # 打开原图并绘制
        img = Image.open(img_path).convert("RGB")
        img = draw_boxes(img, boxes, confs, clses, model.names, colors)

        # 保存带框图
        save_img = out_dir / f"{img_path.stem}_result.jpg"
        img.save(save_img, quality=95)

        # 保存 YOLO txt（归一化中心点+宽高+置信度）
        W, H = img.size
        save_txt = out_dir / f"{img_path.stem}.txt"
        with open(save_txt, "w", encoding="utf-8") as f:
            for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clses):
                x_c = (x1 + x2) / 2 / W
                y_c = (y1 + y2) / 2 / H
                w   = (x2 - x1) / W
                h   = (y2 - y1) / H
                f.write(f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {float(conf):.6f}\n")

        print(f"Saved: {save_img}  |  {save_txt}")
