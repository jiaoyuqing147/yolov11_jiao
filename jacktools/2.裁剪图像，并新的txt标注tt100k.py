# -*- coding: utf-8 -*-
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_and_crop(
    img_path: str,
    x: float, y: float, w: float, h: float,
    out_dir: str,
    rect_color=(255, 0, 0),  # 红色
    rect_width=4,            # 边框粗细
    label: str | None = None # 可选：在框上方写点字
):
    """
    在图像上画出矩形 (x,y,w,h)，保存标注图；再裁剪该区域保存到 out_dir。
    坐标单位：像素，(x,y) 是左上角。
    """

    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 打开图像
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # 边界裁剪（避免越界）
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))

    # 如果给的框超出边界，这里会自动夹到图像内；也可以选择直接报错
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"无效的裁剪区域：({x},{y},{w},{h}) 在图像尺寸 ({W}x{H}) 外。")

    # 1) 在原图上画矩形与可选标签
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=rect_color, width=rect_width)

    if label:
        try:
            # 新版 Pillow 优先用 textbbox 计算文本尺寸
            l, t, r, b = draw.textbbox((0, 0), label)
            tw, th = r - l, b - t
        except Exception:
            # 兼容旧版
            font = ImageFont.load_default()
            tw, th = draw.textsize(label, font=font)

        tx = max(0, min(x1, W - tw - 1))
        ty = max(0, y1 - th - 2)
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=rect_color)
        draw.text((tx, ty), label, fill=(0, 0, 0))

    # 保存带框图
    marked_name = f"{img_path.stem}_box_{x1}_{y1}_{x2-x1}x{y2-y1}.jpg"
    marked_path = out_dir / marked_name
    img.save(marked_path, quality=95)

    # 2) 裁剪并保存
    crop = Image.open(img_path).convert("RGB").crop((x1, y1, x2, y2))
    crop_name = f"{img_path.stem}_crop_{x1}_{y1}_{x2-x1}x{y2-y1}.png"
    crop_path = out_dir / crop_name
    crop.save(crop_path)

    print(f"[OK] 标注图: {marked_path}")
    print(f"[OK] 裁剪图: {crop_path}")
    return str(marked_path), str(crop_path)


def crop_yolo_labels_rel(
    original_txt: str,
    crop_txt: str,
    x_rel: float, y_rel: float, w_rel: float, h_rel: float,
    img_w: int, img_h: int
):
    """
    使用“比例 ROI”生成裁剪区域内对应的 YOLO 新标签。

    original_txt: 原 txt 路径
        支持两种格式：
        1) cls cx cy w h
        2) cls cx cy w h conf   ← 会保留并写回到新 txt 中

    crop_txt: 输出新的 txt 路径（YOLO 格式，归一化到裁剪图）
    x_rel, y_rel, w_rel, h_rel: 相对整张图宽、高的比例（0~1）
    img_w, img_h: 原图尺寸（像素）
    """

    # 1) 把比例 ROI 转成像素 ROI，TT100K直接按比例裁就行了
    x = x_rel * img_w
    y = y_rel * img_h
    w = w_rel * img_w
    h = h_rel * img_h

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"无效的裁剪区域(比例)：({x_rel},{y_rel},{w_rel},{h_rel}) -> 像素({x1},{y1},{x2},{y2})"
        )

    crop_w = x2 - x1
    crop_h = y2 - y1

    with open(original_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_labels = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            # 这一行太短，不合法，跳过
            continue

        # 支持：
        # 1) 5 列: cls cx cy w h
        # 2) 6 列: cls cx cy w h conf
        cls_str, cx_str, cy_str, w_str, h_str = parts[:5]
        conf_str = parts[5] if len(parts) >= 6 else None  # ← 可能存在的置信度

        try:
            cls = int(cls_str)
            cx = float(cx_str) * img_w
            cy = float(cy_str) * img_h
            w_box = float(w_str) * img_w
            h_box = float(h_str) * img_h
            score = float(conf_str) if conf_str is not None else None
        except ValueError:
            continue

        # 原图像素 bbox（中心 -> 左上右下）
        x_min = cx - w_box / 2
        y_min = cy - h_box / 2
        x_max = cx + w_box / 2
        y_max = cy + h_box / 2

        # 与裁剪区域求交集
        inter_x1 = max(x_min, x1)
        inter_y1 = max(y_min, y1)
        inter_x2 = min(x_max, x2)
        inter_y2 = min(y_max, y2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            # 没有交集，跳过
            continue

        # 映射到裁剪图坐标系（左上角为 (0,0)）
        new_x_min = inter_x1 - x1
        new_y_min = inter_y1 - y1
        new_x_max = inter_x2 - x1
        new_y_max = inter_y2 - y1

        # 裁剪图中的 YOLO 归一化坐标
        new_cx = (new_x_min + new_x_max) / 2 / crop_w
        new_cy = (new_y_min + new_y_max) / 2 / crop_h
        new_w = (new_x_max - new_x_min) / crop_w
        new_h = (new_y_max - new_y_min) / crop_h

        # 过滤异常框
        if new_w <= 0 or new_h <= 0:
            continue

        if score is None:
            # 原来没有置信度：保持 5 列
            new_labels.append(
                f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n"
            )
        else:
            # 原来有置信度：写 6 列，保留原置信度
            new_labels.append(
                f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f} {score:.6f}\n"
            )

    # 写出新的裁剪标签文件
    with open(crop_txt, "w", encoding="utf-8") as f:
        f.writelines(new_labels)

    print(f"[OK] 生成裁剪标签: {crop_txt}  ({len(new_labels)} 个对象)")


if __name__ == "__main__":

    # ====== 路径配置 ======
    img_path = r"E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation\result_XGradCAM.png"
    txt_path = r"E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation\9447.txt"
    out_dir  = r"E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation"

    # ====== ROI 比例（根据原来 640x640 的 ROI 缩放）======
    # 例如原来是：x=285, y=265, w=210, h=75（在 640x640 图上）
    # x_rel = 285 / 640
    # y_rel = 265 / 640
    # w_rel = 210 / 640
    # h_rel = 75  / 640

    x_rel = 352 / 640
    y_rel = 265 / 640
    w_rel = 167 / 640
    h_rel = 75 / 640

    # ====== 读取当前图像尺寸 ======
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # 用比例算当前图的实际像素 ROI
    x = x_rel * W
    y = y_rel * H
    w = w_rel * W
    h = h_rel * H

    # 1) 画框 + 裁剪图像
    marked_path, crop_path = draw_and_crop(
        img_path, x, y, w, h,
        out_dir,
        rect_color=(0, 112, 192),
        rect_width=4,
        label="ROI"
    )

    # 2) 生成对应的裁剪 YOLO 标签（txt）
    crop_txt_path = Path(crop_path).with_suffix(".txt")

    crop_yolo_labels_rel(
        original_txt=txt_path,
        crop_txt=str(crop_txt_path),
        x_rel=x_rel, y_rel=y_rel, w_rel=w_rel, h_rel=h_rel,
        img_w=W, img_h=H
    )

    print("[DONE] 裁剪图 & 新标签 已完成")
