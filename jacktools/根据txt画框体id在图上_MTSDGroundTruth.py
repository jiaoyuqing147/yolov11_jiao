import cv2
import os

# -------------------------------------------------------
# 10 种固定的鲜艳颜色（按 id 循环使用）——和 yolov11_heatmap 保持一致
# -------------------------------------------------------
COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    # (0, 0, 255),    # 蓝（你这份代码里是注释掉的，我也保持一致）
    (255, 128, 0),    # 橙
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青
    (128, 0, 255),    # 紫
    (0, 128, 255),    # 蓝绿
    (128, 255, 0),    # 黄绿
    (255, 0, 128),    # 粉
]

# ====== 参数：和 yolov11_heatmap.__init__ 默认一致 ======
BOX_THICKNESS = 1       # box_thickness
FONT_THICKNESS = 1      # font_thickness
FONT_SCALE = 0.8        # font_scale（真正用的时候会乘 0.6）
ANTIALIAS = True
LINE_TYPE = cv2.LINE_AA if ANTIALIAS else cv2.LINE_8

# ===================
# 配置路径
# ===================
img_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115.jpg"
label_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115.txt"
save_resized_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115_640.png"
save_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115_640_drawn.png"


# ===================
# 1. 读取图像并缩放到 640×640（方便和 Grad-CAM 对齐）
# ===================
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {img_path}")

h, w = img.shape[:2]
target = (640, 640)
if (w, h) != target:
    img = cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]

cv2.imwrite(save_resized_path, img)

# ===================
# 2. 文字避让结构（现在不用了，先保留变量）
# ===================
used_label_boxes = []   # 每个元素：(tx, ty, tw, th)


# ===================
# 3. 读取 txt 并绘制框 + id（GroundTruth，不要置信度）
# ===================
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            print(f"[WARN] 行格式不足 5 列，跳过：{line}")
            continue

        # 支持：cls x y w h [conf]
        cls = float(parts[0])
        x_c = float(parts[1])
        y_c = float(parts[2])
        bw  = float(parts[3])
        bh  = float(parts[4])
        cls_id = int(cls)

        # —— 原来的置信度读取逻辑（预测框用），现在是 GT，不需要置信度 —— #
        # if len(parts) >= 6:
        #     conf = float(parts[5])
        # else:
        #     conf = 1.0  # 没有写置信度就默认 1.0（和你 txt 读取逻辑一致）

        # Ground Truth 情况：即使有第 6 列也忽略，不使用置信度
        # conf = None

        # YOLO 归一化坐标 -> 像素坐标（此时 img 已经是 640×640）
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # 颜色按 id 循环选取（与 yolov11_heatmap 完全一致）
        color = COLORS[cls_id % len(COLORS)]

        # ====== 画框：和 yolov11_heatmap 一致 ======
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            int(BOX_THICKNESS),
            lineType=LINE_TYPE
        )

        # ====== 文字：GT 版本，只显示 id，不带置信度 ======
        # label = f"id:{cls_id}_{conf:.2f}"  # ← 原来预测框可视化会用到置信度
        label = f"id:{cls_id}"              # ← 现在是 GroundTruth，只显示类别 id

        # 字体参数：与 yolov11_heatmap.draw_detections 中完全一致
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = float(FONT_SCALE) * 0.6
        thickness = max(1, int(round(FONT_THICKNESS)))

        # 当前文字尺寸（现在不做避让，只是保留计算）
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 默认放在框上方
        tx = x1
        ty = max(y1 - 5, th + baseline)

        # —— 原来的文字避让逻辑，已经注释掉 —— #
        # while any(
        #     tx < ux + uw and tx + tw > ux and
        #     ty - th < uy and ty > uy - uh
        #     for (ux, uy, uw, uh) in used_label_boxes
        # ):
        #     ty += th + baseline + 2  # 往下挪一行

        # used_label_boxes.append((tx, ty, tw, th + baseline))

        # 绘制文字（固定位置）
        cv2.putText(
            img,
            label,
            (tx, ty),
            font,
            font_scale,
            color,
            thickness,
            lineType=LINE_TYPE
        )


# ===================
# 4. 保存结果
# ===================
cv2.imwrite(save_path, img)
print(f"缩放后原图已保存到: {save_resized_path}")
print(f"标注图已保存到: {save_path}")
