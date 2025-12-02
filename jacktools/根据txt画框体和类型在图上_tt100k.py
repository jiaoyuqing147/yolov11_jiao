import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =======================
# 配置区：修改路径即可
# =======================
img_path = r"E:\DataSets\forpaper\ceshiTT100k\9447.jpg"
label_path = r"E:\DataSets\forpaper\ceshiTT100k\9447.txt"

save_drawn_path = r"E:\DataSets\forpaper\ceshiTT100k\9447_640_drawn.png"        # 缩放后 + 框体 + id（位图）
save_resized_path = r"E:\DataSets\forpaper\ceshiTT100k\9447_640.png"            # 仅缩放后的原图（位图）
save_drawn_original_path = r"E:\DataSets\forpaper\ceshiTT100k\9447_drawn.png"   # 原始尺寸 + 框体 + id（位图）

# 新增：矢量文件输出路径（论文/PPT 用）
save_vector_640_pdf = r"E:\DataSets\forpaper\ceshiTT100k\9447_640_vector.pdf"
save_vector_640_svg = r"E:\DataSets\forpaper\ceshiTT100k\9447_640_vector.svg"

# YOLO 类别名列表（当前不用，仅保留备份）
# names = [
#   "pl80", "p6", "ph4.2", "pa13", "im", "w58", "pl90", "il70", "p5", "pm55", "pl60", "ip", "p11", "pdd", "wc", "i2r",
#   "w30", "pmr", "p23", "pl15", "pm10", "pss", "w34", "iz", "p1n", "pr70", "pg", "il80", "pb", "pbm", "pm40", "ph4",
#   "w45", "i4", "pl70", "i14", "p29", "pne", "pr60", "ph4.5", "p12", "p3", "pl5", "w13", "p14", "i4l", "pr30", "p17",
#   "ph3", "pt", "pl30", "pctl", "pr50", "pm35", "i1", "pcd", "pbp", "pcr", "ps", "w18", "p10", "pn", "pa14", "p2", "ph2.5",
#   "w55", "pw3", "pw4.5", "i12", "phclr", "i10", "i13", "w10", "p26", "p8", "w42", "il50", "p13", "pr40", "p25", "w41",
#   "pl20", "pm30", "pl40", "pmb", "pr20", "p18", "i2", "w22", "w47", "pl120", "ph2.8", "w32", "pm15", "ph5", "pw3.2",
#   "pl10", "il60", "w57", "pl100", "p16", "pl110", "w59", "w20", "ph2", "p9", "il100", "p19", "ph3.5", "pa10", "pcl",
#   "pl35", "p15", "phcs", "w3", "pl25", "il110", "p1", "w46", "pn-2", "w63", "pm20", "i5", "il90", "w21", "p27",
#   "pl50", "ph2.2", "pm2", "pw4"
# ]

# 10 种鲜艳颜色（BGR for cv2 / RGB for matplotlib 时注意顺序）
colors_bgr = [
    (255,   0,   0),   # Red
    (255, 165,   0),   # Orange
    (255, 255,   0),   # Yellow
    (  0, 255,   0),   # Green
    (  0, 255, 255),   # Cyan
    (  0,   0, 255),   # Blue
    (255,   0, 255),   # Magenta
    (128,   0, 255),   # Purple
    (255,  20, 147),   # Deep Pink
    (  0, 128, 255),   # Sky Blue
]

# =======================
# 1. 读取图像
# =======================
if not os.path.exists(img_path):
    raise FileNotFoundError(f"图片路径不存在: {img_path}")

img_orig = cv2.imread(img_path)
if img_orig is None:
    raise RuntimeError(f"cv2.imread 读取图片失败，请检查路径或文件是否损坏: {img_path}")

h0, w0 = img_orig.shape[:2]
img_drawn_original = img_orig.copy()

# 缩放到 640×640
img_640 = cv2.resize(img_orig, (640, 640), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(save_resized_path, img_640)
print(f"缩放后的原图已保存到: {save_resized_path}")

# 为了后续矢量绘制，先把所有 bbox / cls 读出来，存到列表里
bboxes_640 = []      # (cls, x1, y1, x2, y2) in 640x640
bboxes_orig = []     # (cls, x1o, y1o, x2o, y2o) in original size

# =======================
# 2. 读取 label
# =======================
if not os.path.exists(label_path):
    raise FileNotFoundError(f"标签文件不存在: {label_path}")

with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        cls, x_c, y_c, bw, bh = map(float, line.split())
        cls = int(cls)

        # ---- 640×640 上的坐标 ----
        x1 = int((x_c - bw / 2) * 640)
        y1 = int((y_c - bh / 2) * 640)
        x2 = int((x_c + bw / 2) * 640)
        y2 = int((y_c + bh / 2) * 640)
        bboxes_640.append((cls, x1, y1, x2, y2))

        # ---- 原图上的坐标 ----
        x1o = int((x_c - bw / 2) * w0)
        y1o = int((y_c - bh / 2) * h0)
        x2o = int((x_c + bw / 2) * w0)
        y2o = int((y_c + bh / 2) * h0)
        bboxes_orig.append((cls, x1o, y1o, x2o, y2o))

# =======================
# 3. 用 OpenCV 画（位图）
# =======================
img_640_drawn = img_640.copy()

for cls, x1, y1, x2, y2 in bboxes_640:
    color = colors_bgr[cls % len(colors_bgr)]
    cv2.rectangle(img_640_drawn, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    label = f"id {cls}"
    cv2.putText(
        img_640_drawn,
        label,
        (x1, max(y1 - 5, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        lineType=cv2.LINE_AA,
    )

for cls, x1o, y1o, x2o, y2o in bboxes_orig:
    color = colors_bgr[cls % len(colors_bgr)]
    cv2.rectangle(img_drawn_original, (x1o, y1o), (x2o, y2o), color, 2, lineType=cv2.LINE_AA)
    label = f"id {cls}"
    cv2.putText(
        img_drawn_original,
        label,
        (x1o, max(y1o - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        lineType=cv2.LINE_AA,
    )

cv2.imwrite(save_drawn_path, img_640_drawn)
print(f"标注图(640x640, 位图)已保存到: {save_drawn_path}")

cv2.imwrite(save_drawn_original_path, img_drawn_original)
print(f"原始尺寸标注图(位图)已保存到: {save_drawn_original_path}")

# =======================
# 4. 用 matplotlib 画（矢量框和文字）
# =======================
# 注意：matplotlib 用的是 RGB，需要把 BGR → RGB
img_640_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)

# 创建画布，figsize + dpi 可以控制最终像素大小
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # 约 1800x1800 像素
ax.imshow(img_640_rgb)
ax.axis("off")  # 去掉坐标轴

for cls, x1, y1, x2, y2 in bboxes_640:
    # BGR → RGB
    bgr = colors_bgr[cls % len(colors_bgr)]
    color_rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)

    w = x2 - x1
    h = y2 - y1

    # 矩形是矢量对象
    rect = Rectangle(
        (x1, y1),
        w,
        h,
        linewidth=0.6,
        edgecolor=color_rgb,
        facecolor="none",
    )
    ax.add_patch(rect)

    # 文字也是矢量
    ax.text(
        x1,
        y1 - 4,
        f"id {cls}",
        fontsize=6,          # 可以根据缩放调
        color=color_rgb,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.5),
    )

# 保存为 PDF（矢量）
plt.savefig(
    save_vector_640_pdf,
    bbox_inches="tight",
    pad_inches=0,
)
print(f"矢量标注图(640x640) PDF 已保存到: {save_vector_640_pdf}")

# 保存为 SVG（也是矢量，Word/PPT 里放大很清晰）
plt.savefig(
    save_vector_640_svg,
    bbox_inches="tight",
    pad_inches=0,
    format="svg",
)
print(f"矢量标注图(640x640) SVG 已保存到: {save_vector_640_svg}")

plt.close(fig)
