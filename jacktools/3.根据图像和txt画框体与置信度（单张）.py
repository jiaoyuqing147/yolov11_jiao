# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ================= 路径配置 =================
# img_path = r"E:\DataSets\forpaper\ceshiTT100K\9447.jpg"
# txt_path = r"E:\DataSets\forpaper\ceshiTT100K\9447.txt"
# img_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115.jpg"
# txt_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115.txt"
# img_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115_1280_crop.png"
# txt_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115_1280_crop.txt"
img_path = r"E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation\result_XGradCAM_crop.png"
txt_path = r"E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation\result_XGradCAM_crop.txt"

# 导出 PDF（矢量标注），你也可以改成 .svg
save_path = r"E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation\result_XGradCAM_crop_vec.pdf"

# ================= 调色板（文本底色用） =================
hexs = (
    'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
    '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
    '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
    'FF95C8', 'FF37C7'
)
# hexs = (
#     'FF3838',  # 红 ✔
#     'FF9D97',  # 粉红 ✔
#     'FF701F',  # 红橙 ✔
#     'FFB21D',  # 橙黄 ✔
#     'FF5540',  # 🔴 替换原 CFD231 绿色 → 红橙
#     'FF2D55',  # 🔴 替换原 48F90A 荧光绿 → 玫红
#     'FF6F61',  # 🔴 替换原 92CC17 黄绿 → 暗粉红
#     'FF4D73',  # 🔴 替换原 3DDB86 绿青 → 暗玫红
#     'FF1C3B',  # 🔴 替换原 1A9334 深绿 → 深红
#     'FF6DAE',  # 🔴 替换原 00D4BB 蓝绿 → 亮粉红
#     'FF4E80',  # 🔴 替换原 2C99A8 青蓝 → 玫紫
#     'FF1E8F',  # 🔴 替换原 00C2FF 蓝 → 偏紫红
#     'B40030',  # 🔴 替换原 344593 深蓝 → 酒红
#     'C00062',  # 🔴 替换原 6473FF 亮蓝紫 → 深玫
#     '8C0033',  # 🔴 替换原 0018EC 蓝 → 暗酒红
#     '8438FF',  # 紫红 ✔（保留）
#     '520085',  # 暗紫红 ✔（保留）
#     'CB38FF',  # 粉紫 ✔（保留）
#     'FF95C8',  # 粉色 ✔（保留）
#     'FF37C7',  # 玫红 ✔（保留）
# )

# hexs = (
#     'FF0000','FF0000','FF0000','FF0000','FF0000',
#     'FF0000','FF0000','FF0000','FF0000','FF0000',
#     'FF0000','FF0000','FF0000','FF0000','FF0000',
#     'FF0000','FF0000','FF0000','FF0000','FF0000'
# )

def hex2rgb01(h):
    """'#RRGGBB' -> (r,g,b) in [0,1] for matplotlib"""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

palette = [hex2rgb01(c) for c in hexs]


# ================= 读取图像 =================
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"图像读取失败：{img_path}")

# OpenCV BGR -> RGB（matplotlib 用 RGB）
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# ================= 创建画布 =================
# figsize 按图像宽高比设置，dpi 决定位图底图分辨率（框和字是矢量）
aspect = w / h
fig_height = 6  # 你可以调大一点，比如 8
fig_width = fig_height * aspect

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
ax.imshow(img)
ax.axis("off")   # 不要坐标轴


# ================= 读取标签并画框/文字 =================
with open(txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    a = line.strip().split()
    if len(a) < 5:
        # 连 cls x y w h 都不齐，跳过
        continue

    cls = int(a[0])

    if len(a) >= 6:
        # 预测结果：cls x y w h score
        x, y, bw, bh, score = map(float, a[1:6])
    else:
        # Groundtruth：cls x y w h（没有 score）
        x, y, bw, bh = map(float, a[1:5])
        score = None  # 用 None 标记为 GT

    # 只对有 score 的行做置信度过滤
    # if score is not None and score < 0.5:
    #     continue

    # YOLO -> 像素坐标
    cx, cy = x * w, y * h
    ww, hh = bw * w, bh * h
    x1, y1 = cx - ww / 2, cy - hh / 2

    # ====== 1) 画方框（纯红，矢量） ======
    rect = Rectangle(
        (x1, y1),           # 左上角
        ww, hh,             # 宽高
        linewidth=5.5,      # 这里调方框粗细
        edgecolor="red",    # 纯红色
        facecolor="none" ,   # 不填充
        alpha = 1.0  # ← 透明度
    )
    ax.add_patch(rect)

    # ====== 2) 文本 + 背景（矢量） ======
    label = f"id:{cls} {score:.2f}" if score is not None else f"id:{cls}"


    # 文本背景颜色：从调色板取
    bg_color = palette[cls % len(palette)]  # (r,g,b) in [0,1]

    # 文本位置：放在框左上角上方一点
    text_x = x1
    # text_y = y1 - 2  # 稍微往上移一点
    text_y = y1
    # 注意：fontsize 控制文字大小
    ax.text(
        text_x, text_y,
        label,
        fontsize=40,         # ← 这里调文本大小，8, 10, 12...
        color="white",      # 文字颜色
        # color="red",
        va="bottom",        # 垂直对齐：文字底部对齐指定位置
        ha="left",          # 水平对齐：左对齐
        bbox=dict(
            facecolor=bg_color,
            alpha=0.5,      # 文本背景透明度（0~1），0.9 接近不透明
            edgecolor="none",
            pad=0.2         # 文本与框之间的内边距
        )
    )

# ================= 保存为 PDF（矢量标注） =================
plt.tight_layout(pad=0)

# --- 1) 保存 PDF（矢量） ---
save_path_pdf = save_path  # 你原来的路径
fig.savefig(save_path_pdf, bbox_inches="tight", pad_inches=0)

# --- 2) 保存 PNG（高分辨率位图） ---
save_path_png = save_path.replace(".pdf", "_hd.png")
fig.savefig(save_path_png, dpi=1200, bbox_inches="tight", pad_inches=0)

plt.close(fig)

print("✔ 已保存（矢量）：", save_path_pdf)
print("✔ 已保存（PNG高清）：", save_path_png)

