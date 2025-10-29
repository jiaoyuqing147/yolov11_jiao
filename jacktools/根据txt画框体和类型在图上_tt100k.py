import cv2
import os

# === 配置 ===
img_path = r"E:\DataSets\ceshiTT100k\74631.jpg"
label_path = r"E:\DataSets\ceshiTT100k\74631.txt"
save_drawn_path = r"E:\DataSets\ceshiTT100k\74631_640_drawn.png"
save_resized_path = r"E:\DataSets\ceshiTT100k\74631_640.png"
save_drawn_original_path = r"E:\DataSets\ceshiTT100k\74631_drawn.png"  # ← 新增：原始图上绘制的保存路径

# YOLO 类别名列表
names = [
  "pl80", "p6", "ph4.2", "pa13", "im", "w58", "pl90", "il70", "p5", "pm55", "pl60", "ip", "p11", "pdd", "wc", "i2r",
  "w30", "pmr", "p23", "pl15", "pm10", "pss", "w34", "iz", "p1n", "pr70", "pg", "il80", "pb", "pbm", "pm40", "ph4",
  "w45", "i4", "pl70", "i14", "p29", "pne", "pr60", "ph4.5", "p12", "p3", "pl5", "w13", "p14", "i4l", "pr30", "p17",
  "ph3", "pt", "pl30", "pctl", "pr50", "pm35", "i1", "pcd", "pbp", "pcr", "ps", "w18", "p10", "pn", "pa14", "p2", "ph2.5",
  "w55", "pw3", "pw4.5", "i12", "phclr", "i10", "i13", "w10", "p26", "p8", "w42", "il50", "p13", "pr40", "p25", "w41",
  "pl20", "pm30", "pl40", "pmb", "pr20", "p18", "i2", "w22", "w47", "pl120", "ph2.8", "w32", "pm15", "ph5", "pw3.2",
  "pl10", "il60", "w57", "pl100", "p16", "pl110", "w59", "w20", "ph2", "p9", "il100", "p19", "ph3.5", "pa10", "pcl",
  "pl35", "p15", "phcs", "w3", "pl25", "il110", "p1", "w46", "pn-2", "w63", "pm20", "i5", "il90", "w21", "p27",
  "pl50", "ph2.2", "pm2", "pw4"
]

# === 1. 读取图像 ===
img_orig = cv2.imread(img_path)
h0, w0 = img_orig.shape[:2]

# === 复制一份用于绘制原始图标注 ===
img_drawn_original = img_orig.copy()

# === resize 到 640×640 并保存 ===
img = cv2.resize(img_orig, (640, 640), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(save_resized_path, img)
print(f"缩放后的原图已保存到: {save_resized_path}")

# === 2. 读取txt并绘制 ===
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        cls, x_c, y_c, bw, bh = map(float, line.split())
        cls = int(cls)

        # ---- 在缩放图上绘制 ---- #
        x1 = int((x_c - bw/2) * 640)
        y1 = int((y_c - bh/2) * 640)
        x2 = int((x_c + bw/2) * 640)
        y2 = int((y_c + bh/2) * 640)

        color = (0, 0, 255)
        line_type = cv2.LINE_AA
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=line_type)
        label = names[cls] if cls < len(names) else str(cls)
        cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, lineType=line_type)

        # ---- 在原始图上绘制 ---- #
        x1o = int((x_c - bw/2) * w0)
        y1o = int((y_c - bh/2) * h0)
        x2o = int((x_c + bw/2) * w0)
        y2o = int((y_c + bh/2) * h0)
        cv2.rectangle(img_drawn_original, (x1o, y1o), (x2o, y2o), color, 2, lineType=line_type)
        cv2.putText(img_drawn_original, label, (x1o, max(y1o - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=line_type)

# === 3. 保存结果 ===
cv2.imwrite(save_drawn_path, img)
print(f"标注图(640x640)已保存到: {save_drawn_path}")

cv2.imwrite(save_drawn_original_path, img_drawn_original)
print(f"原始尺寸标注图已保存到: {save_drawn_original_path}")
