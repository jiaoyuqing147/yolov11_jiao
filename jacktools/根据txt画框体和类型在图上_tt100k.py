import cv2
import os

# === 配置 ===
img_path = r"E:\DataSets\ceshiTT100k\9447.jpg"
label_path = r"E:\DataSets\ceshiTT100k\9447.txt"
save_drawn_path = r"E:\DataSets\ceshiTT100k\9447_640_drawn.png"
save_resized_path = r"E:\DataSets\ceshiTT100k\9447_640.png"   # ← 新增：保存缩放后的原图

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
img = cv2.imread(img_path)
h, w = img.shape[:2]

# —— resize 到 640×640 —— #
target = (640, 640)
if (w, h) != target:
    img = cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]

# === 新增：保存缩放后的原图 ===
cv2.imwrite(save_resized_path, img)
print(f"缩放后的原图已保存到: {save_resized_path}")

# === 2. 读取txt并绘制 ===
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        cls, x_c, y_c, bw, bh = map(float, line.split())
        cls = int(cls)

        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)

        c = (0, 0, 255)
        line_type = cv2.LINE_AA

        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1, lineType=line_type)

        label = names[cls] if cls < len(names) else str(cls)
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            img, label,
            (x1, max(y1 - 5, 0)),
            font_face,
            0.3,
            c,
            1,
            lineType=line_type
        )

# === 3. 保存结果 ===
cv2.imwrite(save_drawn_path, img)
print(f"标注图已保存到: {save_drawn_path}")
