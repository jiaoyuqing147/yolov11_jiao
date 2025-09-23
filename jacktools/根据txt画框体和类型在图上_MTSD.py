import cv2
import os

# === 配置 ===
img_path = r"E:\DataSets\ceshiMTSD\p1840115.jpg"
label_path = r"E:\DataSets\ceshiMTSD\p1840115.txt"
save_drawn_path = r"E:\DataSets\ceshiMTSD\p1840115_640_drawn.png"
save_resized_path = r"E:\DataSets\ceshiMTSD\p1840115_640.png"   # ← 新增：保存缩放后的原图

# YOLO 类别名列表
names = [
  "U-turn", "Keep right", "Keep left", "Pass either side", "Stop", "No Left Turn", "No right turn", "No U-turn", "No entry",
  "Weight limit sign 5T", "Height limit sign 2.-m", "Height limit sign 3.-m", "Height limit sign 4.-m", "Height limit sign 5.-m",
  "Speed Limit 30", "Speed Limit 40", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70", "Speed Limit 80", "Speed Limit 90", "Speed Limit 110",
  "No Entry for Vehicles Exceeding 5T", "Heavy vehicles no driving on right lane", "No Parking", "No Stopping", "Give way",
  "Road Work", "Camera operation zone", "Crosswind area", "Caution! Hump", "Hump ahead", "Towing zone", "Left bend",
  "Pedestrian crossing opt1", "Pedestrian crossing opt2", "School children crossing opt1", "School children crossing opt2",
  "Caution", "Narrow roads on the left", "Traffic lights ahead", "Obstacles ahead", "Crossroads to the right", "Crossroads to the left",
  "Exit to the left", "Crossroads", "Minor road on right", "Minor road on left", "Minor road on left opt2", "Cattle crossing",
  "Roundabout ahead", "Narrow bridge", "Split way", "Curve on the left"
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
