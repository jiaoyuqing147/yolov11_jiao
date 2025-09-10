import cv2
# python
from ultralytics.solutions.heatmap import Heatmap

# 1) 初始化热力图解决方案
#   - model: 可用你项目里的权重，例如 "yolo11n.pt" 或你训练好的 best.pt
#   - show: 设为 False，避免窗口弹出
heatmap = Heatmap(model="runsMTSD/yolo11_train/exp/weights/best.pt", show=True)

# 2) 读取你的单张图像
img_path = "E:\p1840144.jpg"
im0 = cv2.imread(img_path)

# 3) 生成热力图结果（会在 im0 上做半透明叠加）
out = heatmap.generate_heatmap(im0)

# 4) 保存结果
cv2.imwrite("your_image_heatmap.jpg", out)
print("已保存：your_image_heatmap.jpg")