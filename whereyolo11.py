from ultralytics import YOLO
import os

# 检查 yolo11s.yaml 是否存在
print("yolo11s.yaml 是否存在:", os.path.exists("yolo11s.yaml"))

# 直接加载 YOLO 模型
model = YOLO("yolo11s.yaml")

# 检查实际加载的 YAML 配置
print("模型参数:", model.model.args)
