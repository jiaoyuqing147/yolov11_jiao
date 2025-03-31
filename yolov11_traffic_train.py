from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO("yolo11s.pt")  # 加载 YOLOv11 预训练模型
    model = YOLO("yolo11s.yaml")
    model.train(data="ultralytics/cfg/datasets/traffic_sign.yaml", epochs=300, batch=16, imgsz=640, device=0)
