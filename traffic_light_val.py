from ultralytics import YOLO

if __name__ == '__main__':  # ✅ 解决 Windows 多进程问题
    # 1️⃣ 加载训练好的模型
    #model = YOLO("runs/detect/train23/weights/best.pt")  # 载入训练好的模型
    model = YOLO("runs/detect/train/weights/best.pt")  # 载入训练好的模型
    # 2️⃣ 指定数据集配置文件并进行验证
    metrics = model.val(data="traffic_light.yaml")  # 指定数据集的yaml文件

    # 3️⃣ 输出验证结果
    print(metrics)  # 输出验证结果
