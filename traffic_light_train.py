from ultralytics import YOLO

if __name__ == '__main__':  # ✅ 解决 Windows 多进程问题
    # 1️⃣ 从零开始训练 YOLOv11

    # model = YOLO("yolo11.yaml")  # ✅ 确保路径正确
    #model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
    model = YOLO("ultralytics/cfg/models/11/yolo11_C3K2Lite.yaml")
    # 2️⃣ 训练模型
    train_results = model.train(
        #data="traffic_light_myxlab.yaml",  # ✅ 确保数据集路径正确
        data="traffic_light.yaml",  # ✅ 确保数据集路径正确
        epochs=100,  # 训练轮数
        imgsz=640,  # ✅ 建议尝试 960 以提高小目标检测能力
        device="cuda",  # ✅ 使用 GPU 训练
        batch=16,  # ✅ 调整 batch size 适应 GPU
        workers=0,  # ✅ 解决 Windows 多进程问题，避免 multiprocessing 错误
        optimizer="AdamW",  # ✅ AdamW 比 SGD 更适合小目标检测
        lr0=0.001,  # ✅ 默认 0.01 可能太大，调整为 0.001（更稳定）
        weight_decay=0.0005,  # ✅ 防止过拟合
        amp=0,  # ✅ 关闭 AMP 适配 GTX 1660 SUPER，防止 NaN 问题
    )

    # 3️⃣ 评估模型（在验证集上测试）
    metrics = model.val()

    # # 4️⃣ 测试模型（推理）
    # results = model("path/to/image.jpg")
    # results[0].show()
    #
    # # 5️⃣ 导出模型为 ONNX（用于部署）
    # path = model.export(format="onnx")


