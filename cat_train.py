from ultralytics import YOLO
from ultralytics.utils.torch_utils import EarlyStopping


if __name__ == '__main__':  # ✅ 解决 Windows 多进程问题
    # 1️⃣ 从零开始训练 YOLOv11
    model = YOLO("ultralytics/cfg/models/11/yolo11-BiFPN-MobileNetV4.yaml")
    early_stop = EarlyStopping(patience=100)  # 设置为 300
    # 2️⃣ 设置早停法
    # early_stop = {
    #     'patience': 10,  # 停止的耐心次数（如果验证损失在10个epoch内没有改善，则停止训练）
    #     'verbose': True,  # 输出早停信息
    #     'min_delta': 0.001,  # 每次提升的最小阈值
    # }


    # 3️⃣ 设置保存路径
    project_dir = "runs/cat_detect"  # 设置训练结果的根目录
    experiment_name = "cat_detection_model"  # 设置子目录名称

    # 4️⃣ 训练模型
    train_results = model.train(
        data="coco_cat.yaml",  # ✅ 确保数据集路径正确
        epochs=100,  # 训练轮数
        imgsz=640,  # ✅ 建议尝试 960 以提高小目标检测能力
        device="cuda",  # ✅ 使用 GPU 训练
        batch=16,  # ✅ 调整 batch size 适应 GPU
        workers=0,  # ✅ 解决 Windows 多进程问题，避免 multiprocessing 错误
        optimizer="AdamW",  # ✅ AdamW 比 SGD 更适合小目标检测
        lr0=0.001,  # ✅ 默认 0.01 可能太大，调整为 0.001（更稳定）
        weight_decay=0.0005,  # ✅ 防止过拟合
        amp=0,  # ✅ 关闭 AMP 适配 GTX 1660 SUPER，防止 NaN 问题
        # overrides={'callbacks': [early_stop]},  # 将早停法回调添加到训练中
        project=project_dir,  # 设置训练结果的根目录
        name=experiment_name  # 设置子目录名称
    )

    # 5️⃣ 评估模型（在验证集上测试）
    metrics = model.val()

    # # 6️⃣ 测试模型（推理）
    # results = model("path/to/image.jpg")
    # results[0].show()

    # # 7️⃣ 导出模型为 ONNX（用于部署）
    # path = model.export(format="onnx")
