import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

if __name__ == '__main__':
    model_t = YOLO(r'runs/tt100k_yolo11_CIOU+BCELoss_train(batch48worker6)/exp/weights/best.pt')  # 此处填写教师模型的权重文件地址

    model_t.model.model[-1].set_Distillation = True  # 不用理会此处用于设置模型蒸馏

    model_s = YOLO(r'yolo11n.yaml')  # 学生文件的yaml文件 or 权重文件地址

    # 配置训练参数
    train_args = dict(
        model='yolo11n.yaml',
        data=r'ultralytics/cfg/datasets/tt100k_laptop.yaml',
        cache=False,
        imgsz=640,
        epochs=100,
        single_cls=False,  # 是否是单类别检测
        batch=1,
        close_mosaic=10,
        workers=0,
        device='cpu',
        optimizer='SGD',  # using SGD
        amp=True,  # 如果出现训练损失为Nan可以关闭amp
        project='runs/train',
        name='exp',
        model_t=model_t.model
    )
    
    # 创建训练器
    trainer = DetectionTrainer(overrides=train_args)
    
    # 确保模型被正确初始化
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    
    # 关键设置 - 激活蒸馏损失
    trainer.distillonline = True  # False or True
    trainer.logit_loss = True  # False or True
    
    print("蒸馏参数已设置：")
    print(f"- distillonline = {trainer.distillonline}")
    print(f"- logit_loss = {trainer.logit_loss}")
    print("现在应该能看到 dfeaLoss, dlineLoss, dlogitLoss 都有非零值！")
    
    # 开始训练
    trainer.train() 