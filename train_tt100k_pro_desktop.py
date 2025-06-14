import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO(r'ultralytics/cfg/models/11/yolo11-P234-FASFFHead_Jack.yaml')#使用这个结构,用大的版本就改字母yolo11l-P2-FASFFHead.yaml# 续训yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
    #model = YOLO(r'ultralytics/cfg/models/11/yolo11-P234-deeper-FASFFHead_Jack.yaml')#使用这个结构,用大的版本就改字母yolo11l-P2-FASFFHead.yaml# 续训yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
    #model = YOLO(r'ultralytics/cfg/models/11/yolo11-P234-deeper-FASFFHead_Jack1.yaml')
    model = YOLO(r'ultralytics/cfg/models/11/yolo11-SDI-BiFPN.yaml')
    #model = YOLO(r'runs/tt100k_yolo11_P234-deeper-FASFFHead_Jack_train/exp/weights/last.pt')#延续之前的训练
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load('yolov11n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    model.train(
                # data=r"ultralytics/cfg/datasets/tt100k_myxlab.yaml",
                #data=r"ultralytics/cfg/datasets/tt100k_chu.yaml",
                data=r"ultralytics/cfg/datasets/tt100k_desk.yaml",
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                task='detect',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=1,#chu 用64,cpu爆红,GPU还是很空闲,4080用16
                close_mosaic=9999, #不进行马赛克增强
                workers=1,
                device='0',
                optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                resume=False, # 续训的话这里填写True
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/tt100k_yolo11-SDI_train',
                name='exp',
                iou= 0.5,
                )

