import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import pandas as pd  # 用于保存CSV

if __name__ == '__main__':
    #model = YOLO(r'runs/tt100k_yolo11_P234-deeper-FASFFHead_Jack_train/exp/weights/best.pt')
    model = YOLO(r'runs/tt100k_yolo11_WIOU+GNCloss_train/exp7/weights/best.pt')
    # model = YOLO(r'runs/tt100k_yolo11_P2-FASFFHead_train/exp/weights/best.pt')
    model.val(data=r"ultralytics/cfg/datasets/tt100k_chu.yaml",
              split='val',
              imgsz=640,
              batch=64,
              # rect=False,
              save_json=True, # 这个保存coco精度指标的开关
              #save_txt=True,  # 保存每个图片的预测文本，一般不用
              #save_conf=True,  # 保存每个预测的置信度，一般不用
              project='vals/tt100k_yolo11_WIOU+GNCloss_val',
              name='exp',
              )

