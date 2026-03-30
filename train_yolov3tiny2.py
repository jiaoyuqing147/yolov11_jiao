
import warnings

# 精准只关掉 timm 相关的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.helpers")
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# ✅ 设置随机种子与确定性选项
def set_seed(seed=42):
    print(f"🔒 Setting seed = {seed} for full reproducibility")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    torch.set_num_threads(8)  # 控制 CPU 线程数

# ✅ 设置多线程加载数据时每个 worker 的随机种子（DataLoader 级别）
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ✅ 主函数入口
if __name__ == '__main__':
    set_seed(42)

    # ✅ 选择模型结构（自定义yaml）
    model = YOLO(r'ultralytics/cfg/models/v3/yolov3-tiny.yaml')

    # ✅ 输出模型结构和 GFLOPs
    model_info(model.model, verbose=True, imgsz=640)

    # ✅ 启动训练（全参数可控 + 可复现）
    model.train(
        data='ultralytics/cfg/datasets/GTSDB_chu.yaml',  # 修改为你的数据集 yaml
        task='detect',
        imgsz=640,
        epochs=200,
        batch=48,
        workers=8,  # 可设置为多线程，配合 seed_worker 保证复现
        device='cuda',  # 'cpu' or 'cuda'
        cache='ram',  # ✅ 高速缓存到内存
        optimizer='SGD',
        resume=False,
        amp=True,
        project='runsYOLOv3/GTSDB_train200',
        name='exp',

        # ✅ 与 batch/epoch 稳定性有关
        single_cls=False,
        close_mosaic=9999,
        mosaic=0.0,
        mixup=0.0,

        # ✅ 学习率与优化器设置
        # lr0=0.000625,  # 0.01 * batch / 64
        # weight_decay=0.0002,
        # momentum=0.9,
        # warmup_epochs=2,
        # cos_lr=True,

        # ✅ 复现关键参数
        seed=42,
        deterministic=True,

    )