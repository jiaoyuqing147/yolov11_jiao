import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import torch

from ultralytics import RTDETR          # ✅ 换成 RTDETR
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


# ✅ DataLoader 的 worker 随机种子（如果后面想用的话）
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    set_seed(42)

    # ✅ 使用 RT-DETR 模型（你自定义的 yolo11_AIFI_2 版本）
    # 确保这个 yaml 是按 RT-DETR 的结构写的，放在 rt-detr 目录下
    model = RTDETR(r'ultralytics/cfg/models/rt-detr/yolo11_AIFI_1.yaml')

    # ✅ 输出模型结构和 GFLOPs
    model_info(model.model, verbose=True, imgsz=640)

    # ✅ 启动训练（RT-DETR 也是同一套 train 接口）
    model.train(
        data='ultralytics/cfg/datasets/MTSD_laptop.yaml',  # 你的数据集 yaml
        imgsz=640,
        epochs=100,
        batch=24,
        workers=0,
        device='cuda',
        cache='ram',
        optimizer='SGD',      # 如果想更贴近官方 RT-DETR，可以换成 'AdamW'
        resume=False,
        amp=True,
        project='runsYOGA/MTSD_yolo11_AIFI_1_RTDETR_train',
        name='exp',

        # ✅ 数据增强和复现相关
        single_cls=False,
        close_mosaic=9999,
        mosaic=0.0,
        mixup=0.0,

        # ✅ 复现关键参数
        seed=42,
        deterministic=True,
        # 下面这些是 DataLoader 级别的，Ultralytics 暂时不能直接传：
        # worker_init_fn=seed_worker,
        # persistent_workers=False,
        # shuffle=True,
    )
