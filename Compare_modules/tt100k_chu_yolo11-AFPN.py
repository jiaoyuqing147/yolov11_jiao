import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


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

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Warning: deterministic algorithms could not be fully enabled: {e}")

    torch.set_num_threads(8)


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    set_seed(42)

    model = YOLO(r'ultralytics/cfg/models/11compare/yolo11-AFPN.yaml')

    model_info(model.model, verbose=True, imgsz=640)

    model.train(
        data=r'ultralytics/cfg/datasets/tt100k_chu.yaml',
        task='detect',
        imgsz=640,
        epochs=200,
        batch=64,
        workers=16,
        device='cuda',
        cache='ram',
        optimizer='SGD',

        # 新对比实验默认不要 resume，避免误接旧实验
        resume=False,

        amp=True,
        project=r'runsCompare_modules/tt100k_chu_yolo11-AFPN_train200',
        name='exp',

        # 每10个epoch保存一次权重
        save_period=10,

        single_cls=False,
        close_mosaic=9999,
        mosaic=0.0,
        mixup=0.0,

        seed=42,
        deterministic=True,
    )