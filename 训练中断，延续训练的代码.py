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
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)

if __name__ == '__main__':
    set_seed(42)

    ckpt = r'runsGTSDB/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200/exp/weights/last.pt'

    # 续训时从 last.pt 加载
    model = YOLO(ckpt)

    # 可选：看一下当前模型信息
    model_info(model.model, verbose=True, imgsz=640)

    model.train(
        resume=True
    )


#上次从哪断，这次就从哪续上