import time
import yaml
import types
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def dict_to_obj(d):
    """将 dict 转为对象，使得可以 obj.xxx 访问"""
    return types.SimpleNamespace(**d)

def measure_dataloader_speed(
    data_yaml='ultralytics/cfg/datasets/tt100k_chu.yaml',
    img_dir='/home/jiaoyuqing/AlgorithmCodes/datasets/TT100K/tt100k_2021/yolo143/train',
    batch=32,
    workers=4,
    imgsz=640,
    num_batches=20
):
    cfg = get_cfg(overrides={
        'data': data_yaml,
        'imgsz': imgsz,
        'batch': batch,
        'workers': workers
    })

    data_dict = load_yaml('ultralytics/cfg/datasets/tt100k_chu.yaml')
    hyp_dict = load_yaml('ultralytics/cfg/default.yaml')
    hyp = dict_to_obj(hyp_dict)

    dataset = YOLODataset(
        img_path=img_dir,
        imgsz=imgsz,
        batch_size=batch,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix='',
        data=data_dict,  # ✅ 正确地传 dict
        task="detect"
    )

    train_loader = build_dataloader(
        dataset=dataset,
        batch=batch,
        workers=workers,
        shuffle=True
    )

    start = time.time()
    for i, data in enumerate(train_loader):
        if i == num_batches:
            break
    total_time = time.time() - start
    print(f"加载 {num_batches} 个 batch 用时: {total_time:.2f} 秒，平均每 batch 用时: {total_time/num_batches:.2f} 秒")

if __name__ == '__main__':
    measure_dataloader_speed(
        data_yaml='ultralytics/cfg/datasets/tt100k_chu.yaml',
        img_dir='/home/jiaoyuqing/AlgorithmCodes/datasets/TT100K/tt100k_2021/yolo143/images/train',
        batch=24,
        workers=2,
        imgsz=640,
        num_batches=48
    )
