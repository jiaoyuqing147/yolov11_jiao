
# class BaseTrainer:
#     def __init__(self):
#         print("👷 BaseTrainer 初始化中...")
#
# class DetectionTrainer(BaseTrainer):
#     def __init__(self):
#         super().__init__()  # 必须先初始化父类
#         print("🚀 DetectionTrainer 初始化完成")
#
#
#
# trainer = DetectionTrainer()
# from ultralytics.nn.modules.block import C3LiteShuffle
# test = C3LiteShuffle(128, 256, shortcut=False, e=0.5)
# print(test)

# from ultralytics.nn.modules import C3LiteShuffle
#
# test = C3LiteShuffle(128, 256)  # 👈 实例化触发
# print("✅ C3LiteShuffle 加载成功")

# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# import torch
# print(torch.version.cuda)
# import cv2
# img = cv2.imread('/home/jiaoyuqing/bigspace/workspaceJack/datasets/TT100K/tt100k_2021/train/00001.jpg')
# print(type(img), img.shape if img is not None else "None")
# import torch
# print(torch.__version__)
# import numpy as np
# print(np.__version__)
# print(np.arange(5))


import os
import numpy as np

def check_and_fix_npy(npy_dir, save_fixed=False, expected_dtype=np.float32, expected_shape=None):
    """
    检查并可选修复 npy 文件
    :param npy_dir: npy 文件目录
    :param save_fixed: 如果 True，会覆盖修复后的 npy
    :param expected_dtype: 期望的 numpy dtype
    :param expected_shape: 期望的 shape，tuple，None 表示不检查
    """
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    print(f"发现 {len(npy_files)} 个 .npy 文件")

    for npy_file in npy_files:
        path = os.path.join(npy_dir, npy_file)
        try:
            data = np.load(path)
            if not isinstance(data, np.ndarray):
                print(f"[错误] {npy_file} 不是 ndarray，而是 {type(data)}")
                continue

            if np.isnan(data).any():
                print(f"[警告] {npy_file} 存在 NaN")

            if np.isinf(data).any():
                print(f"[警告] {npy_file} 存在 Inf")

            if data.dtype != expected_dtype:
                print(f"[修复] {npy_file} dtype: {data.dtype} -> {expected_dtype}")
                data = data.astype(expected_dtype)
                if save_fixed:
                    np.save(path, data)

            if expected_shape and data.shape != expected_shape:
                print(f"[警告] {npy_file} shape 不符: {data.shape}，期望 {expected_shape}")

        except Exception as e:
            print(f"[加载失败] {npy_file} : {e}")

    print("✅ 检查完成")


npy_dir = '/home/jiaoyuqing/bigspace/workspaceJack/datasets/TT100K/tt100k_2021/yolo143/images/test/'
check_and_fix_npy(npy_dir, save_fixed=True, expected_dtype=np.float32, expected_shape=None)