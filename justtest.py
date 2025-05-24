
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

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
import torch
print(torch.version.cuda)
