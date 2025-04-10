import torch
from ultralytics.nn.modules.conv import Conv, GhostConv
from ultralytics.nn.modules.block import C3k2Lite, GhostSPPF, ConvECA, C2ECA

# 模拟输入：图像尺寸 (1, 3, 640, 640)
x = torch.randn(1, 3, 640, 640)
print(f"输入: {x.shape}")

# 每一层依次执行
x = GhostConv(3, 64, k=3, s=2)(x)
print(f"0-GhostConv(3→64, 3, 2): {x.shape}")

x = Conv(64, 128, k=3, s=2)(x)
print(f"1-Conv(64→128, 3, 2): {x.shape}")

x = C3k2Lite(128, 256, n=2, shortcut=False, e=0.25)(x)
print(f"2-C3k2Lite(128→256): {x.shape}")

x = Conv(256, 256, k=3, s=2)(x)
print(f"3-Conv(256→256, 3, 2): {x.shape}")

x = C3k2Lite(256, 512, n=2, shortcut=False, e=0.25)(x)
print(f"4-C3k2Lite(256→512): {x.shape}")

x = Conv(512, 512, k=3, s=2)(x)
print(f"5-Conv(512→512, 3, 2): {x.shape}")

x = C3k2Lite(512, 512, n=2, shortcut=True)(x)
print(f"6-C3k2Lite(512→512): {x.shape}")

x = Conv(512, 1024, k=3, s=2)(x)
print(f"7-Conv(512→1024, 3, 2): {x.shape}")

x = C3k2Lite(1024, 1024, n=2, shortcut=True)(x)
print(f"8-C3k2Lite(1024→1024): {x.shape}")

x = GhostSPPF(1024, 1024, k=5)(x)
print(f"9-GhostSPPF(1024): {x.shape}")

x = C2ECA(1024, 1024)(x)
print(f"10-C2ECA(1024→1024): {x.shape}")


