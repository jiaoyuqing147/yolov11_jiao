import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv  # 与C2PSA保持一致的Conv封装

__all__ = ['C2CASAB']
#这玩意改编自论文，但是是jack改编的
# ---- 通道注意力 ----
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x) + self.max_pool(x))

# ---- 空间注意力 ----
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.max(x, dim=1, keepdim=True)[0]
        minv = torch.min(x, dim=1, keepdim=True)[0]
        sumv = torch.sum(x, dim=1, keepdim=True)
        sa = torch.cat([mean, maxv, minv, sumv], dim=1)
        return x * self.sigmoid(self.conv(sa))

# ---- C2风格的CASAB模块 ----
class C2CASAB(nn.Module):
    """
    C2CASAB: C2-style Channel + Spatial Attention Block
    结构与C2PSA一致，但内部是轻量的通道+空间注意力。
    """
    def __init__(self, c1, c2, n=1, e=0.5, reduction=16):
        """
        Args:
            c1, c2: 输入/输出通道数（保持相等）
            n: 可堆叠层数（同C2PSA接口）
            e: expansion ratio（默认0.5）
            reduction: 通道注意力压缩率
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.blocks = nn.Sequential(*(self._make_block() for _ in range(n)))

    def _make_block(self):
        return nn.Sequential(
            ChannelAttention(self.c),
            SpatialAttention()
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.blocks(b)
        return self.cv2(torch.cat((a, b), dim=1))
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = C2CASAB(64, 64)

    out = mobilenet_v1(image)
    print(out.size())