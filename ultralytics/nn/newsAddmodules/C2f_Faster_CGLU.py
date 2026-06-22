import torch
import torch.nn as nn

from ultralytics.nn.modules import C3k2, Conv

__all__ = ["C2f_Faster_CGLU"]


class PartialConv3(nn.Module):
    def __init__(self, c, n_div=4):
        super().__init__()
        self.dim_conv = max(c // n_div, 1)
        self.dim_pass = c - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_pass], dim=1)
        return torch.cat((self.conv(x1), x2), dim=1)


class ConvolutionalGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        hidden = max(int(2 * c / 3), 1)
        self.fc1 = nn.Conv2d(c, hidden * 2, 1)
        self.dwconv = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=True), nn.GELU())
        self.fc2 = nn.Conv2d(hidden, c, 1)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, dim=1)
        return self.fc2(self.dwconv(y) * gate)


class FasterCGLUBlock(nn.Module):
    """FasterNet partial convolution block with convolutional GLU."""

    def __init__(self, c, shortcut=True):
        super().__init__()
        self.spatial_mixing = PartialConv3(c)
        self.mlp = ConvolutionalGLU(c)
        self.norm = Conv(c, c, 1, 1)
        self.add = shortcut

    def forward(self, x):
        y = self.norm(self.mlp(self.spatial_mixing(x)))
        return x + y if self.add else y


class C2f_Faster_CGLU(C3k2):
    """C3k2-compatible CSW-YOLO C2f-Faster-CGLU block."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(FasterCGLUBlock(self.c, shortcut) for _ in range(n))
