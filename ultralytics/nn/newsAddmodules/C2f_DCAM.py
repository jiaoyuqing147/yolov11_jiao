import torch
import torch.nn as nn

from ultralytics.nn.modules import C3k2, Conv

__all__ = ["C2f_DCAM"]


class DCAMBlock(nn.Module):
    """Dual context attention block for C2f/C3k2-style feature aggregation."""

    def __init__(self, c, shortcut=True, reduction=16):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.local = nn.Sequential(Conv(c, c, 3, 1, g=c), Conv(c, c, 1, 1))
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.proj = Conv(c, c, 1, 1)
        self.add = shortcut

    def forward(self, x):
        y = self.local(x)
        avg = torch.mean(y, dim=1, keepdim=True)
        mx = torch.amax(y, dim=1, keepdim=True)
        y = self.proj(y * self.channel(y) * self.spatial(torch.cat((avg, mx), dim=1)))
        return x + y if self.add else y


class C2f_DCAM(C3k2):
    """C3k2-compatible C2f-DCAM approximation."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(DCAMBlock(self.c, shortcut) for _ in range(n))
