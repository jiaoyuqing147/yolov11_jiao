import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C3k2, Conv

__all__ = ["C3k2_MSEIE"]


class MSEIEBlock(nn.Module):
    """Multi-scale enhanced information extraction block."""

    def __init__(self, c, shortcut=True):
        super().__init__()
        self.dw3 = Conv(c, c, 3, 1, g=c)
        self.dw5 = Conv(c, c, 5, 1, g=c)
        self.dw7 = Conv(c, c, 7, 1, g=c)
        self.edge = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.fuse = Conv(c * 4, c, 1, 1)
        hidden = max(c // 8, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid(),
        )
        self.add = shortcut
        with torch.no_grad():
            kernel = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
            self.edge.weight.copy_(kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1))

    def forward(self, x):
        edge = F.silu(self.edge(x))
        y = self.fuse(torch.cat((self.dw3(x), self.dw5(x), self.dw7(x), edge), dim=1))
        y = y * self.attn(y)
        return x + y if self.add else y


class C3k2_MSEIE(C3k2):
    """C3k2-compatible MSEIE approximation."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(MSEIEBlock(self.c, shortcut) for _ in range(n))
