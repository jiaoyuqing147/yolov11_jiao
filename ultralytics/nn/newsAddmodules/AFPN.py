import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

__all__ = ["AFPN"]


class AFPN(nn.Module):
    """Simplified AFPN-style fusion node for same-resolution P3/P4/P5 features."""

    def __init__(self, c, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.align = nn.ModuleList(nn.Conv2d(c, c, 1, bias=False) for _ in range(length))
        self.fuse = Conv(c, c, 3, 1)
        self.eps = 1e-4

    def forward(self, x):
        weight = torch.relu(self.weight)
        weight = weight / (weight.sum() + self.eps)
        y = sum(weight[i] * self.align[i](x[i]) for i in range(len(x)))
        return self.fuse(y)
