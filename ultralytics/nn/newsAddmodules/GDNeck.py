import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

__all__ = ["GatherBlock", "DistributeBlock", "GDNeck"]


class GatherBlock(nn.Module):
    """Gather same-resolution features with learnable normalized weights."""

    def __init__(self, c, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.proj = nn.ModuleList(nn.Conv2d(c, c, 1, bias=False) for _ in range(length))
        self.eps = 1e-4

    def forward(self, x):
        weight = torch.relu(self.weight)
        weight = weight / (weight.sum() + self.eps)
        return sum(weight[i] * self.proj[i](x[i]) for i in range(len(x)))


class DistributeBlock(nn.Module):
    """Distribute gathered context back to the current fusion level."""

    def __init__(self, c):
        super().__init__()
        self.context = Conv(c, c, 3, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        context = self.context(x)
        return x + context * self.gate(context)


class GDNeck(nn.Module):
    """Simplified Gold-YOLO-style gather-and-distribute fusion node."""

    def __init__(self, c, length):
        super().__init__()
        self.gather = GatherBlock(c, length)
        self.distribute = DistributeBlock(c)

    def forward(self, x):
        return self.distribute(self.gather(x))
