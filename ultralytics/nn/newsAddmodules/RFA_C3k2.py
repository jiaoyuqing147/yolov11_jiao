import torch
import torch.nn as nn

from ultralytics.nn.modules import C3k2, Conv

__all__ = ["RFA_C3k2"]


class RFABlock(nn.Module):
    """Receptive-field attention block with multi-kernel depthwise branches."""

    def __init__(self, c, shortcut=True):
        super().__init__()
        self.branches = nn.ModuleList([Conv(c, c, 3, 1, g=c), Conv(c, c, 5, 1, g=c), Conv(c, c, 7, 1, g=c)])
        hidden = max(c // 4, 8)
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
            nn.Softmax(dim=1),
        )
        self.proj = Conv(c, c, 1, 1)
        self.add = shortcut

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        weights = self.weight(torch.cat(feats, dim=1)).unsqueeze(2)
        y = (torch.stack(feats, dim=1) * weights).sum(dim=1)
        y = self.proj(y)
        return x + y if self.add else y


class RFA_C3k2(C3k2):
    """C3k2 with receptive-field attention aggregation."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(RFABlock(self.c, shortcut) for _ in range(n))
