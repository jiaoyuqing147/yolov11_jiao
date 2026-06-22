import torch.nn as nn

from ultralytics.nn.modules import C3k2, Conv

__all__ = ["C2f_DM"]


class DMixerBlock(nn.Module):
    """Depthwise spatial mixer plus channel mixer used as a D-Mixer approximation."""

    def __init__(self, c, shortcut=True, expansion=2.0):
        super().__init__()
        hidden = int(c * expansion)
        self.spatial_mixer = nn.Sequential(Conv(c, c, (1, 3), 1, g=c), Conv(c, c, (3, 1), 1, g=c))
        self.channel_mixer = nn.Sequential(Conv(c, hidden, 1, 1), nn.Conv2d(hidden, c, 1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut

    def forward(self, x):
        y = self.channel_mixer(self.spatial_mixer(x))
        return self.act(x + y) if self.add else self.act(y)


class C2f_DM(C3k2):
    """C3k2-compatible C2f-DM / C2f-D-Mixer approximation."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(DMixerBlock(self.c, shortcut) for _ in range(n))
