import torch
import torch.nn as nn

__all__ = ["OSA", "OSA_eSE", "eSEModule"]


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution: Conv2d + BN + SiLU."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class eSEModule(nn.Module):
    """Effective Squeeze-Excitation."""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.hsigmoid = HSigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        w = self.hsigmoid(w)
        return x * w


class _OSA_Base(nn.Module):
    """
    Base OSA block for Ultralytics YOLO.

    Args:
        c1 (int): input channels
        c2 (int): output channels
        n (int): number of internal 3x3 conv layers
        shortcut (bool): residual connection when c1 == c2
        e (float): expansion ratio for hidden channels
        use_ese (bool): whether to enable eSE attention
    """

    def __init__(self, c1, c2, n=3, shortcut=True, e=0.5, use_ese=False):
        super().__init__()
        assert n >= 1, "n must be >= 1"

        c_ = int(c2 * e)
        self.layers = nn.ModuleList()

        in_ch = c1
        for _ in range(n):
            self.layers.append(Conv(in_ch, c_, k=3, s=1))
            in_ch = c_

        # concat: original input + outputs of n conv layers
        self.concat_conv = Conv(c1 + n * c_, c2, k=1, s=1)
        self.attn = eSEModule(c2) if use_ese else nn.Identity()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        outputs = [x]

        y = x
        for layer in self.layers:
            y = layer(y)
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        out = self.concat_conv(out)
        out = self.attn(out)

        if self.add:
            out = out + identity
        return out


class OSA(_OSA_Base):
    """OSA block without eSE."""
    def __init__(self, c1, c2, n=3, shortcut=True, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, e=e, use_ese=False)


class OSA_eSE(_OSA_Base):
    """OSA block with eSE."""
    def __init__(self, c1, c2, n=3, shortcut=True, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, e=e, use_ese=True)


if __name__ == "__main__":
    x1 = torch.randn(1, 128, 80, 80)
    m1 = OSA(128, 128, n=3, shortcut=True, e=0.5)
    y1 = m1(x1)
    print("OSA:", x1.shape, "->", y1.shape)

    x2 = torch.randn(1, 384, 40, 40)
    m2 = OSA_eSE(384, 256, n=3, shortcut=False, e=0.5)
    y2 = m2(x2)
    print("OSA_eSE:", x2.shape, "->", y2.shape)