import torch
import torch.nn as nn

__all__ = ['OSA_JACK']



def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BN and SiLU."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class OSA_JACK(nn.Module):
    """
    Standard OSA block:
    input -> reduce -> stacked convs -> one-shot concat -> fuse
    """
    def __init__(self, c1, c2, n=3, e=0.5, shortcut=True, se=False):
        super().__init__()
        assert n >= 1, "n must be >= 1"

        c_ = int(c2 * e)
        c_ = max(c_, 8)

        self.reduce = Conv(c1, c_, 1, 1)
        self.blocks = nn.ModuleList([Conv(c_, c_, 3, 1) for _ in range(n)])
        self.fuse = Conv((n + 1) * c_, c2, 1, 1)

        self.use_shortcut = shortcut and c1 == c2
        self.se = SEBlock(c2) if se else nn.Identity()

    def forward(self, x):
        y = []

        x0 = self.reduce(x)
        y.append(x0)

        xi = x0
        for block in self.blocks:
            xi = block(xi)
            y.append(xi)

        out = self.fuse(torch.cat(y, dim=1))
        out = self.se(out)

        return x + out if self.use_shortcut else out


class OSA_Lite(nn.Module):
    """
    A lighter OSA variant for fair comparison with lightweight YOLO blocks.
    """
    def __init__(self, c1, c2, n=2, e=0.5, shortcut=True, se=False):
        super().__init__()
        assert n >= 1, "n must be >= 1"

        c_ = int(c2 * e)
        c_ = max(c_, 8)

        self.reduce = Conv(c1, c_, 1, 1)
        self.blocks = nn.ModuleList([Conv(c_, c_, 3, 1, g=1) for _ in range(n)])
        self.fuse = Conv((n + 1) * c_, c2, 1, 1)

        self.use_shortcut = shortcut and c1 == c2
        self.se = SEBlock(c2) if se else nn.Identity()

    def forward(self, x):
        feats = []

        x1 = self.reduce(x)
        feats.append(x1)

        x2 = x1
        for block in self.blocks:
            x2 = block(x2)
            feats.append(x2)

        out = self.fuse(torch.cat(feats, dim=1))
        out = self.se(out)

        return x + out if self.use_shortcut else out


if __name__ == "__main__":
    # 生成一个模拟输入（如来自某一层的特征图）
    dummy_input = torch.randn(1, 64, 80, 80)  # (B, C, H, W)

    print(">>> Testing OSA")
    model1 = OSA_JACK(c1=64, c2=64, n=3, e=0.5, shortcut=True, se=True)
    out1 = model1(dummy_input)
    print("OSA output shape:", out1.shape)
    print()

    print(">>> Testing OSA_Lite")
    model2 = OSA_Lite(c1=64, c2=64, n=2, e=0.5, shortcut=True, se=True)
    out2 = model2(dummy_input)
    print("OSA_Lite output shape:", out2.shape)
    print()

    print(">>> Testing OSA without shortcut")
    model3 = OSA_JACK(c1=64, c2=128, n=3, e=0.5, shortcut=False, se=False)
    out3 = model3(dummy_input)
    print("OSA(no shortcut) output shape:", out3.shape)
    print()