#ETSR-yolo论文的代码复现
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck

__all__ = [
    "C3CA",
    "CNeB",
]

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h


class C3CA(nn.Module):
    """
    C3 + Coordinate Attention.
    用于替代 YOLOv5 原生 C3。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

        self.coord_att = CoordAtt(c2, c2)

    def forward(self, x):
        y = torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)
        y = self.cv3(y)
        return self.coord_att(y)


class LayerNorm_s(nn.Module):
    """
    ConvNeXt 使用的 LayerNorm。
    支持 channels_last 和 channels_first。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps
            )

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """
    Stochastic Depth.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device
    )
    random_tensor.floor_()

    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """
    Drop paths per sample.
    """

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class ConvNextBlock(nn.Module):
    """
    ConvNeXt Block.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.norm = LayerNorm_s(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim)),
                requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)

        return shortcut + self.drop_path(x)


class CNeB(nn.Module):
    """
    C3-style Neck block with ConvNeXtBlock.
    用于替代 Neck 中的 C3。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()

        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        self.m = nn.Sequential(
            *(ConvNextBlock(c_) for _ in range(n))
        )

    def forward(self, x):
        y = torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)
        return self.cv3(y)