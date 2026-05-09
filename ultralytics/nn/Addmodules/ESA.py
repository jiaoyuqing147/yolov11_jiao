import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.nn.modules.head import Detect
__all__ = [
    'ConvolutionalGLU',
    'SpatialOperation',
    'ChannelOperation',
    'LocalIntegration',
    'AdditiveTokenMixer',
    'DropPath',
    'CAGM',
    'C3k_CAGB',
    'CAGB',
    'ASP',
    'Fusion',
    'ASPH',
]

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=hidden_features
            ),
            act_layer()
        )

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x

        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return shortcut + x


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.block(x)


class LocalIntegration(nn.Module):
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()

        mid_dim = round(ratio * dim)

        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=1, stride=1, padding=0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()

        self.qkv = nn.Conv2d(
            dim,
            3 * dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=attn_bias
        )

        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim)
        )

        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim)
        )

        self.dwc = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim
        )

        self.proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim
        )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = self.oper_q(q)
        k = self.oper_k(k)

        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)

        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
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
    output = x.div(keep_prob) * random_tensor

    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class CAGM(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 attn_bias=False,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.local_perception = LocalIntegration(
            dim,
            ratio=1,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        self.norm1 = norm_layer(dim)

        self.attn = AdditiveTokenMixer(
            dim,
            attn_bias=attn_bias,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        self.mlp = ConvolutionalGLU(dim)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class C3k_CAGB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)

        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(CAGM(c_) for _ in range(n))
        )

class CAGB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)

        self.m = nn.ModuleList(
            C3k_CAGB(self.c, self.c, 2, shortcut, g)
            if c3k
            else CAGM(self.c)
            for _ in range(n)
        )

class ASP(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()

        self.groups = factor

        assert channels // self.groups > 0, \
            f"channels={channels} must be larger than groups={self.groups}"

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(
            channels // self.groups,
            channels // self.groups
        )

        self.conv1x1 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                channels // self.groups,
                channels // self.groups,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1)
            ),
            nn.Conv2d(
                channels // self.groups,
                channels // self.groups,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0)
            )
        )

    def forward(self, x):
        b, c, h, w = x.size()

        group_x = x.reshape(
            b * self.groups,
            -1,
            h,
            w
        )

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))

        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(
            group_x
            * x_h.sigmoid()
            * x_w.permute(0, 1, 3, 2).sigmoid()
        )

        x2 = self.conv3x3(group_x)

        x11 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )

        x12 = x2.reshape(
            b * self.groups,
            c // self.groups,
            -1
        )

        x21 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )

        x22 = x1.reshape(
            b * self.groups,
            c // self.groups,
            -1
        )

        weights = (
            torch.matmul(x11, x12)
            + torch.matmul(x21, x22)
        ).reshape(
            b * self.groups,
            1,
            h,
            w
        )

        out = (
            group_x * weights.sigmoid()
        ).reshape(
            b,
            c,
            h,
            w
        )

        return out

class Fusion(nn.Module):
    """
    Feature fusion module for ESA-YOLO.

    Supports:
    - weighted_fusion: learnable weighted sum
    - concat: channel concatenation
    - add: direct summation
    """

    def __init__(self, mode="weighted_fusion"):
        super().__init__()
        self.mode = mode

        if self.mode == "weighted_fusion":
            self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.eps = 1e-4

    def forward(self, x):
        """
        x: list of feature maps, usually [x1, x2]
        """
        if not isinstance(x, (list, tuple)):
            return x

        if self.mode == "weighted_fusion":
            assert len(x) == 2, f"Fusion weighted_fusion expects 2 inputs, but got {len(x)}"
            w = torch.relu(self.w)
            w = w / (torch.sum(w) + self.eps)
            return w[0] * x[0] + w[1] * x[1]

        elif self.mode == "add":
            out = x[0]
            for i in range(1, len(x)):
                out = out + x[i]
            return out

        elif self.mode == "concat":
            return torch.cat(x, dim=1)

        else:
            raise ValueError(f"Unsupported fusion mode: {self.mode}")

class ASPH(Detect):
    """
    ESA-YOLO detection head wrapper.

    为了适配你当前 YOLO11 框架，这里继承官方 Detect。
    yaml 中可以写：
    - [[35, 36, 37], 1, ASPH, [nc]]
    """

    def __init__(self, nc=80, ch=()):
        super().__init__(nc=nc, ch=ch)