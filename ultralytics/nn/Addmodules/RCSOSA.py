import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

__all__ = ['C3k2_RepVGG', 'RCSOSA','RCSOSA_Lite','RCSOSA_Lite_SmallObj']


# build RepVGG block
# -----------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class SEBlock(nn.Module):
    def __init__(self, input_channels):
        super(SEBlock, self).__init__()
        internal_neurons = input_channels // 8
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGG(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGG, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.SiLU()
        # self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def fusevggforward(self, x):
        return self.nonlinearity(self.rbr_dense(x))


# RepVGG block end
# -----------------------------

class SR(nn.Module):
    # Shuffle RepVGG
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = int(c1 // 2)
        c2_ = int(c2 // 2)
        self.repconv = RepVGG(c1_, c2_)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.repconv(x2)), dim=1)
        out = self.channel_shuffle(out, 2)
        return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class RCSOSA(nn.Module):
    # VoVNet with Res Shuffle RepVGG
    def __init__(self, c1, c2, n=1, se=False, e=0.5, stackrep=True):
        super().__init__()
        n_ = n // 2
        c_ = make_divisible(int(c1 * e), 8)
        # self.conv1 = Conv(c1, c_)
        self.conv1 = RepVGG(c1, c_)
        self.conv3 = RepVGG(int(c_ * 3), c2)
        self.sr1 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])

        self.se = None
        if se:
            self.se = SEBlock(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        x = torch.cat((x1, x2, x3), 1)
        return self.conv3(x) if self.se is None else self.se(self.conv3(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RepVGG(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_RepVGG(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )



'''
考虑到为了写论文，jack添加了下方改动RCSOSA的代码
'''
class SRLite(nn.Module):
    # 轻量级 SR 模块：DWConv + shuffle（无 RepVGG）
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = c1 // 2
        c2_ = c2 // 2
        self.conv = Conv(c1_, c2_, k=3, s=1, p=1, g=c1_)  # DWConv

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.conv(x2)), dim=1)
        return self.channel_shuffle(out, 2)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        cpg = c // groups
        x = x.view(b, groups, cpg, h, w).transpose(1, 2).contiguous()
        return x.view(b, -1, h, w)

class RCSOSA_Lite(nn.Module):
    def __init__(self, c1, c2, n=1, se=False, e=0.5):
        super().__init__()
        c_ = make_divisible(int(c1 * e), 8)
        n_ = max(n // 2, 1)

        self.conv1 = Conv(c1, c_)  # 替代 RepVGG

        self.sr1 = nn.Sequential(*[SRLite(c_, c_) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SRLite(c_, c_) for _ in range(n_)])

        self.concat_proj = Conv(c_ * 3, c2, k=1)

        self.se = SEBlock(c2) if se else nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        x_cat = torch.cat((x1, x2, x3), dim=1)
        return self.se(self.concat_proj(x_cat))



class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        return x

class SRLite_SmallObj(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = c1 // 2
        c2_ = c2 // 2
        # 使用带 dilation 的 DWConv（扩大感受野）
        self.conv = nn.Conv2d(c1_, c2_, kernel_size=3, stride=1, padding=2, dilation=2, groups=c1_, bias=False)
        self.bn = nn.BatchNorm2d(c2_)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.act(self.bn(self.conv(x2)))), dim=1)
        return self.channel_shuffle(out, 2)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        g = groups
        cpg = c // g
        x = x.view(b, g, cpg, h, w).transpose(1, 2).contiguous()
        return x.view(b, -1, h, w)

class RCSOSA_Lite_SmallObj(nn.Module):
    def __init__(self, c1, c2, n=1, se=False, e=0.5):
        super().__init__()
        c_ = max(8, int(c1 * e))
        n_ = max(1, n // 2)

        # 更强浅层表达（放大通道）
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c_ * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_ * 2),
            nn.SiLU()
        )

        self.sr1 = nn.Sequential(*[SRLite_SmallObj(c_ * 2, c_ * 2) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SRLite_SmallObj(c_ * 2, c_ * 2) for _ in range(n_)])

        # 拼接后通道为 3*c_，输出压缩至 c2
        self.concat_proj = nn.Sequential(
            nn.Conv2d(c_ * 6, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 可选注意力
        self.att = CBAM(c2) if se else nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        x_cat = torch.cat((x1, x2, x3), dim=1)
        out = self.concat_proj(x_cat)
        return self.att(out)


if __name__ == "__main__":
    # Generating Sample image
    # image_size = (1, 64, 240, 240)
    # image = torch.rand(*image_size)
    #
    # # Model
    # mobilenet_v1 = C3k2_RepVGG(64, 64)
    #
    # out = mobilenet_v1(image)
    # print(out.size())
    # 生成一个模拟输入（如来自某一层的特征图）
    dummy_input = torch.randn(1, 64, 80, 80)  # (B, C, H, W)

    print(">>> Testing RCSOSA_Lite")
    model1 = RCSOSA_Lite(c1=64, c2=64, n=2, se=True, e=0.5)
    out1 = model1(dummy_input)
    print("Output shape:", out1.shape)
    print()

    print(">>> Testing RCSOSA_Lite_SmallObj")
    model2 = RCSOSA_Lite_SmallObj(c1=64, c2=64, n=2, se=True, e=0.5)
    out2 = model2(dummy_input)
    print("Output shape:", out2.shape)
    print()

    print(">>> Testing SRLite")
    sr = SRLite(64, 64)
    out3 = sr(dummy_input)
    print("SRLite output shape:", out3.shape)
    print()

    print(">>> Testing SRLite_SmallObj")
    sr_small = SRLite_SmallObj(64, 64)
    out4 = sr_small(dummy_input)
    print("SRLite_SmallObj output shape:", out4.shape)
    print()

    print(">>> Testing CBAM module")
    cbam = CBAM(64)
    out5 = cbam(dummy_input)
    print("CBAM output shape:", out5.shape)
    print()