import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange

from ultralytics.nn.modules.conv import Conv

__all__ = [
    "EnhancementModel",
    "CSSA_1",
]
# -----------------------------
# DTR / Detail feature extraction
# -----------------------------
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.theta_phi = InvertedResidualBlock(inp // 2, inp // 2, 2)
        self.theta_rho = InvertedResidualBlock(inp // 2, inp // 2, 2)
        self.theta_eta = InvertedResidualBlock(inp // 2, inp // 2, 2)
        self.shffleconv = nn.Conv2d(inp, inp, 1, 1, 0, bias=True)

    def separateFeature(self, x):
        return x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, inp, num_layers=3):
        super().__init__()
        self.net = nn.Sequential(*[DetailNode(inp) for _ in range(num_layers)])

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# -----------------------------
# PGE / PGFE module
# -----------------------------
class SCINet(nn.Module):
    def __init__(self, channels, layers=3):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            for _ in range(layers)
        ])

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        illu = fea + x
        return torch.clamp(illu, 0.0001, 1)


class ContrastEnhancement(nn.Module):
    def __init__(self, alpha=2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        return (x - mean) * self.alpha + mean


class SharpenFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def make_gaussian_2d_kernel(self, ksize=13, sigma=5, channels=3):
        if sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        center = ksize // 2
        xs = np.arange(ksize, dtype=np.float32) - center
        kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
        kernel = np.outer(kernel1d, kernel1d)
        kernel = torch.from_numpy(kernel).float().repeat(channels, 1, 1, 1)
        return kernel / kernel.sum()

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        channels = x.shape[1]

        weight = self.make_gaussian_2d_kernel(25, channels=channels).to(device=device, dtype=dtype)
        bias = torch.zeros(channels, device=device, dtype=dtype)

        output = F.conv2d(
            x,
            weight,
            bias=bias,
            padding=(weight.shape[2] // 2, weight.shape[3] // 2),
            stride=1,
            groups=channels,
        )

        diff = (x - output).clamp(min=0)
        img_out = diff * 2.5 + x
        return img_out.clamp(min=0, max=1)


class EnhancementModel(nn.Module):
    """
    PGFE module.
    YAML: [-1, 1, EnhancementModel, [64]]
    parse_model should pass EnhancementModel(c1, c2).
    """

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.contrast = ContrastEnhancement()
        self.sharpen = SharpenFilter()
        self.sci = SCINet(c1)
        self.detailFeature = DetailFeatureExtraction(inp=c1)
        self.cv = Conv(c1 * 2, c2, 1, 1)

    def forward(self, x):
        detail_feature = self.detailFeature(x)
        x_light = self.sci(x)
        x_contrast = self.contrast(x_light)
        x_sharpen = self.sharpen(x_contrast)
        fuse = torch.cat((detail_feature, x_sharpen), dim=1)
        return self.cv(fuse)


# -----------------------------
# HRFM-SOD / CSSA_1 module
# -----------------------------
class BABlock(nn.Module):
    def __init__(self, pre_channels, cur_channel, reduction=16):
        super().__init__()
        self.pre_fusions = nn.ModuleList([
            nn.Sequential(nn.Conv2d(pre_channel, cur_channel // reduction, 1, bias=False))
            for pre_channel in pre_channels
        ])

        self.cur_fusion = nn.Sequential(
            nn.Conv2d(cur_channel, cur_channel // reduction, 1, bias=False)
        )

        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(cur_channel // reduction, cur_channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        pre_fusions = [self.pre_fusions[i](pre_layers[i]) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer)
        fusion = cur_fusion + sum(pre_fusions)
        return self.generation(fusion)


class SABlock(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv4 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f1, f2, f3, f4):
        def max_avg(f):
            return torch.cat([torch.max(f, dim=1, keepdim=True)[0], torch.mean(f, dim=1, keepdim=True)], dim=1)

        f1_fusion = self.conv1(max_avg(f1))
        f2_fusion = self.conv2(max_avg(f2))
        f3_fusion = self.conv3(max_avg(f3))
        f4_fusion = self.conv4(max_avg(f4))

        return self.sigmoid(f1_fusion + f2_fusion + f3_fusion + f4_fusion)


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CSSA_1(nn.Module):
    """
    Dynamic HRFM-SOD module.
    支持 n/s/m/l/x 自动缩放后的通道。

    YAML:
        - [[4, 6, 8, 11], 1, CSSA_1, [128]]

    tasks.py 中需要：
        elif m is CSSA_1:
            c2 = args[0]
            args = [c2, [ch[x] for x in f]]
    """

    def __init__(self, c1, ch_list=None):
        super().__init__()

        if ch_list is None:
            # 兼容作者原始 l 结构
            ch_list = [c1, c1 * 2, c1 * 4, c1 * 4]

        c_f1, c_f2, c_f3, c_f4 = ch_list

        print(f"CSSA_1 init: c1={c1}, ch_list={ch_list}")

        self.f1_proj = nn.Identity() if c_f1 == c1 else nn.Conv2d(c_f1, c1, kernel_size=1)

        self.upsample_f2 = nn.Sequential(
            nn.Conv2d(c_f2, c1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.upsample_f3 = nn.Sequential(
            nn.Conv2d(c_f3, c1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )

        self.upsample_f4 = nn.Sequential(
            nn.Conv2d(c_f4, c1, kernel_size=1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        )

        self.softmax_conv = Conv(c1 * 4, c1, 1, 1)
        self.conv3x3 = Conv(c1, c1, 3, 1)

        self.ba1 = BABlock([c1, c1, c1], c1)
        self.ba2 = BABlock([c1, c1, c1], c1)
        self.ba3 = BABlock([c1, c1, c1], c1)
        self.ba4 = BABlock([c1, c1, c1], c1)

        self.ba1_2 = BABlock([c1, c1, c1], c1)
        self.ba2_2 = BABlock([c1, c1, c1], c1)
        self.ba3_2 = BABlock([c1, c1, c1], c1)
        self.ba4_2 = BABlock([c1, c1, c1], c1)

        self.sa1_2 = SABlock()
        self.sa2_2 = SABlock()
        self.sa3_2 = SABlock()
        self.sa4_2 = SABlock()

        self.feature_extraction = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f1, f2, f3, f4 = x

        # 统一到 c1 通道、f1 空间分辨率
        f1 = self.f1_proj(f1)
        f2_c = self.upsample_f2(f2)
        f3_c = self.upsample_f3(f3)
        f4_c = self.upsample_f4(f4)

        f1_ap = self.feature_extraction(f1)
        f2_ap = self.feature_extraction(f2_c)
        f3_ap = self.feature_extraction(f3_c)
        f4_ap = self.feature_extraction(f4_c)

        att_ba1 = self.ba1([f2_ap, f3_ap, f4_ap], f1_ap)
        att_ba2 = self.ba2([f1_ap, f3_ap, f4_ap], f2_ap)
        att_ba3 = self.ba3([f1_ap, f2_ap, f4_ap], f3_ap)
        att_ba4 = self.ba4([f2_ap, f3_ap, f1_ap], f4_ap)

        f1_c2 = f1 * att_ba1
        f2_c2 = f2_c * att_ba2
        f3_c2 = f3_c * att_ba3
        f4_c2 = f4_c * att_ba4

        f1_ap2 = self.feature_extraction(f1_c2)
        f2_ap2 = self.feature_extraction(f2_c2)
        f3_ap2 = self.feature_extraction(f3_c2)
        f4_ap2 = self.feature_extraction(f4_c2)

        att_ba1_2 = self.ba1_2([f2_ap2, f3_ap2, f4_ap2], f1_ap2)
        att_ba2_2 = self.ba2_2([f1_ap2, f3_ap2, f4_ap2], f2_ap2)
        att_ba3_2 = self.ba3_2([f1_ap2, f2_ap2, f4_ap2], f3_ap2)
        att_ba4_2 = self.ba4_2([f2_ap2, f3_ap2, f1_ap2], f4_ap2)

        f1_ba2 = f1 * att_ba1_2
        f2_ba2 = f2_c * att_ba2_2
        f3_ba2 = f3_c * att_ba3_2
        f4_ba2 = f4_c * att_ba4_2

        att_sa1_2 = self.sa1_2(f1_ba2, f2_ba2, f3_ba2, f4_ba2)
        att_sa2_2 = self.sa2_2(f1_ba2, f2_ba2, f3_ba2, f4_ba2)
        att_sa3_2 = self.sa3_2(f1_ba2, f2_ba2, f3_ba2, f4_ba2)
        att_sa4_2 = self.sa4_2(f1_ba2, f2_ba2, f3_ba2, f4_ba2)

        result1 = f1_c2 * att_sa1_2
        result2 = f2_c2 * att_sa2_2
        result3 = f3_c2 * att_sa3_2
        result4 = f4_c2 * att_sa4_2

        concat = torch.cat((result1, result2, result3, result4), dim=1)
        concat = self.softmax_conv(concat)
        result = self.conv3x3(concat)

        return result