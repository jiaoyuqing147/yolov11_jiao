import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv  # YOLO 原生卷积封装（Conv+BN+SiLU）

__all__ = ['C2CASAB_heavey']


# ========= 1️⃣ 深度可分离卷积块（与第一份 CASAB 一致） =========
class ConvBlock(nn.Module):
    """
    Depthwise Conv + Pointwise Conv + BN + LeakyReLU
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


# ========= 2️⃣ 通道注意力（与第一份 CASAB 一致） =========
class ChannelAttention(nn.Module):
    """
    GAP + GMP → shared MLP → Sigmoid → 通道权重
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = avg_out + max_out
        return x * scale


# ========= 3️⃣ 空间注意力（与第一份 CASAB 一致：mean/max/min/sum 四种统计） =========
class SpatialAttention(nn.Module):
    """
    mean / max / min / sum → concat(4C) → 7x7 conv → 1x1 conv → Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.max(x, dim=1, keepdim=True)[0]
        minv = torch.min(x, dim=1, keepdim=True)[0]
        sumv = torch.sum(x, dim=1, keepdim=True)
        sa = torch.cat([mean, maxv, minv, sumv], dim=1)
        weight = self.conv(sa)
        return x * weight


# ========= 4️⃣ C2CASAB：外壳保持 C2PSA 接口，内部用“原版 CASAB 风格” =========
class C2CASAB_heavey(nn.Module):
    """
    C2CASAB with original CASAB-style inner structure.

    外部结构：
        x → Conv(1x1) → split(a,b) → CASAB(b) → concat(a, b_out) → Conv(1x1)

    内部 CASAB：
        b → ConvBlock → x1
        ca = ChannelAttention(x1)
        sa = SpatialAttention(x1)
        b_out = ca + sa
    """
    def __init__(self, c1, c2, c3=None, n=1, e=0.5, reduction=16):
        """
        注意：
            - c1, c2 由 YOLO 框架自动传入（输入/输出通道）
            - c3 用来“吃掉” YAML 里多传的那个参数（例如 [1024]），这里不会作为 n 使用
            - n 为 CASAB 堆叠次数，默认 1；为安全起见，如果传入特别大的值，会被限制
        """
        super().__init__()
        assert c1 == c2, "C2CASAB: 输入输出通道必须相等 (c1 == c2)"
        self.c = int(c1 * e)

        # 安全限制：避免 n 被误传成很大（例如 256），导致网络过深、显存爆炸
        if isinstance(n, int) and n > 3:
            n = 1

        # C2 风格的前后 1x1 卷积，保持与 C2PSA、C2f 等模块接口一致
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = Conv(2 * self.c, c1, k=1, s=1)

        # 堆叠 n 个 CASAB block（每个 block：ConvBlock → CA → SA → ca+sa）
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "conv": ConvBlock(self.c, self.c),
                "ca": ChannelAttention(self.c, reduction),
                "sa": SpatialAttention()
            })
            for _ in range(n)
        ])

    def forward(self, x):
        # C2 风格：先把通道分成两半，一半走注意力，一半走“旁路”
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 逐层堆叠原版 CASAB 结构
        for blk in self.blocks:
            x1 = blk["conv"](b)     # ConvBlock 提特征
            ca = blk["ca"](x1)     # ChannelAttention(x1)
            sa = blk["sa"](x1)     # SpatialAttention(x1)
            b = ca + sa            # ★ 原文 CASAB 风格：ca + sa（不加残差）

        out = self.cv2(torch.cat((a, b), dim=1))
        return out


# ========= 5️⃣ 简单自测 =========
if __name__ == "__main__":
    x = torch.rand(1, 64, 32, 32)
    m = C2CASAB_heavey(64, 64, n=1, e=0.5, reduction=16)
    y = m(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
