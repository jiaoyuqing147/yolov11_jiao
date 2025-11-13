import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv
from ultralytics.utils.torch_utils import profile

__all__ = ["RLAB"]


class RLABBlock(nn.Module):
    """
    单个 RLAB 子模块：
        Conv1x1 -> Conv3x3 -> 通道注意力(SE风格) -> 残差

    Args:
        c (int): 通道数
        e (float): 中间通道比例 (expand ratio)
        reduction (int): 通道注意力的压缩比例
    """

    def __init__(self, c: int, e: float = 0.5, reduction: int = 16):
        super().__init__()
        c_mid = max(1, int(c * e))

        self.cv1 = Conv(c, c_mid, k=1, s=1)
        self.cv2 = Conv(c_mid, c, k=3, s=1)

        # 通道注意力 (GAP + GMP + 1x1 Conv)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(1, c // reduction), 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(max(1, c // reduction), c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))               # 基本卷积路径
        w = self.fc(self.avg(y) + self.max(y))  # 通道注意力
        y = y * w
        return y + x                            # 残差连接


class RLAB(nn.Module):
    """
    RLAB: Residual Lightweight Attention Block

    用途：
        用于 FPN / PAN 中的特征融合层，替代 C3k2 / C2fAttn / C2PSA 这一类模块。
        一般放在 Concat 之后，用来融合上采样特征与 backbone skip 特征。

    Args:
        c1 (int): 输入通道数（通常是 concat 之后的通道数）
        c2 (int): 输出通道数
        n  (int): 内部 RLABBlock 的重复次数
        e  (float): 每个子块的中间通道比例
        reduction (int): 通道注意力压缩比例

    形状：
        输入:  [B, c1, H, W]
        输出:  [B, c2, H, W]
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, reduction: int = 16):
        super().__init__()
        # 先把 concat 后的通道映射到 c2
        self.cv_proj = Conv(c1, c2, k=1, s=1)
        self.blocks = nn.Sequential(
            *[RLABBlock(c2, e=e, reduction=reduction) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv_proj(x)
        x = self.blocks(x)
        return x

    def profile(self, x: torch.Tensor):
        """
        配合 ultralytics.utils.torch_utils.profile 使用时，返回 (y, flops)。
        这里只给一个大致估算，方便比较相对复杂度。
        """
        y = self.forward(x)
        b, c_in, h, w = x.shape
        c_out = y.shape[1]
        k_h = k_w = 3
        flops = b * c_out * h * w * (c_in * k_h * k_w)
        return y, flops


if __name__ == "__main__":
    # 简单自测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 模拟 concat 后的特征：通道128，尺寸 40x40
    x = torch.randn(1, 128, 40, 40).to(device)

    m = RLAB(128, 64, n=1).to(device)
    print(">>> Profiling RLAB")
    profile(x, m)

    y = m(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)  # 期望 [1, 64, 40, 40]
