import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv
from ultralytics.utils.torch_utils import profile

__all__ = ["DSUB"]


class DSUB(nn.Module):
    """
    DSUB: Depth-to-Space Upsampling Block

    功能：
        使用 Conv + PixelShuffle 实现无插值上采样（Depth-to-Space），
        可以替代 nn.Upsample 或 ConvTranspose 用在 FPN / PAN 的上采样位置。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数（上采样后特征的通道）
        scale (int): 上采样倍数，默认 2

    形状：
        输入:  [B, c1, H, W]
        输出:  [B, c2, H * scale, W * scale]
    """

    def __init__(self, c1: int, c2: int, scale: int = 2):
        super().__init__()
        assert scale >= 1, "scale must be >= 1"
        self.scale = scale
        # 先卷积到 c2 * (scale^2) 通道，再像素重排到空间维度
        self.conv = Conv(c1, c2 * (scale ** 2), k=3, s=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.shuffle(x)  # Depth-to-Space
        return x

    def profile(self, x: torch.Tensor):
        """
        配合 ultralytics.utils.torch_utils.profile 使用时，返回 (y, flops)。
        这里只估算主 Conv 的 FLOPs，作为参考。
        """
        y = self.forward(x)
        b, c_in, h, w = x.shape
        c_out = self.conv.conv.out_channels
        k_h = k_w = self.conv.conv.kernel_size[0]
        flops = b * c_out * h * w * (c_in * k_h * k_w)
        return y, flops


if __name__ == "__main__":
    # 简单自测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 假设来自 P5 特征：通道64，尺寸32x32
    x = torch.randn(1, 64, 32, 32).to(device)

    m = DSUB(64, 64, scale=2).to(device)
    print(">>> Profiling DSUB")
    profile(x, m)

    y = m(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)  # 期望 [1, 64, 64, 64]
