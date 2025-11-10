
import torch.nn as nn

__all__ = ['ARConv']


#-------------必备的组件---------------------#
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

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
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

#-------------必备的组件---------------------#

import torch
import torch.nn as nn


class ARConv(nn.Module):
    """
    YOLO 兼容版 Adaptive Rectangular Convolution

    接口与 ultralytics.nn.modules.conv.Conv 前几个参数对齐：
    __init__(c1, c2, k=3, s=1, p=None, g=1, act=True, ...)

    现在约定：
    - 只支持 stride=1，用作“3×3 卷积的自适应替代块”，不负责下采样；
    - 下采样一律交给外面的 Conv(s=2) 等模块。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,              # 占位，保持接口一致
        hw_range=(1, 18),
        warmup_iters: int = 100,
        modulation: bool = True,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # 1. stride 逻辑统一：只允许 s=1，强制写死
        # -------------------------------------------------------------
        if s != 1:
            print(f"[ARConv] Warning: stride s={s} is not supported, forcing to 1.")
        s = 1

        if p is None:
            p = k // 2  # 与 Conv 保持一致

        self.inc = c1
        self.outc = c2
        self.stride = 1          # 采样网格用的 stride，固定为 1
        self.padding = p
        self.modulation = modulation

        assert isinstance(hw_range, (list, tuple)) and len(hw_range) == 2
        self.hw_range = hw_range
        self.hw_max = hw_range[1]
        self.hw_min = hw_range[0]

        self.warmup_iters = warmup_iters
        self._iter = 0

        # -------------------------------------------------------------
        # 2. 预定义矩形核大小列表：(3,3),(3,5),(5,3)...(7,7)
        # -------------------------------------------------------------
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]

        # 按 padding 做 ZeroPad
        self.zero_padding = nn.ZeroPad2d(self.padding)

        # 对应不同矩形采样窗口的普通卷积
        # 这里 stride 固定为 (N_X, N_Y)，负责把 H*N_X, W*N_Y 打回原尺寸
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    c1,
                    c2,
                    kernel_size=(i // 10, i % 10),
                    stride=(i // 10, i % 10),
                    padding=0,
                    groups=g,
                    bias=False,
                )
                for i in self.i_list
            ]
        )

        # -------------------------------------------------------------
        # 3. 所有分支 stride 都改成 1（和采样逻辑保持一致）
        # -------------------------------------------------------------
        # 仿射变换 M 分支：调制项
        self.m_conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

        # 仿射变换 B 分支：偏置项
        self.b_conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, stride=1),
        )

        # 生成 offset 的特征
        self.p_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.0),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(),
        )

        # 学习 “高度” l(x)
        self.l_conv = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),  # [0,1]
        )

        # 学习 “宽度” w(x)
        self.w_conv = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),  # [0,1]
        )

        self.dropout2 = nn.Dropout2d(0.3)

        # 记录探索阶段学到的 (N_X, N_Y)
        self.register_buffer(
            "reserved_NXY",
            torch.tensor([3, 3], dtype=torch.int32),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        return: (B, C_out, H, W)
        """
        # 近似把前 warmup_iters 次 forward 当成“探索阶段”，之后固定 N_X, N_Y
        if self.training and self._iter < self.warmup_iters:
            epoch_flag = self._iter
            self._iter += 1
        else:
            epoch_flag = self.warmup_iters + 1  # 强制走“固定阶段”的分支

        hw_min, hw_max = self.hw_range
        scale = hw_max // 9
        if hw_min == 1 and hw_max == 3:
            scale = 1

        # 仿射变换相关
        m = self.m_conv(x)       # 调制项 (B, C_out, H, W)
        bias = self.b_conv(x)    # 偏置项 (B, C_out, H, W)

        # 为了让 offset 更敏感，论文实现中乘了一个 100
        offset_feat = self.p_conv(x * 100.0)

        # l, w ∈ [1, hw_max]
        l = self.l_conv(offset_feat) * (hw_max - 1) + 1.0  # (B, 1, H, W)
        w = self.w_conv(offset_feat) * (hw_max - 1) + 1.0  # (B, 1, H, W)

        # 根据论文思想：前若干轮“探索”卷积核的采样点数量，然后固定
        if epoch_flag <= self.warmup_iters:
            # 按 batch 和空间维求均值，作为整体尺度的估计
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)  # scalar
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)  # scalar

            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)

            def phi(xi: int) -> int:
                # 保证为奇数
                if xi % 2 == 0:
                    xi -= 1
                return xi

            N_X, N_Y = phi(N_X), phi(N_Y)
            # 限制在 [3,7] 之间
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)
            N_X, N_Y = min(N_X, 7), min(N_Y, 7)

            # 在“探索期”最后一次，把结果写入 reserved_NXY
            if epoch_flag == self.warmup_iters:
                with torch.no_grad():
                    new_val = torch.tensor(
                        [N_X, N_Y],
                        dtype=torch.int32,
                        device=self.reserved_NXY.device,
                    )
                    self.reserved_NXY.copy_(new_val)
        else:
            # 之后都使用固定好的 reserved_NXY
            N_X = int(self.reserved_NXY[0].item())
            N_Y = int(self.reserved_NXY[1].item())

        N = N_X * N_Y

        # 将 l、w 扩展到 (B, N, H, W)，用来与网格一起生成偏移坐标
        l_rep = l.repeat(1, N, 1, 1)
        w_rep = w.repeat(1, N, 1, 1)
        offset = torch.cat((l_rep, w_rep), dim=1)  # (B, 2N, H, W)

        dtype = offset.data.type()

        # 为了支持大核采样，先对输入做 padding
        if self.padding:
            x_padded = self.zero_padding(x)
        else:
            x_padded = x

        # 生成采样点坐标 p: (B, 2N, H, W)
        p = self._get_p(offset, dtype, N_X, N_Y)
        # (B, H, W, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        # 计算双线性插值所需的四个角点整数坐标
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x_padded.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x_padded.size(3) - 1),
            ],
            dim=-1,
        ).long()

        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x_padded.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x_padded.size(3) - 1),
            ],
            dim=-1,
        ).long()

        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # p 也需要裁剪到合法范围
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x_padded.size(2) - 1),
                torch.clamp(p[..., N:], 0, x_padded.size(3) - 1),
            ],
            dim=-1,
        )

        # 计算双线性插值的权重 (B, H, W, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )

        # 从 x_padded 中取出四个角点对应的特征值 (B, C, H, W, N)
        x_q_lt = self._get_x_q(x_padded, q_lt, N)
        x_q_rb = self._get_x_q(x_padded, q_rb, N)
        x_q_lb = self._get_x_q(x_padded, q_lb, N)
        x_q_rt = self._get_x_q(x_padded, q_rt, N)

        # 按权重进行双线性插值，得到偏移采样后的特征 (B, C, H, W, N)
        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        # 把 (B, C, H, W, N) reshape 成 (B, C, H*N_X, W*N_Y)
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        x_offset = self.dropout2(x_offset)

        # 根据 (N_X, N_Y) 选择对应的普通卷积
        key = N_X * 10 + N_Y
        assert key in self.i_list, f"Kernel size ({N_X}, {N_Y}) not in predefined list"
        conv = self.convs[self.i_list.index(key)]
        x_offset = conv(x_offset)  # (B, C_out, H, W)

        # 最终输出：y = conv(features) ⊙ M + B
        out = x_offset * m + bias
        return out

    # ------------------------------------------------------------------
    # 下面是生成采样坐标和 gather 的工具函数
    # ------------------------------------------------------------------
    def _get_p_n(self, N, dtype, n_x, n_y):
        # 以卷积核中心为原点的规则网格偏移
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
            indexing='ij',
        )

        p_n = torch.cat([p_n_x.reshape(-1), p_n_y.reshape(-1)], dim=0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # 规则网格的“中心位置”，相当于常规卷积的采样中心
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing='ij',
        )

        p_0_x = p_0_x.reshape(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = p_0_y.reshape(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], dim=1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype, n_x, n_y):
        # 结合 offset（由 l,w 归一化后得到）与规则网格偏移，得到最终采样坐标 p
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offset_norm = torch.cat([L, W], dim=1)

        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat(1, 1, h, w)

        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offset_norm * p_n
        return p

    def _get_x_q(self, x, q, N):
        # 从 x 中按照 q 指定的坐标取像素
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)

        # (B, C, H*W)
        x_flat = x.contiguous().view(b, c, -1)
        max_index = x_flat.size(-1) - 1  # H*W - 1

        # 计算一维 index = y * W + x
        index = q[..., :N] * padded_w + q[..., N:]

        # 再做一次安全裁剪，防止任何越界
        index = index.clamp_(0, max_index).long()

        # 展开到 (B, C, H*W*N)
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )

        x_offset = x_flat.gather(dim=-1, index=index)
        x_offset = x_offset.contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        # 将 (B, C, H, W, N) 展成 (B, C, H*n_x, W*n_y)
        b, c, h, w, N = x_offset.size()
        cols = []
        for s in range(0, N, n_y):
            cols.append(
                x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y)
            )
        x_offset = torch.cat(cols, dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset

class Bottleneck_ARConv(nn.Module):
    """Bottleneck block that uses ARConv as the second conv layer."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        c1: in_channels
        c2: out_channels
        shortcut: whether to use residual connection
        g: groups (保留参数以兼容 Bottleneck 接口，一般用不到)
        k: tuple kernel sizes for the first Conv, e.g. (3,3)
        e: expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 第一层还是普通 Conv，保持风格与 Bottleneck_LDConv 一致 :contentReference[oaicite:1]{index=1}
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 第二层用 ARConv（内部自适应矩形卷积）
        self.cv2 = ARConv(c_, c2)  # 用你的 YOLO 版 ARConv 默认参数即可
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C2kARConv(C3):
    """
    ARConv 版本的 C3k 模块：
    继承 C3，只是把内部的 self.m 换成 Bottleneck_ARConv。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        c1, c2: in/out channels
        n: Bottleneck_ARConv 堆叠次数
        k: 传给第一层 Conv 的 kernel size（等价于 (k,k)）
        其它参数保持与 C3 / C3k 接口一致。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(
                Bottleneck_ARConv(c_, c_, shortcut, g, k=(k, k), e=1.0)
                for _ in range(n)
            )
        )

class C3k2_ARConv1(C2f):
    """ARConv version of C3k2_LDConv1: Faster CSP bottleneck with ARConv inside."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        c1, c2: in/out channels
        n: number of blocks in C2f (same as original C3k2_LDConv1)
        c3k: if True, use C2kARConv block, else use Bottleneck_ARConv
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C2kARConv(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_ARConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

class C3k2_ARConv2(C2f):
    """ARConv version of C3k2_LDConv2: mix of ARConv-block and standard Bottleneck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        c1, c2: in/out channels
        n: number of blocks in C2f
        c3k: if True, use C2kARConv, else use standard Bottleneck (pure Conv)
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C2kARConv(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )



if __name__ == "__main__":
    import torch

    x = torch.randn(1, 64, 80, 80)

    m1 = C2kARConv(64, 128, n=1)
    y1 = m1(x)
    print("C2kARConv:", y1.shape)

    m2 = C3k2_ARConv1(64, 128, n=2)
    y2 = m2(x)
    print("C3k2_ARConv1:", y2.shape)

    m3 = C3k2_ARConv2(64, 128, n=2)
    y3 = m3(x)
    print("C3k2_ARConv2:", y3.shape)
