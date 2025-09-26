# FASFFHead_P345.py
# -*- coding: utf-8 -*-
"""
三路(P3, P4, P5)特征融合并检测的头部实现。
用法（Ultralytics YAML 示例）：
  - [[22, 25, 10], 1, FASFFHead_P345, [nc]]  # Detect(P3, P4, P5)

其中 22/25/10 分别是网络中的 P3/P4/P5 节点索引，请按你实际网络调整。
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["FASFFHead_P345"]

# -------------------------
# 基础工具
# -------------------------
def autopad(k, p=None, d=1):
    """根据 kernel/膨胀自动计算 SAME padding。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d + BN + SiLU"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # 融合后推理（去 BN）
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depthwise Convolution（逐通道卷积）"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DFL(nn.Module):
    """
    Distribution Focal Loss 的积分模块
    参考：Generalized Focal Loss (https://ieeexplore.ieee.org/document/9792391)
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # B, 4*reg_max, A
        # (B, 4, reg_max, A) -> softmax(reg) -> (B, 4, A)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# -------------------------
# 三路融合：P3/P4/P5
# -------------------------
class FASFF_3in_P345(nn.Module):
    """
    三路(P3,P4,P5)多尺度融合：
      level=0 -> 输出 P5'（下采样/32）
      level=1 -> 输出 P4'（下采样/16）
      level=2 -> 输出 P3'（下采样/8）
    约定 ch 顺序为 [C3, C4, C5]
    """
    def __init__(self, level, ch, multiplier=1, rfb=False, vis=False):
        super().__init__()
        assert len(ch) == 3, "FASFF_3in_P345 expects ch=[C3,C4,C5]"
        self.level = level
        self.vis = vis

        C3 = int(ch[0] * multiplier)
        C4 = int(ch[1] * multiplier)
        C5 = int(ch[2] * multiplier)
        # 统一的中间通道选择（按输出尺度 P5/P4/P3）
        self.dim = [C5, C4, C3]      # 对应 [P5, P4, P3]
        self.inter_dim = self.dim[self.level]

        if level == 0:  # 输出 P5'：P4->/2，P3->/4
            self.stride_p4 = Conv(C4, self.inter_dim, 3, 2)         # P4: /2 -> P5 分辨率
            self.pool_p3 = nn.MaxPool2d(3, 2, 1)                     # P3: 8->16
            self.stride_p3 = Conv(C3, self.inter_dim, 3, 2)         # 16->32
            self.expand = Conv(self.inter_dim, C5, 3, 1)
        elif level == 1:  # 输出 P4'：P5 x2 上采样，P3 /2 下采样
            self.compress_p5 = Conv(C5, self.inter_dim, 1, 1)
            self.stride_p3 = Conv(C3, self.inter_dim, 3, 2)         # P3: 8->16
            self.expand = Conv(self.inter_dim, C4, 3, 1)
        else:  # level == 2，输出 P3'：P5 x4，上采样；P4 x2 上采样
            self.compress_p5 = Conv(C5, self.inter_dim, 1, 1)
            self.up_p5 = nn.Upsample(scale_factor=4, mode='nearest')  # 32->8
            self.compress_p4 = Conv(C4, self.inter_dim, 1, 1)
            self.up_p4 = nn.Upsample(scale_factor=2, mode='nearest')  # 16->8
            self.expand = Conv(self.inter_dim, C3, 3, 1)

        # 注意力权重分支（3 路）
        compress_c = 8 if rfb else 16
        self.weight_l0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_l1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_l2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)

    def forward(self, x):
        # x: [P3, P4, P5]
        x_p3, x_p4, x_p5 = x[0], x[1], x[2]

        if self.level == 0:      # 输出 P5'
            l0 = x_p5
            l1 = self.stride_p4(x_p4)
            l2 = self.stride_p3(self.pool_p3(x_p3))
        elif self.level == 1:    # 输出 P4'
            l0 = F.interpolate(self.compress_p5(x_p5), scale_factor=2, mode='nearest')
            l1 = x_p4
            l2 = self.stride_p3(x_p3)
        else:                    # 输出 P3'
            l0 = self.up_p5(self.compress_p5(x_p5))
            l1 = self.up_p4(self.compress_p4(x_p4))
            l2 = x_p3

        # 3 路注意力加权
        w0 = self.weight_l0(l0)
        w1 = self.weight_l1(l1)
        w2 = self.weight_l2(l2)
        weights = F.softmax(self.weight_levels(torch.cat((w0, w1, w2), 1)), dim=1)

        fused = (
            l0 * weights[:, 0:1, :, :] +
            l1 * weights[:, 1:2, :, :] +
            l2 * weights[:, 2:3, :, :]
        )
        out = self.expand(fused)
        if self.vis:
            return out, weights, fused.sum(dim=1)
        return out


# -------------------------
# 检测头：P3/P4/P5
# -------------------------
class FASFFHead_P345(nn.Module):
    """
    检测头（3 路）：输入 [P3, P4, P5]，内部三次融合得到 [P3', P4', P5']，再分别进行 bbox/cls 预测。
    与 P234 头的结构/推理逻辑保持一致，便于替换与对比实验。
    """
    dynamic = False
    export = False
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), multiplier=1, rfb=False):
        """
        Args:
            nc (int): 类别数
            ch (tuple/list): 输入通道 [C3, C4, C5]
            multiplier (float): 通道/宽度缩放
            rfb (bool): 注意力分支的压缩宽度选择（True 更省显存）
        """
        super().__init__()
        assert len(ch) == 3, "FASFFHead_P345 expects ch=[C3,C4,C5]"
        self.nc = nc
        self.nl = 3
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # 中间通道策略（与 P234 头保持一致）
        def _c2(cin): return max(16, cin // 4, self.reg_max * 4)   # bbox 分支中间通道 >= 64
        def _c3(cin): return max(cin, min(self.nc, 100))           # cls 分支中间通道

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, _c2(c), 3), Conv(_c2(c), _c2(c), 3), nn.Conv2d(_c2(c), 4 * self.reg_max, 1))
            for c in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(c, c, 3), Conv(c, _c3(c), 1)),
                nn.Sequential(DWConv(_c3(c), _c3(c), 3), Conv(_c3(c), _c3(c), 1)),
                nn.Conv2d(_c3(c), self.nc, 1),
            )
            for c in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 三个尺度的融合器（level0->P5'，level1->P4'，level2->P3'）
        self.fuse_l0 = FASFF_3in_P345(level=0, ch=ch, multiplier=multiplier, rfb=rfb)
        self.fuse_l1 = FASFF_3in_P345(level=1, ch=ch, multiplier=multiplier, rfb=rfb)
        self.fuse_l2 = FASFF_3in_P345(level=2, ch=ch, multiplier=multiplier, rfb=rfb)

        # end2end（可选）
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    # 训练/推理主路径
    def forward(self, x):
        """
        Args:
            x: List[Tensor]，顺序为 [P3, P4, P5]
        Returns:
            训练：list of feature-head outputs
            推理：tensor 或 (tensor, x_head) 取决于 self.export
        """
        # 融合得到 P5'/P4'/P3'
        p5p = self.fuse_l0(x)     # P5'
        p4p = self.fuse_l1(x)     # P4'
        p3p = self.fuse_l2(x)     # P3'

        # 头部预测顺序按分辨率从大到小给出：[P3', P4', P5']
        x = [p3p, p4p, p5p]

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        if self.end2end:
            return self.forward_end2end_from_headouts(x)

        y = self._inference(x)
        return y if self.export else (y, x)

    # 可选：O2O 路径（与 P234 头一致的接口）
    def forward_end2end_from_headouts(self, x):
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1)
            for i in range(self.nl)
        ]
        if self.training:
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    # 推理：拼接/解码
    def _inference(self, x):
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """与 YOLOv8 风格一致的 bias 初始化（需要 stride 已就绪）"""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        preds: (B, A, 4+nc) with [x, y, w, h, cls...]
        return: (B, K, 6) with [x, y, w, h, score, cls]
        """
        B, A, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, A))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, A))
        i = torch.arange(B)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
