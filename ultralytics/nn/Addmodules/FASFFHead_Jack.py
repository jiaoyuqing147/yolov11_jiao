import copy
import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors
import math
import torch.nn.functional as F

__all__ = ['FASFFHead_Jack']

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


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

#FASFF 是一种 多尺度特征融合模块，核心目标是，把不同层级的特征图（例如 P2、P3、P4）对齐成同一个尺度后融合在一起，增强当前层级的表达能力。
class FASFF(nn.Module):
    def __init__(self, level, ch, multiplier=1, rfb=False, vis=False):
        '''
        Args:这里的level主要是根据当前融合的层 level，定义从其他层拉取信息并融合的模块结构
            level:
                对不同 level（0~2）采取不同融合策略

                        Level 0：融合 P4, P3, P2 → 输出增强后的P4。
                        Level 1：融合 P4, P3, P2 → 输出增强后的P3
                        Level 2：融合 P4, P3, P2 → 输出增强后的p2
            融合的过程是：把其它层调整为当前层分辨率和通道数，再通过 softmax 加权融合（3路融合），再用 expand 卷积还原维度
            ch: 接收了 3 层特征图作为输入
            multiplier:控制每一层通道数的缩放比例（一般用于不同模型规模，比如 YOLOv8-n、s、m、l、x）
            rfb:是否启用 RFB（Receptive Field Block）策略的开关
            vis:是否在前向传播时输出额外的可视化信息，方便调试或分析
        '''
        super(FASFF, self).__init__()

        self.level = level
        self.dim = [int(c * multiplier) for c in ch[::-1]]  # ch 是 [32, 64, 128]，反转为 [128, 64, 32]
        self.inter_dim = self.dim[self.level]#后面准备融合三个通道，以哪个通道为最终的统一的通道数

        if level == 0:
            self.stride_p3 = Conv(self.dim[1], self.inter_dim, 3, 2) #用来给p3下采样的，可以跟p4通道对齐
            self.pool_p2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#用来给p2最大池化的
            self.stride_p2 = Conv(self.dim[2], self.inter_dim, 3, 2) #用来给最大池化后的p2下采样的，可以跟p4通道对齐
            self.expand = Conv(self.inter_dim, self.dim[0], 3, 1) # 输出增强后的 P4：128→128，保持 40×40
        elif level == 1:
            self.compress_p4 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)#对P4降通道128→64
            self.stride_p2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)#P2：通道 32→64，同时下采样 160→80
            self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1) #输出增强型 P3：通道 64 → 64（可不变），尺寸保持 80×80
        elif level == 2:
            self.compress_p4 = Conv(self.dim[0], self.inter_dim, 1, 1)#用来给P4降通道的，把128通道降成32
            self.up_p4 = nn.Upsample(scale_factor=4, mode='nearest')#处理降低通道后的P4，上采样 P4：40×40→160×160
            self.compress_p3 = Conv(self.dim[1], self.inter_dim, 1, 1)# 降通道 P3：64→32
            self.up_p3 = nn.Upsample(scale_factor=2, mode='nearest')# 上采样 P3：80×80→160×160
            self.expand = Conv(self.inter_dim, self.dim[2], 3, 1)
        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16#临时压缩通道数
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):
        x_p2, x_p3, x_p4 = x[0], x[1], x[2]

        if self.level == 0:
            l0 = x_p4
            l1 = self.stride_p3(x_p3)
            l2 = self.stride_p2(self.pool_p2(x_p2))

        elif self.level == 1:
            l0 = F.interpolate(self.compress_p4(x_p4), scale_factor=2, mode='nearest')# P4: compress_p4降通道128→64，然后后上采样 40→80
            l1 = x_p3#p3保持不变
            l2 = self.stride_p2(x_p2)#下采样，这个stride_p2既能让通道32->64。又因为空间下采样，160——》80

        elif self.level == 2:
            l0 = self.up_p4(self.compress_p4(x_p4))
            l1 = self.up_p3(self.compress_p3(x_p3))
            l2 = x_p2

        w0 = self.weight_level_0(l0)
        w1 = self.weight_level_1(l1)
        w2 = self.weight_level_2(l2)

        #根据特征图内容自动学习出 P4、P3、P2 三个分支的融合权重，实现软注意力加权融合。
        weights_input = torch.cat([w0, w1, w2], dim=1)  # 第一步：拼接三个注意力分支的特征# shape: [B, 3 × compress_c, H, W]
        weights_logits = self.weight_levels(weights_input)  # 第二步：通过 1 个卷积生成每个分支的注意力分数（未归一化） # shape: [B, 3, H, W]
        weights = F.softmax(weights_logits, dim=1)   # 第三步：对每个位置的 3 个通道进行 softmax → 得到注意力权重（归一化）# shape: [B, 3, H, W]

        fused = l0 * weights[:, 0:1, :, :] + l1 * weights[:, 1:2, :, :] + l2 * weights[:, 2:3, :, :]
        out = self.expand(fused)

        if self.vis:
            return out, weights, fused.sum(dim=1)
        else:
            return out



class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)



class FASFFHead_Jack(nn.Module):
    """YOLOv8 Detect head for detection models. CSDNSnu77"""

    dynamic = False  # force grid reconstruction  是否强制重新构建 grid（用于推理时动态 shape）
    export = False  # export mode 是否为导出模式（onnx/tflite）
    end2end = False  # end2end  是否开启 One2One matching 模式
    max_det = 300  # max_det  每张图最多输出目标个数
    shape = None       #缓存特征图 shape
    anchors = torch.empty(0)  # init  # 初始化 anchor
    strides = torch.empty(0)  # init  # 初始化 stride

    def __init__(self, nc=80, ch=(), multiplier=1, rfb=False):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        print("[DEBUG] Entered FASFFHead.__init__")
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers  #ch: 主干网络输出特征的通道列表，比如 [128, 256, 512, 1024],这四个是backbone最终要传给检测头的，
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)#这里也可以考虑用ch[0] // 16，但是目前是已经写死了，写成了16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        # c2 是 bbox 分支的中间通道数，必须满足 reg_max×4 的最小需求（比如 4×16=64），否则 DFL 无法输出；
        # c3 是分类分支的中间通道数，需兼顾输入特征通道 ch[0] 和类别数（最多不超过 100），保证表达力。
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels，c2表示 bbox 分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        # self.l0_fusion = FASFF(level=0, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l1_fusion = FASFF(level=1, ch=ch, multiplier=multiplier, rfb=rfb)#准备在这里进行调整，输出增强后的P4
        self.l2_fusion = FASFF(level=2, ch=ch, multiplier=multiplier, rfb=rfb)#准备在这里进行调整，输出增强后的P3
        self.l3_fusion = FASFF(level=3, ch=ch, multiplier=multiplier, rfb=rfb)#准备在这里进行调整，输出增强后的P2
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        #x1 = self.l0_fusion(x)#本来是处理P5的，删除
        x2 = self.l1_fusion(x)
        x3 = self.l2_fusion(x)
        x4 = self.l3_fusion(x)
        x = [x4, x3, x2]#融合后的p2 p3 p4
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        '''
                x[i] 是第 i 层的融合特征图，比如 x[0] 是融合后的 P2，x[1] 是融合后的 P3，以此类推。
                cv2[i] 是预测 边界框分支（bbox） 的头部，输出为 4 * reg_max 通道（用于 DFL 分布回归）。
                cv3[i] 是预测 类别分支（cls） 的头部，输出为 nc 通道（每个 anchor 的类别概率）。
                比如 上面处理后的p2，经过 cv2，得到的是 B，64，160，160；经过cv3得到的是 B，232，160，160 . cat起来就是296，160，160
        '''
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


if __name__ == "__main__":
    # 假设模型结构 ch=[32, 64, 128]，乘上 multiplier=1
    ch = [32, 64, 128]

    model = FASFF(level=2, ch=ch, multiplier=1, vis=True)

    # 构造伪造输入（batch=1）
    x_p2 = torch.randn(1, 32, 160, 160)
    x_p3 = torch.randn(1, 64, 80, 80)
    x_p4 = torch.randn(1, 128, 40, 40)

    # 前向传播
    out, weights, fused_sum = model([x_p2, x_p3, x_p4])

    # 打印输出信息
    print("输出特征图 shape：", out.shape)
    print("注意力权重 shape：", weights.shape)
    print("融合后特征图（降维前） sum shape：", fused_sum.shape)