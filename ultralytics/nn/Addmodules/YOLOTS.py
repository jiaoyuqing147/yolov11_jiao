import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ['SimFusion_3in', 'dilation_block']

class SimFusion_3in(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_t, x_s, x_m = x
        B, C, H, W = x_t.shape

        x_s = F.interpolate(x_s, size=(H, W), mode='bilinear', align_corners=False)
        x_m = F.interpolate(x_m, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_t, x_s, x_m], 1)
        return out


class dilation_block(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c2, 1, 1)

        self.cv2 = Conv(c_, c_, 3, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 3, 1)
        self.cv5 = Conv(c_, c_, 3, 1)
        self.cv6 = Conv(c_, c_, 3, 1)

        self.cv8 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(c_, c_, 3, 1, d=3)

        self.cv9 = Conv(5 * c_, c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        block1 = self.cv3(self.cv2(y[-1]))
        block2 = self.cv5(self.cv4(block1))
        block3 = self.cv7(self.cv6(block2))

        y[0] = self.cv8(y[0])
        y.extend([block1, block2, block3])

        out = self.cv9(torch.cat(y, 1))
        return out