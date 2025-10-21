import torch
import torch.nn as nn
import torch.nn.functional as F  # ★ 新增
from torch.nn.modules.utils import _pair as to_2tuple
from timm.models.layers import DropPath, to_2tuple
from functools import partial
import warnings

__all__ = [
    "LSKNET_Wavelet_Tiny", "LSKNET_Wavelet_Large"
]

# ========= 1) 小波：Haar DWT =========
def _haar_filters(device, dtype):
    # 1D Haar
    h = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / 2.0
    g = torch.tensor([1.0,-1.0], device=device, dtype=dtype) / 2.0
    # 2D separable (LL, LH, HL, HH)
    LL = torch.outer(h, h)
    LH = torch.outer(h, g)
    HL = torch.outer(g, h)
    HH = torch.outer(g, g)
    K = torch.stack([LL, LH, HL, HH], dim=0)  # [4, 2, 2]
    return K

class DWT2D(nn.Module):
    """Haar DWT as fixed depthwise filters; stride=2."""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def _make_weight(self, C, device, dtype):
        K = _haar_filters(device, dtype).unsqueeze(1)  # [4,1,2,2]
        W = K.repeat(C, 1, 1, 1)                       # [4*C,1,2,2]
        return W

    def forward(self, x):  # x: [B,C,H,W]
        B, C = x.shape[:2]
        W = self._make_weight(C, x.device, x.dtype)          # [4*C,1,2,2]
        y = F.conv2d(x, W, stride=2, padding=0, groups=C)    # [B,4*C,H/2,W/2]

        # ★ 关键修改：从 y 读出下采样后的 h2,w2，避免用 H//2, W//2
        h2, w2 = y.shape[-2], y.shape[-1]
        # 用 reshape / view 都行，这里用 reshape 更稳妥
        y = y.reshape(B, 4, C, h2, w2).contiguous()

        LL, LH, HL, HH = y[:, 0], y[:, 1], y[:, 2], y[:, 3]  # each [B,C,h2,w2]
        return LL, LH, HL, HH


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    """改造：在原两条 DW 分支基础上，加入 Wavelet 分支并做 3 路空间选择。"""
    def __init__(self, dim):
        super().__init__()
        # 原两条 DW 分支
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)                     # DW 5x5
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # DW 7x7 dil=3
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)   # → C/2
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)   # → C/2

        # ★ 新增：Wavelet 分支（DWT→上采样→1x1压缩到 C/2）
        self.dwt = DWT2D()
        self.wave_proj = nn.Conv2d(dim * 4, dim // 2, 1)   # 4个子带拼接到 C/2
        self.wave_bn = nn.BatchNorm2d(dim // 2)
        self.wave_act = nn.GELU()

        # 空间注意力：由 2 通道 (avg/max) 生成 3 路权重
        self.conv_squeeze = nn.Conv2d(2, 3, 7, padding=3)  # ★ 原来是 2→2，这里改为 2→3
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 分支1 & 分支2：DW 卷积 → 1x1降到 C/2
        attn1 = self.conv0(x)                  # [B,C,H,W]
        attn2 = self.conv_spatial(attn1)       # [B,C,H,W]
        attn1 = self.conv1(attn1)              # [B,C/2,H,W]
        attn2 = self.conv2(attn2)              # [B,C/2,H,W]

        # 分支3：Wavelet（DWT 返回 H/2,W/2，需要上采样回 H,W）
        LL, LH, HL, HH = self.dwt(x)           # each [B,C,H/2,W/2]
        up = lambda t: F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
        wave = torch.cat([up(LL), up(LH), up(HL), up(HH)], dim=1)   # [B,4C,H,W]
        wave = self.wave_act(self.wave_bn(self.wave_proj(wave)))    # [B,C/2,H,W]

        # 三路拼接 → 生成空间权重 (3 maps) → 逐像素加权
        attn_cat = torch.cat([attn1, attn2, wave], dim=1)   # [B,3*(C/2),H,W]
        avg_attn = torch.mean(attn_cat, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn_cat, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)        # [B,2,H,W]
        sig = self.conv_squeeze(agg).sigmoid()              # [B,3,H,W]

        w1 = sig[:, 0:1, :, :]
        w2 = sig[:, 1:2, :, :]
        w3 = sig[:, 2:3, :, :]
        attn = w1 * attn1 + w2 * attn2 + w3 * wave          # [B,C/2,H,W]
        attn = self.conv(attn)                               # [B,C,H,W]
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = nn.BatchNorm2d(norm_cfg, dim)
            self.norm2 = nn.BatchNorm2d(norm_cfg, dim)
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = nn.BatchNorm2d(norm_cfg, embed_dim)
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class LSKNet_Wavelet(nn.Module):
    def __init__(self, factor=0.5, img_size=224, in_chans=3, dim=None, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__()
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        embed_dims = [int(dim * factor) for dim in embed_dims]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def LSKNET_Wavelet_Tiny(factor):
    model = LSKNet_Wavelet(factor=factor, depths=[2, 2, 2, 2])
    return model


def LSKNET_Wavelet_Large(factor):
    model = LSKNet_Wavelet(factor=factor, depths=[3, 4, 6, 3])
    return model






if __name__ == '__main__':
    model = LSKNET_Wavelet_Large(factor=0.5)
    inputs = torch.randn((1, 3, 640, 640))
    for i in model(inputs):
        print(i.size())

