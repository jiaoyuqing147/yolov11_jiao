"""
Implementation of Prof-of-Concept Network: StarNet.
We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.
Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

__all__ = ['starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4', 'starnet_s050', 'starnet_s100', 'starnet_s150']

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class StarNet(nn.Module):
    def __init__(self, depth=0.25, width=0.5, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0,
                 num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        base_dim = _make_divisible(int(base_dim * width), 8)
        self.in_channel = _make_divisible(int(32 * width), 8)
        # self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        # self.norm = nn.BatchNorm2d(self.in_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        unique_tensors = {}
        for stage in self.stages:
            x = stage(x)
            width, height = x.shape[2], x.shape[3]
            unique_tensors[(width, height)] = x
        result_list = list(unique_tensors.values())[-4:]
        return result_list
        # x = torch.flatten(self.avgpool(self.norm(x)), 1)
        # return self.head(x)


def starnet_s1(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=24, depths=[2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


def starnet_s2(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=32, depths=[1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


def starnet_s3(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=32, depths=[2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


def starnet_s4(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=32, depths=[3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


def starnet_s050(depth=0.5, width=0.25, pretrained=False, **kwargs):
    return StarNet(depth=depth, width=width, base_dim=16, depths=[1, 1, 3, 1], mlp_ratio=3, **kwargs)


def starnet_s100(depth=0.5, width=0.25, pretrained=False, **kwargs):
    return StarNet(depth=depth, width=width, base_dim=20, depths=[1, 2, 4, 1], mlp_ratio=4, **kwargs)


def starnet_s150(depth=0.5, width=0.25, pretrained=False, **kwargs):
    return StarNet(depth=depth, width=width, base_dim=24, depths=[1, 2, 4, 2], mlp_ratio=3, **kwargs)


if __name__ == "__main__":
    model = starnet_s1()
    inputs = torch.randn((1, 3, 640, 640))
    for i in model(inputs):
        print(i.size()) 