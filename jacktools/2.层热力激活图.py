import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------------------------------
# 全局开关：是否使用 letterbox
# True  = 使用 letterbox
# False = 不使用 letterbox，直接用原图
# -------------------------------------------------------
USE_LETTERBOX = False

# 可选：如果开启 letterbox，目标尺寸是多少
LETTERBOX_SIZE = (640, 640)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


class FeatureMapExtractor:
    """
    提取指定层的激活图
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = []
        self.handles = []

        for layer in self.target_layers:
            self.handles.append(layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, inp, out):
        # out: Tensor [B, C, H, W]
        self.activations.append(out.detach())

    def clear(self):
        self.activations = []

    def release(self):
        for h in self.handles:
            h.remove()


class yolov11_feature_heatmap:
    def __init__(
        self,
        weight,
        device,
        layer,
        use_letterbox=USE_LETTERBOX,
        letterbox_shape=LETTERBOX_SIZE,
        aggregate='mean',   # 'mean' / 'max'
        renormalize=True
    ):
        self.device = torch.device(device)
        self.model = attempt_load_weights(weight, self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        # 支持单层或多层
        if isinstance(layer, int):
            layer = [layer]

        self.layer_indices = layer
        self.target_layers = [self.model.model[i] for i in layer]
        self.extractor = FeatureMapExtractor(self.model, self.target_layers)

        self.use_letterbox = use_letterbox
        self.letterbox_shape = letterbox_shape
        self.aggregate = aggregate
        self.renormalize = renormalize
        self.weight_path = weight

    def preprocess(self, img_path):
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

        if self.use_letterbox:
            img, ratio, dwdh = letterbox(img0, new_shape=self.letterbox_shape)
        else:
            img = img0.copy()
            ratio = (1.0, 1.0)
            dwdh = (0.0, 0.0)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(
            np.transpose(img_norm, (2, 0, 1))
        ).unsqueeze(0).to(self.device)

        return img0, img_rgb, img_norm, tensor, ratio, dwdh

    def aggregate_feature_map(self, feat):
        """
        feat: Tensor [1, C, H, W]
        返回: heatmap [H, W]
        """
        feat = feat[0].detach().cpu().numpy()  # [C, H, W]

        if self.aggregate == 'max':
            heatmap = np.max(feat, axis=0)
        else:
            heatmap = np.mean(feat, axis=0)

        # 只保留正响应，更符合“热力图”直觉
        heatmap = np.maximum(heatmap, 0)

        return heatmap

    def merge_multi_layers(self, heatmaps, target_size):
        """
        多层热力图融合
        heatmaps: list of [H, W]
        """
        resized_maps = []
        for hm in heatmaps:
            hm_resized = cv2.resize(hm, target_size, interpolation=cv2.INTER_LINEAR)
            resized_maps.append(hm_resized)

        merged = np.mean(np.stack(resized_maps, axis=0), axis=0)

        if self.renormalize:
            min_v, max_v = merged.min(), merged.max()
            merged = (merged - min_v) / (max_v - min_v + 1e-8)

        return merged

    def compute_feature_heatmap(self, img_path):
        img0, img_rgb, img_norm, tensor, ratio, dwdh = self.preprocess(img_path)

        self.extractor.clear()

        with torch.no_grad():
            _ = self.model(tensor)

        if len(self.extractor.activations) == 0:
            raise RuntimeError("没有成功获取到目标层激活，请检查 layer 是否正确。")

        target_size = (img_norm.shape[1], img_norm.shape[0])  # (W, H)
        heatmaps = []

        for feat in self.extractor.activations:
            hm = self.aggregate_feature_map(feat)
            heatmaps.append(hm)

        merged_heatmap = self.merge_multi_layers(heatmaps, target_size)
        cam_image = show_cam_on_image(img_norm, merged_heatmap, use_rgb=True)

        return cam_image, merged_heatmap, img0, ratio, dwdh

    def save_feature_heatmap(self, img_path, save_dir, tag='featuremap'):
        os.makedirs(save_dir, exist_ok=True)

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model_name = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(self.weight_path)
                )
            )
        )

        cam_image, heatmap, _, _, _ = self.compute_feature_heatmap(img_path)

        out_img = os.path.join(
            save_dir,
            f"{img_name}_{tag}_{model_name}.png"
        )
        out_gray = os.path.join(
            save_dir,
            f"{img_name}_{tag}_{model_name}_gray.png"
        )

        Image.fromarray(cam_image).save(out_img)

        gray_uint8 = np.uint8(np.clip(heatmap * 255, 0, 255))
        Image.fromarray(gray_uint8).save(out_gray)

        print(f"[INFO] 已保存彩色层激活热力图: {out_img}")
        print(f"[INFO] 已保存灰度层激活图: {out_gray}")


def get_params():
    params = {
        # 'weight': r'../runsTT100k130/yolo11-FASFFHead_P234_train200/exp/weights/best.pt',
        'weight': r'../runsTT100k130/yolo11_train200/exp/weights/best.pt',
        'device': 'cuda:0',

        # 这里可以写单层，也可以写多层
        # 单层例子：19
        # 多层例子：[19, 22, 25]
        # 'layer': [19, 22, 25],
        'layer': [16],

        'use_letterbox': False,
        'letterbox_shape': (640, 640),

        # mean: 更平滑，更稳
        # max : 更突出局部最强响应
        'aggregate': 'max',

        'renormalize': True,
    }
    return params


if __name__ == '__main__':
    img_path = r"F:\DataSets\resultTT100k130train\multi_model_comparenew\TopK_vis_640\23858.jpg"
    save_dir = r"F:\DataSets\resultTT100k130train\multi_model_comparenew\TopK_vis_640"

    model = yolov11_feature_heatmap(**get_params())
    model.save_feature_heatmap(img_path, save_dir, tag='layer_activation')