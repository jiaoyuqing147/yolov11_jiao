import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import cv2
import os
import numpy as np

np.random.seed(0)
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from pytorch_grad_cam import (
    GradCAMPlusPlus,
    GradCAM,
    XGradCAM,
    EigenCAM,
    HiResCAM,
    LayerCAM,
    RandomCAM,
    EigenGradCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image


USE_LETTERBOX = True
LETTERBOX_SIZE = (640, 640)

# 是否把热力图贴回原始大图
OVERLAY_ON_ORIGINAL = True


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (w, h)
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
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def remove_letterbox_padding(cam_map, dwdh):
    """
    把 640x640 上的 CAM 去掉 padding，只保留有效图像区域
    """
    dw, dh = dwdh
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    h, w = cam_map.shape[:2]

    x1 = left
    x2 = w - right if right > 0 else w
    y1 = top
    y2 = h - bottom if bottom > 0 else h

    # 防御性处理，避免切片越界
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        # 极端情况下返回原图
        return cam_map

    return cam_map[y1:y2, x1:x2]


class NoValidDetectionsError(Exception):
    pass


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def _normalize_output(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        统一整理成 [N, 4+nc]
        兼容：
        - [1, C, N]
        - [C, N]
        - [N, C]
        """
        x = model_output

        if isinstance(x, (list, tuple)):
            x = x[0]

        if not torch.is_tensor(x):
            raise TypeError(f"model_output 不是 tensor，而是 {type(x)}")

        # [1, C, N] -> [C, N]
        if x.dim() == 3:
            if x.size(0) == 1:
                x = x[0]
            else:
                x = x[0]

        if x.dim() != 2:
            raise RuntimeError(f"无法处理的输出维度: shape={tuple(x.shape)}")

        # 若当前是 [C, N]，通常 C=4+nc 比 N 小，需要转置到 [N, C]
        if x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)

        return x

    def forward(self, model_output):
        x = self._normalize_output(model_output)   # [N, 4+nc]

        if x.size(1) <= 4:
            raise RuntimeError(f"输出最后一维过小，无法切分 box 和 cls: shape={tuple(x.shape)}")

        boxes_ = x[:, :4]      # [N, 4]
        logits_ = x[:, 4:]     # [N, nc]

        scores = logits_.max(dim=1)[0]
        _, indices = torch.sort(scores, descending=True)

        post_result = logits_[indices]       # [N, nc]
        pre_post_boxes = boxes_[indices]     # [N, 4]

        outputs = []

        max_num = int(post_result.size(0) * self.ratio)
        max_num = max(1, max_num)
        max_num = min(max_num, post_result.size(0))

        for i in range(max_num):
            score = post_result[i].max()

            if float(score) < self.conf:
                break

            if self.ouput_type == 'class' or self.ouput_type == 'all':
                outputs.append(score)

            if self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    outputs.append(pre_post_boxes[i, j])

        if len(outputs) == 0:
            raise NoValidDetectionsError(
                f"没有满足 conf_threshold={self.conf} 的目标"
            )

        return sum(outputs)


class yolov11_heatmap:
    def __init__(
        self,
        weight,
        device,
        method,
        layer,
        backward_type,
        conf_threshold,
        ratio,
        renormalize=False,
        use_letterbox=USE_LETTERBOX,
        letterbox_shape=LETTERBOX_SIZE,
        merge_mode='mean',             # 'mean' / 'max' / 'weighted'
        merge_weights=None,            # 例如 [0.5, 0.3, 0.2]
        overlay_on_original=OVERLAY_ON_ORIGINAL,
    ):
        device = torch.device(device)
        model = attempt_load_weights(weight, device)
        model.info()

        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        self.target = yolov8_target(backward_type, conf_threshold, ratio)

        if isinstance(layer, int):
            layer = [layer]

        self.layer_ids = layer
        self.target_layers = [model.model[l] for l in layer]

        self.model = model
        self.device = device
        self.method_name = method
        self.renormalize = renormalize
        self.weight_path = weight
        self.use_letterbox = use_letterbox
        self.letterbox_shape = letterbox_shape

        self.merge_mode = merge_mode
        self.merge_weights = merge_weights
        self.overlay_on_original = overlay_on_original

    def _build_cam_for_single_layer(self, layer_module):
        return eval(self.method_name)(self.model, [layer_module])

    def _merge_cams(self, cams):
        """
        cams: list of [H, W]
        """
        stack = np.stack(cams, axis=0)  # [L, H, W]

        if self.merge_mode == 'mean':
            merged = np.mean(stack, axis=0)

        elif self.merge_mode == 'max':
            merged = np.max(stack, axis=0)

        elif self.merge_mode == 'weighted':
            if self.merge_weights is None:
                raise ValueError("merge_mode='weighted' 时必须提供 merge_weights")
            weights = np.array(self.merge_weights, dtype=np.float32)
            if len(weights) != stack.shape[0]:
                raise ValueError(
                    f"merge_weights 长度 {len(weights)} 与层数 {stack.shape[0]} 不一致"
                )
            weights = weights / (weights.sum() + 1e-8)
            merged = np.sum(stack * weights[:, None, None], axis=0)

        else:
            raise ValueError("merge_mode 必须是 'mean' / 'max' / 'weighted'")

        return merged

    def _overlay_cam_on_original(self, merged_cam, img0, dwdh):
        """
        将 640x640(含padding) 上的 CAM 正确映射回原图尺寸后叠加
        """
        # 先去掉 padding
        cam_crop = remove_letterbox_padding(merged_cam, dwdh)

        # 再 resize 回原图大小
        cam_orig = cv2.resize(
            cam_crop,
            (img0.shape[1], img0.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # 归一化到 0~1
        cam_orig = cam_orig.astype(np.float32)
        cam_orig = (cam_orig - cam_orig.min()) / (cam_orig.max() - cam_orig.min() + 1e-8)

        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img0_norm = np.float32(img0_rgb) / 255.0

        cam_image = show_cam_on_image(img0_norm, cam_orig, use_rgb=True)
        return cam_image

    def _overlay_cam_on_input_view(self, merged_cam, img_rgb):
        """
        叠加在 640x640 输入图上
        """
        img_norm = np.float32(img_rgb) / 255.0
        merged_cam = merged_cam.astype(np.float32)
        merged_cam = (merged_cam - merged_cam.min()) / (merged_cam.max() - merged_cam.min() + 1e-8)
        cam_image = show_cam_on_image(img_norm, merged_cam, use_rgb=True)
        return cam_image

    def compute_cam(self, img_path):
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
        tensor = torch.from_numpy(np.transpose(img_norm, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        cams = []

        # 每层单独算 CAM，再自己融合
        for layer_id, layer_module in zip(self.layer_ids, self.target_layers):
            cam_method = self._build_cam_for_single_layer(layer_module)
            try:
                grayscale_cam = cam_method(tensor, [self.target])   # [1, h, w]
                grayscale_cam = grayscale_cam[0, :]

                # resize 到输入图尺寸（一般是640x640）
                grayscale_cam = cv2.resize(
                    grayscale_cam,
                    (img_norm.shape[1], img_norm.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

                cams.append(grayscale_cam)

            except NoValidDetectionsError:
                print(f"[INFO] {self.method_name}: 第 {layer_id} 层没有有效目标，已跳过。")
                continue

        # 所有层都没有有效目标
        if len(cams) == 0:
            if self.overlay_on_original:
                blank_heatmap = np.zeros((img0.shape[0], img0.shape[1]), dtype=np.float32)
                img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img0_norm = np.float32(img0_rgb) / 255.0
                blue_img = show_cam_on_image(img0_norm, blank_heatmap, use_rgb=True)
            else:
                blank_heatmap = np.zeros((img_norm.shape[0], img_norm.shape[1]), dtype=np.float32)
                blue_img = show_cam_on_image(img_norm, blank_heatmap, use_rgb=True)

            print(f"[INFO] {self.method_name}: 所有层都没有有效目标，返回干净蓝底图。")
            return blue_img, img0, ratio, dwdh, False

        merged_cam = self._merge_cams(cams)

        if self.renormalize:
            merged_cam = scale_cam_image(merged_cam)

        merged_cam = merged_cam.astype(np.float32)
        min_v, max_v = merged_cam.min(), merged_cam.max()
        merged_cam = (merged_cam - min_v) / (max_v - min_v + 1e-8)

        if self.overlay_on_original:
            cam_image = self._overlay_cam_on_original(merged_cam, img0, dwdh)
        else:
            cam_image = self._overlay_cam_on_input_view(merged_cam, img_rgb)

        return cam_image, img0, ratio, dwdh, True

    def save_cam(self, img_path, save_dir, grad_name):
        os.makedirs(save_dir, exist_ok=True)

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        model_name = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(self.weight_path)
                )
            )
        )

        cam_image, _, _, _, has_target = self.compute_cam(img_path)

        suffix = "" if has_target else "_no_det"
        size_tag = "_orig" if self.overlay_on_original else "_input640"

        out_file = os.path.join(
            save_dir,
            f"{img_name}_{model_name}_{grad_name}_{self.merge_mode}{size_tag}{suffix}.png"
        )

        Image.fromarray(cam_image).save(out_file)

        if has_target:
            print(f"[INFO] 已保存热力图: {out_file}")
        else:
            print(f"[INFO] 已保存干净蓝底图: {out_file}")


def get_params():
    grad_list = [
        # 'GradCAM',
        # 'GradCAMPlusPlus',
        # 'XGradCAM',
        # 'EigenCAM',
        'HiResCAM',
        'LayerCAM',
        # 'RandomCAM',
        'EigenGradCAM'
    ]

    #layers = [16, 19, 22]
    layers = [19, 22, 25]

    for grad_name in grad_list:
        params = {
            #'weight': '../runsTT100k130/yolo11_train200/exp/weights/best.pt',
             #'weight': '../runsTT100k130/yolo11-OECSOSAInterleave_train200/exp/weights/best.pt',
            #'weight': '../runsTT100k130/yolo11-FASFFHead_P234_train200/exp/weights/best.pt',
            #'weight': '../runsTT100k130/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200/exp/weights/best.pt',
            'weight': '../runsTT100k130/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation/exp/weights/best.pt',

            'device': 'cuda:0',
            'method': grad_name,
            'layer': layers,
            'backward_type': 'class',
            'conf_threshold': 0.25,
            'ratio': 0.1,
            'renormalize': False,
            'use_letterbox': USE_LETTERBOX,
            'letterbox_shape': LETTERBOX_SIZE,

            # ========= 融合方式 =========
            # 可选: 'mean' / 'max' / 'weighted'
            'merge_mode': 'max',

            # 对应 layers = [19, 22, 25]
            # 小目标优先：浅层权重大一些
            'merge_weights': [0.9, 0.05, 0.05],

            # True: 贴回原始大图
            # False: 仍贴在640输入图
            'overlay_on_original': True,
        }
        yield params


if __name__ == '__main__':
    # img_path = r"F:\DataSets\resultTT100k130test\multi_model_comparenew2\TopK_vis\96661.jpg"
    img_path = r"E:\DataSets\resultTT100k130train\multi_model_comparenew2\TopK_vis\23858.jpg"
    save_path = r"E:\DataSets\resultTT100k130train\multi_model_comparenew2\TopK_vis\23858"

    for each in get_params():
        grad_name = each['method']
        model = yolov11_heatmap(**each)
        model.save_cam(img_path, save_path, grad_name)