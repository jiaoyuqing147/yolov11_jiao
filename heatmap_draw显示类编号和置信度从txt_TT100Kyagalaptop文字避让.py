import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch, yaml, cv2, os, shutil, sys
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
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
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

# -------------------------------------------------------
# 10 种固定的鲜艳颜色（按 id 循环使用）
# -------------------------------------------------------
# COLORS = [
#     (255, 0, 0),      # 红
#     # (0, 255, 0),      # 绿
#     # (0, 0, 255),      # 蓝
#     # (255, 128, 0),    # 橙
#     # (255, 0, 255),    # 品红
#     #
#     # (128, 0, 255),    # 紫
#     # (0, 128, 255),    # 蓝绿
#     # (128, 255, 0),    # 黄绿
#     # (255, 0, 128),    # 粉
# ]
COLORS = [
    (255, 0, 0),     # 柔和红
    # (255, 160, 60),    # 橙色（亮但不刺眼）
    # (255, 220, 0),     # 暖黄（特别适合淡蓝背景）
    # (60, 200, 255),    # 天蓝（兼容，亮）
    (180, 50, 255),    # 紫色（区分度强）
]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return (
            torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]],
            torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]],
            xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
        )

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


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
        show_box,
        renormalize,
        box_thickness=2,      # 框线粗细
        font_thickness=2,     # 字体粗细
        font_scale=0.8,       # 字体大小
        antialias=True
    ):
        # 保存绘制相关参数
        self.box_thickness = box_thickness
        self.font_thickness = font_thickness
        self.font_scale = font_scale
        self.line_type = cv2.LINE_AA if antialias else cv2.LINE_8

        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        # 使用我们自定义的颜色列表
        self.fixed_colors = COLORS

        # 把局部变量全部挂到 self 上（保持你原来的写法）
        self.__dict__.update(locals())

    # ====================================================
    # 从【指定的】txt 标注文件读取框，并映射到 letterbox 后坐标
    # ====================================================
    def load_boxes_from_txt(self, txt_path, orig_shape, ratio, dwdh):
        """
        从指定 txt 标注文件读取 YOLO 格式:
        - GT:    cls x y w h
        - 预测:  cls x y w h conf ...
        返回: [(box_xyxy, cls_id, conf), ...]
        """
        if not os.path.exists(txt_path):
            print(f"[WARN] 标注文件不存在: {txt_path}")
            return []

        h0, w0 = orig_shape
        r = ratio[0]      # 宽高缩放比例
        dw, dh = dwdh     # padding 偏移

        boxes_and_ids = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    print(f"[WARN] 标注行列数不足 5，已跳过: {line}")
                    continue

                cls = float(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                bw  = float(parts[3])
                bh  = float(parts[4])
                cls_id = int(cls)

                if len(parts) >= 6:
                    conf = float(parts[5])
                else:
                    conf = 1.0   # 没有置信度时，默认 1.0（比如 GT）

                # 归一化坐标 → 原图像素坐标
                x_c *= w0
                y_c *= h0
                bw  *= w0
                bh  *= h0

                x1 = x_c - bw / 2
                y1 = y_c - bh / 2
                x2 = x_c + bw / 2
                y2 = y_c + bh / 2

                # 原图坐标 → letterbox 后坐标
                x1 = x1 * r + dw
                y1 = y1 * r + dh
                x2 = x2 * r + dw
                y2 = y2 * r + dh

                boxes_and_ids.append(([x1, y1, x2, y2], cls_id, conf))

        return boxes_and_ids

    def draw_detections(self, box, cls_id, conf, img):
        """
        在图像上绘制检测框 + 显示 id 和置信度，带文字避让
        """
        xmin, ymin, xmax, ymax = list(map(int, list(box)))

        # 颜色按 id 循环选取
        color = self.fixed_colors[int(cls_id) % len(self.fixed_colors)]

        # 绘制矩形框
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            int(self.box_thickness),
            lineType=self.line_type
        )

        # ---------- 文字避让逻辑开始 ★★ ----------
        # 标签格式：id:5_0.87
        label = f"id:{int(cls_id)}_{conf:.2f}"

        # 初始化已使用文字区域列表
        if not hasattr(self, "used_label_boxes"):
            self.used_label_boxes = []

        # 计算文字尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = float(self.font_scale) * 0.6
        thickness = max(1, int(round(self.font_thickness)))  # 至少 1 像素

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 默认放在框上方
        tx = xmin
        ty = max(ymin - 5, th + baseline)

        # 检查是否与已有标签重叠，若重叠则往下挪
        while any(
            tx < ux + uw and tx + tw > ux and
            ty - th < uy and ty > uy - uh
            for (ux, uy, uw, uh) in self.used_label_boxes
        ):
            ty += th + baseline + 2  # 往下挪一行

        # 记录当前文字区域
        self.used_label_boxes.append((tx, ty, tw, th + baseline))

        # 绘制文字
        cv2.putText(
            img,
            label,
            (tx, ty),
            font,
            font_scale,
            color,
            thickness,
            lineType=self.line_type
        )
        # ---------- 文字避让逻辑结束 ★★ ----------

        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, txt_path, save_path):
        # 每张图开始前重置文字区域 ★★
        self.used_label_boxes = []

        # 先读原图（为了根据原尺寸还原 txt 标注）
        img0 = cv2.imread(img_path)
        if img0 is None:
            print(f"[ERROR] 无法读取图像: {img_path}")
            return
        h0, w0 = img0.shape[:2]

        # letterbox 缩放 + padding
        img, ratio, dwdh = letterbox(img0)
        # Grad-CAM 用的是 RGB、float32、0~1
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_norm, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 计算 Grad-CAM
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

        # 根据 txt 读框
        boxes_and_ids = self.load_boxes_from_txt(txt_path, (h0, w0), ratio, dwdh)

        if self.show_box and len(boxes_and_ids) > 0:
            for (box, cls_id, conf) in boxes_and_ids:
                cam_image = self.draw_detections(box, cls_id, conf, cam_image)

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, txt_path, save_path, grad_name):
        # 这里默认只处理单张图 + 单个 txt
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        out_file = os.path.join(save_path, f"result_{grad_name}.png")
        self.process(img_path, txt_path, out_file)


def get_params():
    # 绘制热力图方法列表
    grad_list = [
        'GradCAM',
        'GradCAMPlusPlus',
        'XGradCAM',
        'EigenCAM',
        'HiResCAM',
        'LayerCAM',
        'RandomCAM',
        'EigenGradCAM'
    ]

    # 自定义需要绘制热力图的层索引
    #layers = [16, 19, 22]  # yolo11 用这三个特征图
    layers = [19, 22, 25]    # yolo11-FASFFHead_P234 模型用这三个特征图

    for grad_name in grad_list:
        params = {
            # 'weight': 'runsTT100k130/yolo11_train/exp/weights/best.pt',
             'weight': 'runsTT100k130/yolo11-FASFFHead_P234_train/exp/weights/best.pt',
            #'weight': 'runsTT100k130/yolo11-FASFFHead_P234_RCSOSA_ciou_bce_train/exp/weights/best.pt',
             #'weight': 'runsTT100k130/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_train/exp/weights/best.pt',
            # 'weight': 'runsTT100k130/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_distillation/exp/weights/best.pt',

            'device': 'cuda:0',      # cpu 或者 cuda:0
            'method': grad_name,     # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
            'layer': layers,         # 计算梯度的层索引
            'backward_type': 'class',  # class, box, all
            'conf_threshold': 0.2,   # 置信度阈值
            'ratio': 0.02,           # 取前多少目标参与反传
            'show_box': True,        # 是否显示检测框（这里：根据 txt）
            'renormalize': False,    # 是否在框内重归一化热力图

            # 可选：统一调节绘制风格
            'box_thickness': 1,
            'font_thickness': 1,
            'font_scale': 0.8,
            'antialias': True,
        }
        yield params


if __name__ == '__main__':
    # 例：TT100K / TT100K 单张图
    img_path = r"E:\DataSets\forpaper\ceshiTT100K\9447.jpg"   # 图像路径
    # root_path = r'E:\DataSets\forpaper\ceshiTT100Kresult_yolo11'
    root_path = r'E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234'
    #root_path = r'E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_ciou_bce'
    #root_path = r'E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce'
    #root_path = r'E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation'
    txt_path = root_path + "\9447.txt"   # 标注 txt 路径
    save_path = root_path  # 保存结果的路径

    # 遍历所有的参数并生成热力图
    for each in get_params():
        model = yolov11_heatmap(**each)
        model(img_path, txt_path, save_path, each['method'])
