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
# 颜色：以后如果要画框/文字，用这里的颜色
# -------------------------------------------------------
COLORS = [
    (255, 0, 0),     # 柔和红
    (180, 50, 255),  # 紫色（区分度强）
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
        # 注意：这里返回 [[post_result, pre_post_boxes]]
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


# -------------------------------------------------------
# YOLOv8 风格的 target 定义（不改）
# -------------------------------------------------------
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


# -------------------------------------------------------
# 主类：默认只负责「生成热力图」
# -------------------------------------------------------
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
    ):
        device = torch.device(device)
        ckpt = torch.load(weight, map_location=device)
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        # 反传 target
        target = yolov8_target(backward_type, conf_threshold, ratio)

        # 选择层
        target_layers = [model.model[l] for l in layer]

        # Grad-CAM 方法
        cam_method = eval(method)(model, target_layers)
        cam_method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        # 保存到 self
        self.model = model
        self.device = device
        self.method = cam_method
        self.target = target
        self.renormalize = renormalize

    # -------------------------------
    # 核心函数：只计算 Grad-CAM 并返回 cam 图
    # 返回：
    #   cam_image: RGB uint8 的热力图（已经叠在原图上）
    #   img0:      原始 BGR 图像
    #   ratio:     letterbox 缩放比例
    #   dwdh:      letterbox padding
    # -------------------------------
    def compute_cam(self, img_path):
        # 读原图
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        h0, w0 = img0.shape[:2]

        # letterbox
        img, ratio, dwdh = letterbox(img0)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_norm, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 计算 Grad-CAM
        grayscale_cam = self.method(tensor, [self.target])  # [1, H, W]
        grayscale_cam = grayscale_cam[0, :]

        if self.renormalize:
            grayscale_cam = scale_cam_image(grayscale_cam)

        cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)  # RGB

        return cam_image, img0, ratio, dwdh

    # -------------------------------
    # 对外接口：只保存热力图（不画框、不写字）
    # -------------------------------
    def save_cam(self, img_path, save_dir, grad_name):
        os.makedirs(save_dir, exist_ok=True)
        cam_image, _, _, _ = self.compute_cam(img_path)
        out_file = os.path.join(save_dir, f"result_{grad_name}.png")
        Image.fromarray(cam_image).save(out_file)
        print(f"[INFO] 已保存热力图: {out_file}")


# =======================================================
# 下面是「可选」的框体+文字函数
# 👉 默认不调用，你以后需要的时候再用
# =======================================================
def load_boxes_from_txt(txt_path, orig_shape, ratio, dwdh):
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


def draw_detections_on_image(
    img,
    boxes_and_ids,
    box_thickness=2,
    font_thickness=2,
    font_scale=0.8,
    antialias=True,
):
    """
    在图像上绘制检测框 + id/conf 文字
    img: RGB 或 BGR 的 uint8 数组（会原地修改）
    """
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    font = cv2.FONT_HERSHEY_SIMPLEX
    used_label_boxes = []

    for (box, cls_id, conf) in boxes_and_ids:
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        color = COLORS[int(cls_id) % len(COLORS)]

        # 框
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            int(box_thickness),
            lineType=line_type
        )

        # 文本：id:5_0.87
        label = f"id:{int(cls_id)}_{conf:.2f}"

        font_scale_eff = float(font_scale) * 0.6
        thickness = max(1, int(round(font_thickness)))

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale_eff, thickness)

        tx = xmin
        ty = max(ymin - 5, th + baseline)

        # 简单避让
        while any(
            tx < ux + uw and tx + tw > ux and
            ty - th < uy and ty > uy - uh
            for (ux, uy, uw, uh) in used_label_boxes
        ):
            ty += th + baseline + 2

        used_label_boxes.append((tx, ty, tw, th + baseline))

        cv2.putText(
            img,
            label,
            (tx, ty),
            font,
            font_scale_eff,
            color,
            thickness,
            lineType=line_type
        )

    return img


def overlay_boxes_on_cam(
    cam_image,
    txt_path,
    orig_shape,
    ratio,
    dwdh,
    box_thickness=2,
    font_thickness=2,
    font_scale=0.8,
    antialias=True,
):
    """
    给已经生成的 Grad-CAM 图 cam_image 叠加框体和文字。
    cam_image: RGB uint8
    返回：叠加完的 RGB uint8
    """
    boxes_and_ids = load_boxes_from_txt(txt_path, orig_shape, ratio, dwdh)
    if len(boxes_and_ids) == 0:
        return cam_image

    img_draw = cam_image.copy()
    img_draw = draw_detections_on_image(
        img_draw,
        boxes_and_ids,
        box_thickness=box_thickness,
        font_thickness=font_thickness,
        font_scale=font_scale,
        antialias=antialias,
    )
    return img_draw


# -------------------------------------------------------
# 参数设置
# -------------------------------------------------------
def get_params():
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
            # 'weight': '../runsMTSD/yolo11_train/exp/weights/best.pt',
            #'weight': '../runsMTSD/yolo11-FASFFHead_P234_train/exp/weights/best.pt',
            'weight': '../runsMTSD/yolo11-FASFFHead_P234_RCSOSA_ciou_bce_train/exp/weights/best.pt',
            #'weight': '../runsMTSD/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_train/exp/weights/best.pt',
            #'weight': '../runsMTSD/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_distillation/exp/weights/best.pt',
            'device': 'cuda:0',
            'method': grad_name,
            'layer': layers,
            'backward_type': 'class',
            'conf_threshold': 0.2,
            'ratio': 0.02,      # 取前多少目标参与反传
            'renormalize': False,
        }
        yield params


if __name__ == '__main__':
    img_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115.jpg"
    # root_path = r"E:\DataSets\forpaper\ceshiMTSDresult_yolo11"
    root_path = r"E:\DataSets\forpaper\ceshiMTSDresult_yolo11_FASFFHead_P234_RCSOSA_ciou_bce"
    # root_path = r"E:\DataSets\forpaper\ceshiMTSDresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce"
    #root_path = r"E:\DataSets\forpaper\ceshiMTSDresult_yolo11_FASFFHead_P234_RCSOSA_wiou_bce_distillation"
    txt_path = os.path.join(root_path, "p1840115.txt")
    save_path = root_path

    # ========= 默认用法：只保存「纯热力图」 =========
    for each in get_params():
        grad_name = each['method']
        model = yolov11_heatmap(**each)
        model.save_cam(img_path, save_path, grad_name)

    # ========= 如果以后你想要「热力图 + 框体+文字」，示例： =========
    '''
    for each in get_params():
        grad_name = each['method']
        model = yolov11_heatmap(**each)
        # 先算 cam
        cam_image, img0, ratio, dwdh = model.compute_cam(img_path)
        # 再叠加框体和文字
        cam_with_boxes = overlay_boxes_on_cam(
            cam_image,
            txt_path,
            orig_shape=img0.shape[:2],
            ratio=ratio,
            dwdh=dwdh,
            box_thickness=1,
            font_thickness=1,
            font_scale=0.8,
            antialias=True,
        )
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, f"result_{grad_name}_with_box.png")
        Image.fromarray(cam_with_boxes).save(out_file)
    '''
