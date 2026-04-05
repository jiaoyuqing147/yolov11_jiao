import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch, cv2, os
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


USE_LETTERBOX = False
LETTERBOX_SIZE = (640, 640)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
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
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


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
        统一把输出整理成 [N, 4+nc]
        兼容：
        - [1, C, N]
        - [C, N]
        - [N, C]
        """
        x = model_output

        # 如果是 tuple/list，先取第一个
        if isinstance(x, (list, tuple)):
            x = x[0]

        if not torch.is_tensor(x):
            raise TypeError(f"model_output 不是 tensor，而是 {type(x)}")

        # [1, C, N] -> [C, N]
        if x.dim() == 3:
            if x.size(0) == 1:
                x = x[0]
            else:
                # 理论上 CAM 这里一般是一张图；如果不是，取第一个 batch
                x = x[0]

        # 现在应该是二维
        if x.dim() != 2:
            raise RuntimeError(f"无法处理的输出维度: shape={tuple(x.shape)}")

        # 目标格式应该是 [N, 4+nc]
        # 若当前是 [C, N]，其中 C=4+nc 通常远小于 N，需要转置
        if x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)

        return x

    def forward(self, model_output):
        """
        输入：模型原始输出
        输出：用于反传的标量
        """
        x = self._normalize_output(model_output)   # [N, 4+nc]

        if x.size(1) <= 4:
            raise RuntimeError(f"输出最后一维过小，无法切分 box 和 cls: shape={tuple(x.shape)}")

        boxes_ = x[:, :4]      # [N, 4]
        logits_ = x[:, 4:]     # [N, nc]

        scores = logits_.max(dim=1)[0]   # [N]
        sorted_scores, indices = torch.sort(scores, descending=True)

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

        self.target_layers = [model.model[l] for l in layer]
        self.cam_method = eval(method)(model, self.target_layers)

        self.model = model
        self.device = device
        self.method_name = method
        self.renormalize = renormalize
        self.weight_path = weight
        self.use_letterbox = use_letterbox
        self.letterbox_shape = letterbox_shape

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

        try:
            grayscale_cam = self.cam_method(tensor, [self.target])
            grayscale_cam = grayscale_cam[0, :]

            if self.renormalize:
                grayscale_cam = scale_cam_image(grayscale_cam)

            cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
            return cam_image, img0, ratio, dwdh, True

        except NoValidDetectionsError as e:
            blank_heatmap = np.zeros((img_norm.shape[0], img_norm.shape[1]), dtype=np.float32)
            blue_img = show_cam_on_image(img_norm, blank_heatmap, use_rgb=True)
            print(f"[INFO] {self.method_name}: 未检测到有效目标，返回干净蓝底图。原因: {e}")
            return blue_img, img0, ratio, dwdh, False

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
        out_file = os.path.join(save_dir, f"{img_name}_{grad_name}_{model_name}{suffix}.png")

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
        # 'HiResCAM',
        'LayerCAM',
        # 'RandomCAM',
        # 'EigenGradCAM'
    ]
    #layers = [16, 19, 22]  # yolo11 用这三个特征图
    # layers = [19, 22, 25]
    layers = [19]
    for grad_name in grad_list:
        params = {
           #'weight': '../runsTT100k130/yolo11-OECSOSAInterleave_train200/exp/weights/best.pt',
            #'weight': '../runsTT100k130/yolo11_train200/exp/weights/best.pt',
            #'weight': '../runsTT100k130/yolo11-FASFFHead_P234_train200/exp/weights/best.pt',
            'weight': '../runsTT100k130/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200/exp/weights/best.pt',
             #'weight': '../runsTT100k130/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation/exp/weights/best.pt',

            'device': 'cuda:0',
            'method': grad_name,
            'layer': layers,
            'backward_type': 'class',
            'conf_threshold': 0.20,
            'ratio': 0.05,
            'renormalize': False,
            'use_letterbox': USE_LETTERBOX,
            'letterbox_shape': LETTERBOX_SIZE,
        }
        yield params


if __name__ == '__main__':
    img_path = r"F:\DataSets\resultTT100k130val\multi_model_comparenew\TopK_vis_640\9447.jpg"
    save_path = r"F:\DataSets\resultTT100k130val\multi_model_comparenew\TopK_vis_640"

    for each in get_params():
        grad_name = each['method']
        model = yolov11_heatmap(**each)
        model.save_cam(img_path, save_path, grad_name)