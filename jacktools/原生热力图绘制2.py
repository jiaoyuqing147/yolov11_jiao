import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np

np.random.seed(0)
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import (
    GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM,
    HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # [h, w]
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
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


class yolov11_heatmap:
    def __init__(
        self,
        weight,
        cfg,
        device,
        method,
        layer,
        backward_type,
        conf_threshold,
        ratio,
        topk=1,                  # 只保存前几个目标，默认 1
        cam_percentile=90,       # 保留前 10% 高响应
        blur_ksize=11,           # 高斯平滑核大小，建议奇数
        only_show_hotspots=True  # 只给热点区域上色
    ):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()

        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
        model.load_state_dict(csd, strict=False)
        model.eval()

        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted_scores, indices = torch.sort(logits_.max(1)[0], descending=True)

        return (
            torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]],
            torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]],
            xywh2xyxy(
                torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
            ).cpu().detach().numpy()
        )

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(
            img, str(name), (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            tuple(int(x) for x in color), 2,
            lineType=cv2.LINE_AA
        )
        return img

    def clean_heatmap(self, img_rgb_float, saliency_map):
        """
        让热力图更像论文风格：
        1. 归一化
        2. 高斯平滑
        3. 百分位阈值过滤
        4. 只在热点区域叠加颜色
        """
        saliency_map = saliency_map.astype(np.float32)

        # 归一化
        sal_min, sal_max = saliency_map.min(), saliency_map.max()
        if (sal_max - sal_min) < 1e-12:
            return None, None

        saliency_map = (saliency_map - sal_min) / (saliency_map - sal_min).max()

        # 平滑
        if self.blur_ksize is not None and self.blur_ksize >= 3:
            k = self.blur_ksize
            if k % 2 == 0:
                k += 1
            saliency_map = cv2.GaussianBlur(saliency_map, (k, k), 0)

        # 再归一化一次
        sal_min, sal_max = saliency_map.min(), saliency_map.max()
        if (sal_max - sal_min) < 1e-12:
            return None, None
        saliency_map = (saliency_map - sal_min) / (sal_max - sal_min)

        # 百分位阈值：只保留强响应
        positive_vals = saliency_map[saliency_map > 0]
        if len(positive_vals) == 0:
            return None, None

        thresh = np.percentile(positive_vals, self.cam_percentile)
        saliency_map_clean = saliency_map.copy()
        saliency_map_clean[saliency_map_clean < thresh] = 0

        # 再归一化，增强热点对比
        if saliency_map_clean.max() > 0:
            saliency_map_clean = saliency_map_clean / saliency_map_clean.max()

        # 先生成整张 CAM 图
        cam_image_full = show_cam_on_image(img_rgb_float.copy(), saliency_map_clean, use_rgb=True)

        if not self.only_show_hotspots:
            return cam_image_full, saliency_map_clean

        # 只在热点区域上色，其余保持原图
        mask = saliency_map_clean > 0
        base_img = (img_rgb_float * 255).astype(np.uint8).copy()
        result = base_img.copy()
        result[mask] = cam_image_full[mask]

        return result, saliency_map_clean

    def __call__(self, img_path, save_path):

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        print("save_path =", os.path.abspath(save_path))

        # === 1. 读图 ===
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"图像读取失败: {img_path}")

        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0

        tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(self.device)

        # === 2. 初始化 CAM（只初始化一次！）===
        cam = self.method(
            model=self.model,
            target_layers=self.target_layers
        )

        # === 3. 前向（拿分数做判断）===
        with torch.no_grad():
            result = self.model(tensor)

        post_result, _, _ = self.post_process(result[0])

        print("post_result.shape =", post_result.shape)
        print("top 10 scores =", [float(post_result[i].max()) for i in range(min(10, post_result.size(0)))])

        score_val = float(post_result[0].max())
        print(f"[INFO] top1 score = {score_val:.6f}")

        if score_val < self.conf_threshold:
            print(f"[STOP] 没有满足阈值的目标")
            return

        # === 4. 计算 CAM ===
        grayscale_cam = cam(input_tensor=tensor)[0]

        # === 5. 后处理（你写得很好 👍）===
        clean_cam, _ = self.clean_heatmap(img.copy(), grayscale_cam)

        if clean_cam is None:
            print("[SKIP] 无有效热力图")
            return

        # === 6. 保存 ===
        out_file = os.path.join(save_path, "0.png")
        Image.fromarray(clean_cam).save(out_file)

        print(f"[SAVE] {out_file}")
        print("[DONE]")

def get_params():
    params = {
        'weight': r'../runsTT100k130/yolo11_train200/exp/weights/best.pt',
        'cfg': r'../ultralytics/cfg/models/11/yolo11.yaml',
        'device': 'cuda:0',
        'method': 'HiResCAM',      # 推荐先试 HiResCAM；也可换 GradCAM
        'layer': 'model.model[16]', # 先试 16；也可试 19 / 22
        'backward_type': 'class',
        'conf_threshold': 0.01,
        'ratio': 0.02,

        # 新增
        'topk': 1,                 # 只保存 1 张
        'cam_percentile': 90,      # 85~92 都可以试
        'blur_ksize': 11,          # 9, 11, 13 都可试
        'only_show_hotspots': True
    }
    return params


if __name__ == '__main__':
    model = yolov11_heatmap(**get_params())
    model(
        r'F:\DataSets\resultTT100k130train\multi_model_comparenew\TopK_vis\23858.jpg',
        r'F:\DataSets\resultTT100k130train\multi_model_comparenew\TopK_vis\result_v11_clean'
    )