import os
from PIL import Image


def resize_to_640_512_png(in_path, out_path):
    try:
        print("输入图像：", in_path)
        img = Image.open(in_path).convert("RGB")
        print("原始尺寸：", img.size)

        # 直接缩放为 640 x 512（拉伸或压缩）
        target_size = (640, 512)
        img = img.resize(target_size, Image.LANCZOS)

        # 确保输出目录存在
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # PNG 无损保存
        img.save(out_path, format="PNG", compress_level=0)
        print("保存成功（PNG 无损）：", out_path)
        print("文件是否存在：", os.path.exists(out_path))

    except Exception as e:
        print("出错了：", e)


if __name__ == "__main__":
    in_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115.jpg"
    out_path = r"E:\DataSets\forpaper\ceshiMTSD\p1840115_640x512.png"

    resize_to_640_512_png(in_path, out_path)
