# import subprocess
#
# ink = r"C:\Program Files\WindowsApps\25415Inkscape.Inkscape_1.4.21.0_x64__9waqn51p1ttv2\VFS\ProgramFilesX64\Inkscape\bin\inkscape.exe"
# svg = r"E:\DataSets\forpaper\ceshiTT100K\9447_640_vector.svg"
#
# # 输出仍然是 SVG（矢量，推荐论文用）
# out_svg = r"E:\DataSets\forpaper\ceshiTT100K\9447_roi2.svg"
#
# # 你给的坐标
# x1, y1 = 190, 185
# x2, y2 = 330, 230
#
# cmd = [
#     ink,
#     svg,
#     f"--export-area={x1}:{y1}:{x2}:{y2}",
#     "--export-type=svg",
#     f"--export-filename={out_svg}",
# ]
#
# result = subprocess.run(cmd, capture_output=True, text=True)
#
# print("stdout:", result.stdout)
# print("stderr:", result.stderr)
# print("导出完成:", out_svg)
import xml.etree.ElementTree as ET
from pathlib import Path

def crop_svg_viewbox(svg_in, svg_out, x1, y1, x2, y2):
    svg_in = Path(svg_in)
    svg_out = Path(svg_out)

    if not svg_in.exists():
        raise FileNotFoundError(f"找不到输入 SVG 文件: {svg_in}")

    # 解析 SVG
    tree = ET.parse(svg_in)
    root = tree.getroot()

    # 处理命名空间（否则会出现 ns0:svg 之类前缀）
    ns = {"svg": "http://www.w3.org/2000/svg"}
    ET.register_namespace("", ns["svg"])

    # 计算新的宽高（单位还是用 px）
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("裁剪区域宽或高 <= 0，请检查坐标")

    # 设置 viewBox = x1 y1 width height
    root.set("viewBox", f"{x1} {y1} {width} {height}")

    # 同时把 width / height 改成裁剪后大小（方便 Office / 浏览器显示）
    root.set("width", str(width))
    root.set("height", str(height))

    # 保存新的 SVG
    tree.write(svg_out, encoding="utf-8", xml_declaration=True)
    print(f"[OK] 已保存裁剪后的 SVG: {svg_out}")


if __name__ == "__main__":
    svg_in = r"E:\DataSets\forpaper\ceshiTT100K\9447_640_vector.svg"
    svg_out = r"E:\DataSets\forpaper\ceshiTT100K\9447_roi_viewbox.svg"

    # 你给的坐标，如果是裁png，用这个
    # x1, y1 = 190, 185
    # x2, y2 = 330, 230

    x1, y1 = 140, 130
    x2, y2 = 270, 180

    crop_svg_viewbox(svg_in, svg_out, x1, y1, x2, y2)
