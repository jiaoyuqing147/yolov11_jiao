import xml.etree.ElementTree as ET
from pathlib import Path

def crop_svg_translate(svg_in, svg_out, x1, y1, x2, y2):
    svg_in = Path(svg_in)
    svg_out = Path(svg_out)

    if not svg_in.exists():
        raise FileNotFoundError(f"找不到输入 SVG 文件: {svg_in}")

    tree = ET.parse(svg_in)
    root_old = tree.getroot()

    # SVG namespace
    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("", SVG_NS)

    width = x2 - x1
    height = y2 - y1

    # ⛔ 不要手动设置 xmlns !!
    root_new = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}"
        }
    )

    # 关键：用 <g transform> 平移原图
    g = ET.Element(f"{{{SVG_NS}}}g", {"transform": f"translate({-x1}, {-y1})"})

    # 把所有子节点移动到 g 中
    for child in list(root_old):
        root_old.remove(child)
        g.append(child)

    root_new.append(g)

    # 写出 SVG
    new_tree = ET.ElementTree(root_new)
    new_tree.write(svg_out, encoding="utf-8", xml_declaration=True)
    print(f"[OK] 已保存裁剪后的 SVG: {svg_out}")


if __name__ == "__main__":
    svg_in = r"E:\DataSets\forpaper\ceshiTT100K\9447_640_vector.svg"
    svg_out = r"E:\DataSets\forpaper\ceshiTT100K\9447_roi_viewbox2.svg"

    # 你想要的区域
    x1, y1 = 190, 185
    x2, y2 = 330, 230

    crop_svg_translate(svg_in, svg_out, x1, y1, x2, y2)
