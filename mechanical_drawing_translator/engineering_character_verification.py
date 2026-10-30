# mechanical_drawing_translator/engineering_character_verification.py
"""Image-backed D8-C verification for narrowly bounded engineering glyph risks."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import re

from PIL import Image

from .engineering_source_risk import assess_source_risk
from .models import TextRegion


# 本模块仅处理一种窄风险：
# 角公差末尾的度数被认作零。
# 输入必须仍携带原始OCR字面，
# 图像只为送译候选提供旁证。
# 不据中文译文推断英文字符，
# 也不授予遮挡或导出资格。
_TRAILING_DEGREE_AS_ZERO = re.compile(r"(?P<prefix>±\s*\d+\s*/\s*\d+)0\s*$")


def _component_boxes(image: Image.Image) -> list[tuple[int, int, int, int, int]]:
    # 连通域在局部灰度裁片上计算，
    # 阈值200将深色像素列为前景。
    # 这里不是再次调用OCR模型，
    # 没有新的字词识别置信度。
    # 返回框采用右下开区间，
    # 最后一项保留域内像素数。
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    dark = {(x, y) for y in range(height) for x in range(width) if pixels[x, y] < 200}
    boxes: list[tuple[int, int, int, int, int]] = []
    # 每个像素从集合移除一次，
    # 四邻接避免斜接笔画强行合并。
    # 队列展开的是同一深色组件，
    # 不沿图纸文字阅读顺序遍历。
    # 不足三个像素的噪点不入框。
    while dark:
        seed = dark.pop()
        queue = deque((seed,))
        xs: list[int] = []
        ys: list[int] = []
        while queue:
            x, y = queue.popleft()
            xs.append(x); ys.append(y)
            for point in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if point in dark:
                    dark.remove(point); queue.append(point)
        if len(xs) >= 3:
            boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(xs)))
    return boxes


def _glyph_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    # 过宽组件可能是线稿或底线，
    # 此处不把它当作单个字形。
    # 高度不足两像素同样排除，
    # 以免孤立噪声充当度数证据。
    components = [box for box in _component_boxes(image)
                  if box[2] - box[0] < width * 0.4 and box[3] - box[1] >= 2]
    groups: list[list[tuple[int, int, int, int, int]]] = []
    # 横向相接的组件归为一字形，
    # 允许一像素间隙连接断笔。
    # 分组依据只是横向投影，
    # 不是通用字符分割算法。
    # 上下分离的小圆也可能成组，
    # 因此后续还须检查相对几何。
    for box in sorted(components):
        placed = False
        for group in groups:
            gx0 = min(item[0] for item in group)
            gx1 = max(item[2] for item in group)
            if box[0] <= gx1 + 1 and box[2] >= gx0 - 1:
                group.append(box); placed = True; break
        if not placed:
            groups.append([box])
    return [(min(item[0] for item in group), min(item[1] for item in group),
             max(item[2] for item in group), max(item[3] for item in group))
            for group in groups]


def _looks_like_small_trailing_degree(image: Image.Image) -> bool:
    glyphs = _glyph_boxes(image)
    if len(glyphs) < 2:
        return False
    # 至少两个字形才有比较基准，
    # 只比较末字与紧邻的前字。
    # 度数候选应更小且位置偏高，
    # 宽度也不应显著超过前字。
    # 这些比例是有限几何门禁，
    # 不能推广为所有字体识别率。
    previous, last = glyphs[-2], glyphs[-1]
    previous_height = previous[3] - previous[1]
    last_height = last[3] - last[1]
    previous_center = (previous[1] + previous[3]) / 2
    last_center = (last[1] + last[3]) / 2
    return (
        previous_height >= 8
        and last_height <= previous_height * 0.72
        and (last[2] - last[0]) <= (previous[2] - previous[0]) * 1.15
        and last_center <= previous_center - max(1.0, previous_height * 0.1)
    )


def verify_engineering_region(
    page_image: Image.Image, region: TextRegion, *, x_scale: float, y_scale: float,
) -> TextRegion:
    """Return a copy carrying verified translation input; never replace OCR text."""
    # 先由原文字段确认角公差语境，
    # 普通尺寸中的零不进入修正。
    # 没有风险时直接返回原对象，
    # 避免给无关区域制造核验记录。
    risk = assess_source_risk(region.source_text)
    if "engineering_source_degree_zero_confusion" not in risk.warning_codes:
        return region
    match = _TRAILING_DEGREE_AS_ZERO.search(region.source_text)
    if match is None:
        return replace(
            region,
            verification_warnings=("d8c_character_evidence_insufficient",),
        )
    # 区域坐标先按横纵比例投影，
    # 再加两像素边距保留上标。
    # 四边裁剪到真实页面范围，
    # 不假设输入坐标就是像素。
    # 比例由调用者的页面渲染决定，
    # 本模块不修改源坐标或多边形。
    bounds = region.bounds
    padding = 2
    box = (
        max(0, int(bounds.min_x * x_scale) - padding),
        max(0, int(bounds.min_y * y_scale) - padding),
        min(page_image.width, int(bounds.max_x * x_scale) + padding + 1),
        min(page_image.height, int(bounds.max_y * y_scale) + padding + 1),
    )
    patch = page_image.crop(box)
    if not _looks_like_small_trailing_degree(patch):
        return replace(
            region,
            verification_source="local_image_glyph_geometry",
            verification_warnings=("d8c_character_evidence_insufficient",),
        )
    # 只有语境与图像同时支持才替换，
    # 替换对象仅为verified_source_text。
    # source_text和区域身份仍保留，
    # 后续能解释实际送译的字面。
    # 0.96是该规则给出的证据权重，
    # 不代表模型或专家质量验收。
    verified = _TRAILING_DEGREE_AS_ZERO.sub(r"\g<prefix>°", region.source_text)
    return replace(
        region,
        verified_source_text=verified,
        verification_source="local_image_glyph_geometry",
        verification_confidence=0.96,
        verification_warnings=("d8c_character_auto_verified",),
    )
