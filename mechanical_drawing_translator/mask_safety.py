# mechanical_drawing_translator/mask_safety.py
"""基于源页像素保守判断视觉遮挡是否会覆盖工程线条。"""

from __future__ import annotations

from dataclasses import replace
from math import hypot

from PySide6.QtGui import QImage, qGray

from .annotation_models import AnnotationItem, AnnotationLayout, MaskDecision, Polygon


# 视觉遮挡需要覆盖字形边缘，但扩展会增加擦到邻近工程线的风险。
# 这里沿顶点相对中心的方向扩张，并非几何库意义的等距多边形缓冲。
# 扩张结果必须再检查背景和其它源对象，不能只检查原 OCR 框。
# 函数只创建新顶点元组，不修改原区域多边形。
def _expanded_polygon(polygon: Polygon, padding: float) -> Polygon:
    center_x = sum(point[0] for point in polygon) / len(polygon)
    center_y = sum(point[1] for point in polygon) / len(polygon)
    expanded = []
    for x, y in polygon:
        # 中心到顶点很近时用小下限避免除零。
        # padding 使用源单位，输出仍位于同一源坐标体系。
        # 这种径向扩展保留顶点次序，后续 painter 按同序连线。
        distance = max(0.001, hypot(x - center_x, y - center_y))
        scale = 1.0 + padding / distance
        expanded.append((center_x + (x - center_x) * scale,
                         center_y + (y - center_y) * scale))
    return tuple(expanded)


# 边界采样统一钳制到有效像素，避免四周环带越界访问。
# 灰度只用于本模块的邻域统计，最终纯白资格另由 BlankSpace 检验。
# 浅色彩线可能在灰度中不显著，所以此分析不能单独授权擦除。
def _gray(image: QImage, x: int, y: int) -> int:
    x = min(max(0, x), image.width() - 1)
    y = min(max(0, y), image.height() - 1)
    return qGray(image.pixel(x, y))


def _analyze_polygon(
    layout: AnnotationLayout, image: QImage, polygon: Polygon,
) -> tuple[MaskDecision, Polygon]:
    # 源布局与当前分析栅格分辨率可以不同，分别映射 X/Y。
    # padding 至少为 1.5 源单位，并考虑几个分析像素的覆盖余量。
    # 先在源空间扩展，再映射到像素，避免缩放改变保存的几何语义。
    # 采样框向外取整，保守包含可能被遮挡的边缘。
    # 本模块不证明原墨迹是英文；调用方必须先提供独立原生去字背景。
    sx = image.width() / layout.page_width
    sy = image.height() / layout.page_height
    source_padding = max(1.5, min(layout.page_width / image.width(),
                                  layout.page_height / image.height()) * 3.0)
    expanded = _expanded_polygon(polygon, source_padding)
    pixels = tuple((x * sx, y * sy) for x, y in expanded)
    min_x = max(0, int(min(x for x, _ in pixels)))
    max_x = min(image.width() - 1, int(max(x for x, _ in pixels) + 0.999))
    min_y = max(0, int(min(y for _, y in pixels)))
    max_y = min(image.height() - 1, int(max(y for _, y in pixels) + 0.999))
    # 极小分析块无法形成可靠边界和周围背景样本，直接拒绝。
    # 不能因为像素太少而认为没有碰到工程线。
    # source_padding 同时决定环带厚度，确保分析范围与实际扩大蒙版对应。
    if max_x - min_x < 3 or max_y - min_y < 3:
        return MaskDecision.UNSUPPORTED_BACKGROUND, expanded

    band = max(2, round(min(sx, sy) * source_padding))
    # 上下环带再加左右环带，构成候选框外围背景观测。
    # 不以四角代表整个背景，单侧穿入的引线也必须有机会被采到。
    # 只采候选外部来估计背景明度，避免原文字黑色本身拉低估计。
    # 没有任何外围样本时保守拒绝，不构造默认白色。
    outside_values: list[int] = []
    for x in range(max(0, min_x - band), min(image.width(), max_x + band + 1)):
        for y in range(max(0, min_y - band), min_y):
            outside_values.append(_gray(image, x, y))
        for y in range(max_y + 1, min(image.height(), max_y + band + 1)):
            outside_values.append(_gray(image, x, y))
    for y in range(min_y, max_y + 1):
        for x in range(max(0, min_x - band), min_x):
            outside_values.append(_gray(image, x, y))
        for x in range(max_x + 1, min(image.width(), max_x + band + 1)):
            outside_values.append(_gray(image, x, y))
    if not outside_values:
        return MaskDecision.UNSUPPORTED_BACKGROUND, expanded
    # 平均明度和暗像素比例共同检查，避免局部黑线被高均值掩盖。
    # 偏暗纸面、扫描噪声或复杂纹理在本路径不具备白底覆盖合同。
    # 阈值是保守启发式，不能宣称做了通用工程线语义分割。
    # 通过外围背景只允许继续几何检查，并非完成安全判断。
    mean = sum(outside_values) / len(outside_values)
    dark_ratio = sum(value < 180 for value in outside_values) / len(outside_values)
    if mean < 225 or dark_ratio > 0.08:
        return MaskDecision.UNSUPPORTED_BACKGROUND, expanded

    # 工程线可能是灰色、扫描淡线或抗锯齿边缘；相对明亮背景有明显对比
    # 即按几何处理，不能只识别接近纯黑的像素。
    # 阈值相对周围明度下降，检测淡尺寸线与抗锯齿边。
    # 四侧集合分别记录边缘暗像素，不要求一条线贯穿两个相对侧。
    # 只从一个方向进入的小引线也可能包含工程意义。
    # 范围跨边界内外，避免恰好停在遮挡边上的线被漏过。
    boundary_threshold = max(110, min(225, int(mean - 20)))
    left_hits = {y for y in range(min_y, max_y + 1)
                 if any(_gray(image, x, y) < boundary_threshold
                        for x in range(max(0, min_x - band), min(image.width(), min_x + 2)))}
    right_hits = {y for y in range(min_y, max_y + 1)
                  if any(_gray(image, x, y) < boundary_threshold
                         for x in range(max(0, max_x - 1), min(image.width(), max_x + band + 1)))}
    top_hits = {x for x in range(min_x, max_x + 1)
                if any(_gray(image, x, y) < boundary_threshold
                       for y in range(max(0, min_y - band), min(image.height(), min_y + 2)))}
    bottom_hits = {x for x in range(min_x, max_x + 1)
                   if any(_gray(image, x, y) < boundary_threshold
                          for y in range(max(0, max_y - 1), min(image.height(), max_y + band + 1)))}
    # 任一侧出现从遮挡区外延伸到边界内的暗像素，都可能是只从一侧进入的
    # 引线、斜向尺寸线或轮廓线。宁可误判为不遮挡，也不能擦除工程几何。
    # 任意一侧可疑就拒绝整个多边形，宁可保留英文也不冒险擦线。
    # SAFE 只说明本分析未发现边界风险。
    # 整框内部短线仍需调用方在独立去字背景上逐块查空白。
    # 扫描字形与线相连时，OCR 本身不能提供这种独立背景。
    if left_hits or right_hits or top_hits or bottom_hits:
        return MaskDecision.UNSAFE_GEOMETRY, expanded
    return MaskDecision.SAFE, expanded


def analyze_mask_safety(layout: AnnotationLayout, source_image: QImage) -> AnnotationLayout:
    """返回带安全结论的新布局；失败时从不默认遮挡。"""
    # 未申请遮挡时保持原对象，不额外改变文字位置或状态。
    # 当前双语调用者先清掉旧蒙版后再进入本方法，防止关开关残留。
    # 此函数只处理已传入的候选蒙版，不自行推导新 source 范围。
    if not layout.mask_requested:
        return layout
    # 背景缺失属于不支持，绝不能把空 QImage 当纯白页。
    # 清空全部蒙版后回退到已保存的邻近文字位置。
    # 警告保留原文未擦的事实，用户仍可人工检查版面。
    # 这是历史基础分析的回退行为；双语分支会保持原已定位置。
    if source_image.isNull():
        items = tuple(replace(
            item, text_bounds=item.fallback_text_bounds or item.text_bounds,
            mask_polygons=(), mask_decision=MaskDecision.UNSUPPORTED_BACKGROUND,
            warnings=(*item.warnings, "无法读取背景，已跳过英文遮挡",
                      "译文已移到邻近位置，请人工复核版面"),
        ) for item in layout.items)
        return replace(layout, items=items)

    analyzed_items: list[AnnotationItem] = []
    for item in layout.items:
        # 一个组可能包含多个原文字块，所有多边形都必须安全。
        # 不能只遮掉安全的一半组，造成原文被拆碎或语义误配。
        # 分析得到的新顶点与决策写入新 item，原布局对象不变。
        # 无候选多边形时没有安全证据，不使用空集合 all 的真值通过。
        decisions: list[MaskDecision] = []
        expanded: list[Polygon] = []
        for polygon in item.mask_polygons:
            decision, expanded_polygon = _analyze_polygon(layout, source_image, polygon)
            decisions.append(decision)
            expanded.append(expanded_polygon)
        if decisions and all(decision == MaskDecision.SAFE for decision in decisions):
            analyzed_items.append(replace(
                item, mask_polygons=tuple(expanded), mask_decision=MaskDecision.SAFE,
            ))
        else:
            # 若任一块检测到工程几何，整体优先报告几何不安全。
            # 其它缺证据情况统一背景不支持，方便调用者区分原因。
            # 失败必清空蒙版，同时保留译文和原始源关联。
            # 诊断提示只描述本启发式结果，不可视作专家线稿核验。
            decision = (MaskDecision.UNSAFE_GEOMETRY
                        if MaskDecision.UNSAFE_GEOMETRY in decisions
                        else MaskDecision.UNSUPPORTED_BACKGROUND)
            reason = ("检测到穿越文字区的尺寸线或轮廓线，已跳过英文遮挡"
                      if decision == MaskDecision.UNSAFE_GEOMETRY
                      else "文字区背景复杂或无法安全分析，已跳过英文遮挡")
            analyzed_items.append(replace(
                item,
                text_bounds=item.fallback_text_bounds or item.text_bounds,
                mask_polygons=(), mask_decision=decision,
                warnings=(*item.warnings, reason, "译文已移到邻近位置，请人工复核版面"),
            ))
    return replace(layout, items=tuple(analyzed_items))
