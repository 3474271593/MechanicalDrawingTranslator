# mechanical_drawing_translator/annotation_painter.py
"""在任意 QPainter 目标上绘制同一份中文标注布局。"""

from PySide6.QtCore import QPointF,Qt,QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QFontMetricsF

from .annotation_models import AnnotationLayout, MaskDecision
from .annotation_preview import footprint
from .font_resources import annotation_font
from .glyph_text_geometry import validate_text_origin


# 统一红色供屏幕与矢量覆盖层共享，避免两套绘制样式漂移。
# 颜色是中文标注层样式，原图字形和工程线不在此重新上色。
# 仅绘制布局，不在 painter 中运行 OCR、翻译或重新寻找位置。
ANNOTATION_RED = QColor("#dc2626")


def paint_annotation_layout(painter: QPainter, layout: AnnotationLayout) -> None:
    # 调用者负责设置从源单位到目标设备的变换。
    # 图片通常一单位一像素，PDF 覆盖层以 72 DPI 对齐一单位一 point。
    # 英文模式在任何画笔或蒙版动作前直接退出，确保没有残留覆盖层。
    # save/restore 让本函数不污染调用方的缩放、笔刷与字体状态。
    if not painter.isActive():
        raise ValueError("annotation painter is not active")
    if layout.display_mode == 'english':
        return
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    for item in layout.items:
        # 兼容中文原位模式依赖安全擦除，失败项不能伪装成原位中文。
        # 当前双语不受此条件限制，遮挡失败仍可显示已确定的邻近中文。
        # 待布局或旧 margin 项没有安全绘制位置，仅留数据供点击检查。
        if layout.display_mode=='chinese' and item.mask_decision!=MaskDecision.SAFE:
            continue
        # Pending placement retains its source link in the table, without ink.
        if item.placement in {'needs_layout_review', 'margin'}:
            continue
        # Reject malformed saved origins before either erasing or painting.
        # 先验证基线再画蒙版，避免清掉英文后才发现中文不可绘制。
        # 项目文件可由外部修改，不能因为曾保存过就信任 origin。
        # 异常项跳过整个绘制事务，保持原图作为安全回退。
        try:validate_text_origin(item)
        except (ValueError,TypeError):continue
        # Legacy or malformed project objects cannot extend the output canvas.
        # 必须检查旋转后的真实占用包围框，水平 text_bounds 不足以判越界。
        # 纸张边界先检查，可靠图框存在时进一步收紧。
        # 本函数绝不因布局越界扩大画布，避免输出比例改变。
        # 这里是最终防线，不能代替自动候选阶段的碰撞检测。
        f=footprint(item.text_bounds,item.rotation_degrees)
        if (f.min_x < 0 or f.min_y < 0 or f.max_x > layout.page_width
                or f.max_y > layout.page_height):
            continue
        frame=layout.effective_bounds
        if frame is not None and (f.min_x<frame.min_x or f.min_y<frame.min_y
                or f.max_x>frame.max_x or f.max_y>frame.max_y):
            continue
        # Source diagnostics belong to the explicit inspection/focus layer.
        # An unavailable translation must not silently recolor original ink in
        # either the ordinary preview or a shared export painter.
        # 详情、翻译失败和保留源记录都属于无墨迹项。
        # 它们仍可被界面聚焦，但普通预览与导出不额外染红原文。
        # 同一个共享绘制函数维持这些状态在各输出目标上的一致语义。
        if item.placement in {'detail','unavailable','preserved'}:
            continue
        # 需要安全结论和本次用户开关同时成立，不能单看保存的 SAFE。
        # 绘制只在输出副本上填色，PDF 原始文字对象仍保留在底层。
        # mask_fill 是已存布局颜色，本函数不猜测背景、不补洞、不做修复。
        # 独立证据必须由调用链在构造布局时提供，painter 不产生授权。
        # 蒙版先画，再画中文，避免白底覆盖刚生成的译文字形。
        if item.mask_decision == MaskDecision.SAFE and layout.mask_requested:
            painter.setPen(QPen(QColor(*item.mask_fill), 0))
            painter.setBrush(QColor(*item.mask_fill))
            for polygon in item.mask_polygons:
                painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in polygon]))

        bounds = item.text_bounds
        width = bounds.max_x - bounds.min_x
        height = bounds.max_y - bounds.min_y
        font = annotation_font(item.font_size)
        painter.save()
        # 绕框中心旋转，与 footprint 几何公式使用相同中心。
        # 再移回局部左上角，使基线坐标可以按未旋转框解释。
        # 每项单独保存状态，前一项旋转不会累积到下一项。
        # 源单位字号由 annotation_font 固定，不读显示器缩放。
        painter.translate(bounds.min_x + width / 2, bounds.min_y + height / 2)
        painter.rotate(item.rotation_degrees)
        painter.translate(-width / 2, -height / 2)
        painter.setFont(font)
        painter.setPen(QPen(ANNOTATION_RED, 0))
        # 旧布局按字体 ascent 放第一条基线，保持历史渲染结果。
        # 紧字形布局使用已验证的第一基线偏移，去掉多余字体留白。
        # 行距仍来自同一字体度量，多行不会因紧框而重叠。
        # 两种路径都绘制保存的 lines，不能在这里再裁剪或重排。
        if item.text_origin is None:
            line_height = painter.fontMetrics().lineSpacing()
            baseline_x=0
            baseline = painter.fontMetrics().ascent()
        else:
            line_height=QFontMetricsF(font).lineSpacing()
            baseline_x,baseline=item.text_origin
        # 完整逐行输出，当前行位置只由基线与行距决定。
        # 空行仍参与行距，保留显式段落结构。
        # 外层 restore 成对释放本项和整页状态，不承担结束 painter 的所有权。
        for index, line in enumerate(item.lines):
            painter.drawText(QPointF(baseline_x, baseline + index * line_height), line)
        painter.restore()
    painter.restore()
