# mechanical_drawing_translator/glyph_text_geometry.py
"""Actual glyph ink bounds for tight, readable Chinese annotation footprints.

The baseline offset is explicit so fitting and painting share the same font
metrics. Removing ascent/descent padding never reduces the minimum font size,
changes source coordinates, or grants evidence to erase the original English.
"""
import math
from PySide6.QtCore import QRectF
from PySide6.QtGui import QFontMetricsF
from .font_resources import annotation_font, ensure_font_supports


# 字体 advance/行距包含留白，紧框使用实际字形墨迹包围矩形。
# 每行先放到对应基线，再联合所有非空墨迹框。
# 空行仍影响后续基线偏移，但自己不增加虚假的墨迹范围。
# 返回 QRectF 是局部未旋转坐标，外部再定位和旋转。
# 紧框只改善空间利用，不改变原文坐标或遮挡证据等级。
def _ink_bounds(lines, metrics):
    ink=None
    for index,line in enumerate(lines):
        box=metrics.tightBoundingRect(line).translated(0,index*metrics.lineSpacing())
        if box.isEmpty():continue
        ink=box if ink is None else ink.united(box)
    return ink if ink is not None else QRectF()


# 该适配器复用引擎的工程标识符换行，避免出现另一套断行语义。
# 去掉 ascent/descent 的多余留白不等于缩小字形本身。
# 最小字号至少六源单位，并同时尊重引擎更高的可读下限。
# 返回原点是首条基线相对紧框左上角的偏移。
# 这个偏移必须与 lines、字号、宽高作为整体保存。
# 后续 painter 不得自行换成 ascent，否则实际墨迹会离开验证框。
def tight_text_fit(engine,text,max_width,max_height,initial_size):
    """Return (lines, size, ink_width, ink_height, first_baseline_origin).

    Origin is relative to the top-left of the unrotated ink footprint. Existing
    wrapping keeps engineering identifiers intact. Empty or unreadable fits
    fail, instead of truncating text or shrinking below six source units.
    """
    # 非字符串、纯空白、超长文本或非有限尺寸没有有效几何意义。
    # 有限性先于大小判断，防止 NaN 让比较全部为 False。
    # 宽高非正直接拒绝，不能用零框表示一个已经成功的布局。
    # 长度限制只拒绝异常输入，不裁剪完整译文来凑适配成功。
    if (not isinstance(text,str) or not text.strip() or len(text)>2000
            or not all(math.isfinite(v) for v in (max_width,max_height,initial_size))
            or max_width<=0 or max_height<=0):
        raise ValueError('Invalid tight glyph fit input')
    # 初始字号限制在引擎范围并保证可读下限。
    # 先验证字体是否含全部字形，避免缺字方框被当短中文通过。
    # 实际 QFont 使用整数像素字号，度量与共享绘制一致。
    # 适配循环每次只减一并在下限终止，不做无限收缩。
    minimum=max(6.0,engine.min_font_size)
    size=max(minimum,min(initial_size,engine.max_font_size))
    ensure_font_supports(text,size)
    while True:
        metrics=QFontMetricsF(annotation_font(size))
        lines=engine._wrap(text,metrics,max_width)
        # 测量所有行的联合墨迹，而不是仅最长行或第一行。
        # 宽高同时容纳才返回，bearing 的负左边与负顶边通过 origin 抵消。
        # 空墨迹不能返回成功，否则导出回读找不到实际中文。
        # 返回的负 ink.left/top 是局部基线补偿，不是源页面坐标。
        # 无法在最小字号下放下的译文交给待布局或人工编辑流程。
        ink=_ink_bounds(lines,metrics)
        if not ink.isEmpty() and ink.width()<=max_width and ink.height()<=max_height:
            return lines,size,ink.width(),ink.height(),(-ink.left(),-ink.top())
        if size<=minimum:raise ValueError('中文实际字形在可读最小字号下仍无法放入原位区域')
        size=max(minimum,size-1)


# 已保存项目的基线偏移属于不可信输入，必须在擦除前验证。
# 无显式偏移的旧布局保持历史路径，不强制重算而改变旧位置。
# 此检查只约束框内实际字形，页内、图框和源墨迹碰撞另行判断。
# 构造模型时验证了数值形状，不等于通过实际字体墨迹验证。
# 人工编辑返回前也调用同一方法，避免只验证输入框而漏掉 bearing。
def validate_text_origin(item):
    """Reject untrusted baseline offsets whose actual ink leaves its box.

    Called before painting any mask, and while validating saved projects.
    Legacy None offsets keep their established layout and validation path.
    """
    if item.text_origin is None:return
    origin=item.text_origin
    # 字号、原点及框坐标都必须为有限实数。
    # 最小字号仍独立检查，不能凭持久化 origin 绕过可读性门禁。
    # 无绘制行时不产生墨迹，不再请求空字形联合框。
    # 有行则使用实际当前应用字体重新度量，检测过期或篡改的紧框。
    values=(*origin,item.font_size,item.text_bounds.min_x,item.text_bounds.min_y,
            item.text_bounds.max_x,item.text_bounds.max_y)
    if (len(origin)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
            or item.font_size<6):
        raise ValueError('Invalid tight glyph origin or font')
    if not item.lines:return
    width=item.text_bounds.max_x-item.text_bounds.min_x
    height=item.text_bounds.max_y-item.text_bounds.min_y
    # 把局部墨迹平移到保存的基线原点后，与未旋转框宽高比较。
    # 只允许很小浮点容差，不能用大容差吞掉实际越界笔画。
    # 左右上下都需要包含，负 bearing 尤其容易越过左上边。
    # 验证成功后，旋转占用仍由 footprint 在页面空间复核。
    # 失败抛异常，由绘制器保留原图或由项目恢复拒绝坏项。
    ink=_ink_bounds(item.lines,QFontMetricsF(annotation_font(item.font_size))).translated(*origin)
    if (ink.isEmpty() or ink.left()<-.000001 or ink.top()<-.000001
            or ink.right()>width+.000001 or ink.bottom()>height+.000001):
        raise ValueError('Annotation glyph origin paints outside its final footprint')
