# mechanical_drawing_translator/safe_annotation_edit.py
"""Bounded Chinese edits reuse current evidence without expanding its cover."""
from dataclasses import replace
import math
from PySide6.QtCore import QPointF,Qt
from PySide6.QtGui import QBrush,QColor,QPainter,QPolygonF

from .annotation_models import MaskDecision
from .annotation_preview import BlankSpace,footprint
from .glyph_text_geometry import tight_text_fit,validate_text_origin
from .models import TextBounds


# 历史中文原位微调接口，当前正式双模式界面不新增第三模式入口。
# 调用者必须传当前自动重新生成的 item，旧项目保存 SAFE 不能代替证明。
# 编辑只改变中文几何，不增大原有蒙版，也不新增遮挡权限。
# 源锚点与译文来源保持不变，人工拖动不提升机器审核等级。
# 结果以新 item 返回，验证失败不会部分修改原对象或源图。
# 双语一般人工邻近调整走工作流的独立位置验证，不以此授权擦除。
def edit_chinese_item(engine,item,page,source_image,*,x,y,font_size,width,obstacles=()):
    """Fit fresh ink on a private copy with only already-proven masks applied.

    A SAFE flag is necessary but never creates new polygons. The caller must
    supply the freshly regenerated automatic item, rather than a saved mask.
    A two-source-unit movement budget limits disconnection from the original
    anchor; actual protected pixels and neighboring source boxes still veto.
    """
    # 只有当前已验证且已有完整绘制行的原位项才可微调。
    # 待布局项缺少安全底图，不允许靠人工输入框变成可擦英文。
    # 空图像直接拒绝，避免没有墨迹就被误当纯白背景。
    # 这只是必要条件，后面还需验证实际字形与保留工程墨迹。
    if (item.mask_decision!=MaskDecision.SAFE or not item.mask_polygons
        or item.placement!='inline' or not item.lines or source_image.isNull()):
        raise ValueError('中文调整需要当前已验证的安全遮挡；待布局项不能获得擦除权限')
    # 排除 bool、NaN 和无穷大，防止界面或外部项目传入无意义坐标。
    # 字号不能低于六或引擎更高下限，也不能超过引擎上限。
    # 输入 width 是排版可用宽度，必须为正，不是重新定义原图框。
    # 所有值仍为源单位，与当前屏幕缩放无关。
    values=(x,y,font_size,width)
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in values):
        raise ValueError('调整坐标、字号和宽度必须是有限数值')
    if not max(6.,engine.min_font_size)<=font_size<=engine.max_font_size or width<=0:
        raise ValueError('字号或宽度超出可读安全范围')
    # 原 source_polygons 的并集形成不可变英文锚点范围。
    # 不能用用户刚输入的框反过来定义锚点，否则可无限漂移。
    # 新左上相对当前自动位置最多两源单位，并保留很小浮点容差。
    # 此限制维持中文与原文字面关联，不允许移动到远端冒充原位。
    anchor=TextBounds(min(x for p in item.source_polygons for x,y in p),
        min(y for p in item.source_polygons for x,y in p),
        max(x for p in item.source_polygons for x,y in p),max(y for p in item.source_polygons for x,y in p))
    b=item.text_bounds
    if abs(x-b.min_x)>2.000001 or abs(y-b.min_y)>2.000001:
        raise ValueError('中文位置只能在自动原位锚点附近微调（最多2个源坐标单位）')
    # Right-angle rotation is supported by the shared painter. Oblique source
    # layouts remain under the established replacement policy, never new edits.
    angle=item.rotation_degrees%180
    if min(abs(angle),abs(angle-90),abs(angle-180))>.01:raise ValueError('斜向中文缺少安全原位编辑证据')
    # 90 度文字交换可用宽高，斜向文字缺完整原位证据故拒绝。
    # 锚点各方向仅有限扩展余量，不能把纸张空白当作任意替换范围。
    # 请求宽度先限制，再用实际字体墨迹拟合完整译文。
    # 紧框去掉字体留白，但不会降低可读字号或裁掉译文字面。
    max_width=(anchor.max_y-anchor.min_y if abs(angle-90)<.01 else anchor.max_x-anchor.min_x)+4
    max_height=(anchor.max_x-anchor.min_x if abs(angle-90)<.01 else anchor.max_y-anchor.min_y)+4
    if width>max_width:raise ValueError('宽度超出原英文锚点')
    lines,size,w,h,origin=tight_text_fit(engine,item.translated_text,width,max_height,font_size)
    bounds=TextBounds(x,y,x+w,y+h);box=footprint(bounds,item.rotation_degrees)
    # 旋转后的实际占用必须同时留在锚点邻域和纸张内。
    # 障碍代表其它源字、尺寸和已放中文，任何正面积相交都拒绝。
    # 检查未旋转 bounds 不足，90 度或非方框角落会落到不同区域。
    # 验证通过也不增加 mask_polygons，已有擦除范围保持原样。
    if (box.min_x<max(0,anchor.min_x-2) or box.min_y<max(0,anchor.min_y-2)
        or box.max_x>min(page.width,anchor.max_x+2) or box.max_y>min(page.height,anchor.max_y+2)):
        raise ValueError('中文实际字形超出源锚点或页面')
    if any(engine._overlap_area(box,other)>0 for other in obstacles):
        raise ValueError('中文实际字形与尺寸、线稿或其他文字相交')
    # 在私有图上只应用已有已证明蒙版，观察编辑后中文下方是否仍有墨迹。
    # 绘制先按源单位缩放到该副本像素，不能按屏幕坐标直接填框。
    # 无轮廓画笔避免边缘描线扩大蒙版范围。
    # 多边形点必须有效且有限，未知形状或 NaN 不进入绘制。
    # finally 结束 painter，失败也不污染源图存储。
    copy=source_image.copy();painter=QPainter(copy)
    try:
        painter.scale(copy.width()/page.width,copy.height()/page.height)
        painter.setPen(Qt.PenStyle.NoPen)
        fill=getattr(item,'mask_fill',(255,255,255)) or (255,255,255)
        painter.setBrush(QBrush(QColor(*fill)))
        for polygon in item.mask_polygons:
            if not polygon or any(not math.isfinite(v) for point in polygon for v in point):
                raise ValueError('遮挡多边形无效')
            painter.drawPolygon(QPolygonF([QPointF(px,py) for px,py in polygon]))
    finally:painter.end()
    # Colored OCR-derived canvases have no production erasure authority.
    # Retain the original white-background edit gate; stored SAFE/receipts do
    # not make a gray cover independently verified.
    # 保存的灰底或彩色画布没有当前独立擦除授权，不能靠 SAFE 标签编辑。
    # 白底也只消费原有多边形，不重新猜测哪个源笔画属于英文。
    # BlankSpace 检查新文字实际框，仍有暗线或未授权墨迹就拒绝。
    # 这防止把字从原蒙版内部移到旁边工程线之上。
    if fill!=(255,255,255):raise ValueError('灰色背景缺少独立覆盖授权，暂不支持中文编辑')
    blank=BlankSpace(copy,page)
    if not blank.clear(box):
        raise ValueError('中文实际字形覆盖范围仍有未授权墨迹或不支持的背景')
    # 只在全部验证后一次性构造新几何、行、字号与基线原点。
    # 清理旧微调诊断再写本次诊断，保留其它来源和风险警告。
    # fallback 清空，避免后续安全降级突然回到不对应的新旧混合位置。
    # 最后用同一 validate_text_origin 复核实际字形不越紧框。
    # 返回值还需调用方保存人工位置锁，不能在自动重排时静默覆盖。
    edited=replace(item,text_bounds=bounds,font_size=size,lines=lines,text_origin=origin,
        fallback_text_bounds=None,warnings=(*tuple(w for w in item.warnings if not w.startswith('manual_chinese_edit:')),
        'manual_chinese_edit:保留源锚点和现有遮挡，实际字形已重新验证'))
    validate_text_origin(edited)
    return edited
