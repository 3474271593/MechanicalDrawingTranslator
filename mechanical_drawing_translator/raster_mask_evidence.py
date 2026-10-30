# mechanical_drawing_translator/raster_mask_evidence.py
"""Strict, confidence-derived evidence for isolated uppercase raster labels.

This is weaker than independently hiding native PDF text objects. It accepts
only fully isolated, one-component-per-letter words, accounting for every
nonwhite pixel in a white surrounding ring. It never reconstructs background,
inpaints lines, or mutates the original page. Ambiguity keeps the English.
"""
import re
from math import ceil, floor
import cv2
import numpy as np
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtCore import QRectF


# 这是历史栅格候选研究接口，当前生产任务不调用它来授权擦除。
# 即便返回 accepted，也只能解释组件候选，不能证明所有笔画属于英文。
# 与短工程线相连的字形可能通过高置信和连通域形状门禁。
# 当前产品仍须完整原生对象去字背景，不能把本返回集合当 native_verified。
# 函数只在副本上构造背景，从不保存或改写源图文件。
def isolated_raster_evidence(image, page, entries, obstacles, background=None):
    """Return private background, accepted groups, reasons, and exact glyph masks."""
    if image.isNull():
        return background, frozenset(), {}, {}
    # 透明图用于分析前先合成白底，避免透明黑 RGB 被当实际纸墨。
    # 这属于研究检测视图，不是 RGBA 保真导出的处理流程。
    # Qt 行尾可能有填充字节，读取 numpy 前按 bytesPerLine 分行裁切。
    # 源坐标与分析像素比例分别计算，保留非等比取整的真实映射。
    if image.hasAlphaChannel():
        paper=QImage(image.size(),QImage.Format.Format_RGB32);paper.fill(QColor('white'))
        painter=QPainter(paper);painter.drawImage(0,0,image);painter.end();image=paper
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    pixels = np.frombuffer(rgb.constBits(), dtype=np.uint8, count=rgb.sizeInBytes())
    pixels = pixels.reshape(rgb.height(), rgb.bytesPerLine())[:, :rgb.width()*3].reshape(rgb.height(), rgb.width(), 3)
    sx = rgb.width()/page.width; sy = rgb.height()/page.height
    accepted = []; reasons = {}; masks = {}
    # 优先尝试旧单组件词规则，再以投影字形候选补充。
    # 拒绝原因同时保留旧路径和新路径，避免只有失败布尔值难以复核。
    # 多边形返回值与字符串诊断严格区分，不把非空原因串当成功。
    # 候选掩码按 group 保存，不能丢掉原区域关联。
    for e in entries:
        evidence = _inspect(e, obstacles, pixels, sx, sy)
        if isinstance(evidence,str):
            from .raster_text_components import projected_glyph_polygons
            projected=projected_glyph_polygons(e,obstacles,pixels,sx,sy)
            if not isinstance(projected,str):
                evidence=projected
                reasons[e.group_id]='accepted_projection_components_v2'
            else:
                reasons[e.group_id]='legacy:'+evidence+';projection:'+projected
        if isinstance(evidence,str): reasons.setdefault(e.group_id,evidence)
        else: accepted.append(e); masks[e.group_id] = evidence
    if not accepted: return background, frozenset(), reasons, masks
    # 只在私有副本上填候选白矩形，源图像对象保持不变。
    # 已有背景与原图可能分辨率不同，绘制时重新按页面单位换算。
    # 返回的背景是按 OCR 假设合成的，不能冒充原生对象独立去字背景。
    # 调用方必须维持这种证据等级差异，不能只看背景外观是白的。
    result = background.copy() if background is not None and not background.isNull() else image.copy()
    painter = QPainter(result)
    bx = result.width()/page.width; by = result.height()/page.height
    for e in accepted:
        for poly in masks[e.group_id]:
            x0,y0=poly[0];x1,y1=poly[2]
            painter.fillRect(QRectF(x0*bx,y0*by,(x1-x0)*bx,(y1-y0)*by),QColor('white'))
    painter.end()
    return result, frozenset(e.group_id for e in accepted), reasons, masks


def _inspect(e, obstacles, pixels, sx, sy):
    # 旧规则只针对单原区域、完整独立框和有限长度大写英文词。
    # 复杂句、编号、数字与符号不进入这套研究候选。
    # 限制文本类别降低假阳性，但不提供工程线语义保证。
    # 共享坐标意味着不能确定完整字形范围，直接拒绝。
    if (len(e.regions) != 1 or e.coordinate_precision == 'shared_region_bounds'
            or not re.fullmatch(r'[A-Z]{3,}(?: [A-Z]{2,})*', e.source_text)
            or len(e.source_text.replace(' ', '')) > 60):
        return 'not_isolated_uppercase_word'
    # 高置信门槛只是候选门禁，不能把置信度解释为擦线风险概率。
    # 只支持无旋转的轴对齐矩形，倾斜或非矩形字框交给其它研究路径。
    # 多边形需与 bounds 的四角对应，避免裁剪或透视形变被忽略。
    # 任何其它原区域与目标框正交叠都拒绝，不凭字面猜测它无关。
    r = e.regions[0]
    if r.source != 'ocr' or r.confidence is None or r.confidence < .995:
        return 'insufficient_ocr_confidence'
    if abs(r.rotation_degrees % 360) > .01:
        return 'rotated_raster_word'
    b = e.bounds
    expected = ((b.min_x,b.min_y),(b.max_x,b.min_y),(b.max_x,b.max_y),(b.min_x,b.max_y))
    if r.polygon and (len(r.polygon) != 4 or any(abs(x-u) > .05 or abs(y-v) > .05
                                                  for (x,y),(u,v) in zip(r.polygon,expected))):
        return 'nonrectangular_raster_word'
    if any(o.id not in e.region_ids and min(b.max_x,o.bounds.max_x)>max(b.min_x,o.bounds.min_x)
           and min(b.max_y,o.bounds.max_y)>max(b.min_y,o.bounds.min_y) for o in obstacles):
        return 'neighbor_source_collision'
    # 左上向下、右下向上取整，完整包含可能的字形抗锯齿。
    # 周围保留一像素环，不擅自扩为多 point 的整块白区。
    # 边缘和高度不足无法形成完整观测，必须返回不支持。
    # 环带纯白采用三通道精确判断，淡线与噪声都不会被忽略。
    x0=floor(b.min_x*sx); y0=floor(b.min_y*sy); x1=ceil(b.max_x*sx); y1=ceil(b.max_y*sy)
    # Pixel-derived masks need one surrounding blank pixel, not a multi-point
    # white area extending into an unrelated title-cell border.
    px=py=1
    height,width=pixels.shape[:2]
    if x0-px < 0 or y0-py < 0 or x1+px > width or y1+py > height or y1-y0 < 10:
        return 'insufficient_raster_extent'
    # Exact white is deliberate: scans with nonuniform paper, pale lines,
    # antialiasing reaching the boundary, or unknown marks remain unsupported.
    roi = (pixels[y0-py:y1+py,x0-px:x1+px] < 255).any(axis=2)
    ring = roi.copy(); ring[py:py+y1-y0,px:px+x1-x0] = False
    if ring.any(): return 'nonwhite_background_ring'
    ink = roi[py:py+y1-y0,px:px+x1-x0].astype(np.uint8)
    # 组件数必须与去空格后的字母数相同，额外点和线会导致拒绝。
    # 连通域统计还检查高度、宽高比、填充率与基线一致性。
    # 这些仅排除明显异常，附着短线仍可能混在一个字母组件里。
    # 不能把组件数量相同包装成字母形状已逐项语义识别。
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if count-1 != len(e.source_text.replace(' ', '')):
        return 'unaccounted_raster_components'
    glyphs = stats[1:]; max_height = max(glyphs[:,3], default=0)
    if max_height < 7: return 'insufficient_glyph_resolution'
    baseline = glyphs[:,1]+glyphs[:,3]
    for x,y,w,h,area in glyphs:
        if (x <= 0 or y <= 0 or x+w >= ink.shape[1] or y+h >= ink.shape[0]
                or h < max_height*.70 or h > (y1-y0)*.98
                or w < h*.12 or w > h*1.4 or not .07 < area/(w*h) < .85):
            return 'ambiguous_glyph_or_line_component'
    if baseline.max()-baseline.min() > max(2, max_height*.12):
        return 'inconsistent_glyph_baseline'
    # 输出逐组件紧框，而非把整个 OCR 文本矩形涂白。
    # 每个字框周围再次检查邻居墨迹，避免框内空洞吞掉旁边标记。
    # 四分之一像素留量只为抗锯齿边缘，回传时仍换算为源单位。
    # 当前安全边界已撤回此类候选的生产擦除资格，保留算法便于复核历史。
    polygons=[]
    for x,y,w,h,_ in glyphs:
        # Check the complete bounding rectangle, including counters and gaps,
        # against all other components. A nearby mark must not be whitened.
        around=ink[max(0,y-1):min(ink.shape[0],y+h+1),max(0,x-1):min(ink.shape[1],x+w+1)].copy()
        around[1:1+h,1:1+w]=0
        if around.any():return 'nonwhite_glyph_ring'
        left=(x0+x-.25)/sx;top=(y0+y-.25)/sy
        right=(x0+x+w+.25)/sx;bottom=(y0+y+h+.25)/sy
        polygons.append(((left,top),(right,top),(right,bottom),(left,bottom)))
    return tuple(polygons)
