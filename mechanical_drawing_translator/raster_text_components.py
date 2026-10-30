# mechanical_drawing_translator/raster_text_components.py
"""Projection-aligned glyph evidence, with explicit protected-line accounting.

This normalises right-angle text, not the immutable OCR coordinates. Only
complete high-confidence alphabetic regions are eligible. Every foreground
component is either a proved exterior long stroke or belongs to one expected
glyph column. Unknown components and text touching engineering ink reject the
region. This is OCR-derived evidence, not independent semantic certification.
"""
import re
from math import floor,ceil
import cv2
import numpy as np


# 该辅助函数只解释裁剪边上的长直工程线，线本身仍受保护。
# 在 OCR 框外不等于无关，也可能是被裁断的字母或点号。
# 必须整体在 core 外且看到足够长的原图连续延伸，才可归为几何。
# 投影组件算法是研究证据，当前产品不能据它授权自动擦除扫描字。
def _exterior_line_component(component,core,pixels,x0,y0,rotation,nominal):
    """Prove a cropped border by observing its straight continuation in source.

    Being outside an OCR rectangle alone proves nothing: it could be a dot or
    a clipped letter. Require a thin component wholly outside that rectangle
    AND an uninterrupted source stroke at least three text heights long. The
    component remains protected ink; the existing glyph masks and one-pixel
    rings must still be disjoint from it.
    """
    if core[component].any():return False
    # 组件图曾按文字直角方向旋正，检查原图延伸前先旋回。
    # 用原始裁剪 x0/y0 偏移映射到整幅源栅格。
    # 只在检测副本做旋转，不写回 OCR bounds 或 polygon。
    # 宽高是原图方向的组件尺寸，后面据此识别水平或垂直细线。
    original=np.rot90(component,-rotation)
    ys,xs=np.nonzero(original)
    left=x0+int(xs.min());right=x0+int(xs.max())+1
    top=y0+int(ys.min());bottom=y0+int(ys.max())+1
    width=right-left;height=bottom-top

    # 从实际墨迹点向两边扩展，直到遇到断口或图像边缘。
    # 不是只看整行墨迹总数，断开的多段不能拼成连续工程线。
    # 没有起始墨迹直接零长度，避免把纯白区域解释成延伸。
    def observed_run(values,index):
        if not values[index]:return 0
        start=index;end=index+1
        while start>0 and values[start-1]:start-=1
        while end<len(values) and values[end]:end+=1
        return end-start

    # 线宽相对文本高度必须很薄，连续长度至少三倍文字高度。
    # 横、竖分别按原图方向检查，不能只用裁剪框的宽高比推断。
    # 无法证明长延伸时拒绝该边缘组件，不剪断未知笔画来拼字。
    if height<nominal*.18 and width>=height:
        for y in range(top,bottom):
            if observed_run((pixels[y]<255).any(1),left)>=nominal*3:return True
    if width<nominal*.18 and height>=width:
        for x in range(left,right):
            if observed_run((pixels[:,x]<255).any(1),top)>=nominal*3:return True
    return False


def projected_glyph_polygons(entry,obstacles,pixels,sx,sy):
    # 共享框或不完整 SourceSpan 无法逐字解释原始区域，先拒绝。
    # 用区域 ID、起止字符及数量同时验证全覆盖，避免重复 span 伪装完整。
    # 数字、公差、直径、角度及星号属于高风险混合字面，不走字形研究路径。
    # 每个原始区域独立解释，最后返回多区域多边形元组。
    if entry.coordinate_precision=='shared_region_bounds':return 'partial_source_span'
    if entry.spans:
        expected={(r.id,0,len(r.source_text))for r in entry.regions}
        actual={(s.region_id,s.start,s.end)for s in entry.spans}
        if actual!=expected or len(entry.spans)!=len(expected):return 'incomplete_region_span_coverage'
    if re.search(r'[0-9±Ø⌀°*]',entry.source_text):return 'mixed_engineering_literal'
    polygons=[]
    for region in entry.regions:
        text=region.source_text.strip()
        if not re.fullmatch(r'[A-Za-z]+(?:[ \t]+[A-Za-z]+)*[.,:]?',text):
            return 'unsupported_region_text_pattern'
        if region.source!='ocr' or region.confidence is None or region.confidence<.995:
            return 'insufficient_ocr_confidence'
        # 只归一化 0/90/180/270 度，斜向扫描文字不做插值旋正猜测。
        # 插值可能改变淡线和连通域，故未知斜角宁可拒绝。
        # 旋转容差只是浮点角度误差，不允许任意角文字。
        # 其它原区域与源框交叠时拒绝，即使文本表面看起来像独立英文。
        angle=region.rotation_degrees%360;k=round(angle/90)%4
        if min(abs(angle-k*90),abs(angle-k*90-360))>.1:return 'nonorthogonal_region'
        b=region.bounds
        if any(o.id not in entry.region_ids and min(b.max_x,o.bounds.max_x)>max(b.min_x,o.bounds.min_x)
               and min(b.max_y,o.bounds.max_y)>max(b.min_y,o.bounds.min_y)for o in obstacles):
            return 'neighbor_source_collision'
        # A narrow exterior ring accommodates subpixel OCR rounding while
        # requiring complete glyphs; it never expands the source coordinates.
        # 两像素 halo 用来观察字形完整性与周边墨迹，不扩写原 OCR 坐标。
        # 触纸边的 halo 不完整，无法判断被裁剪部分，因此拒绝。
        # 所有非白通道都算墨迹，浅灰短线也不能被阈值滤掉。
        # 旋正后的 expected 去掉排版空白，保留句末标点参与组件计数。
        x0=max(0,floor(b.min_x*sx)-2);y0=max(0,floor(b.min_y*sy)-2)
        x1=min(pixels.shape[1],ceil(b.max_x*sx)+2);y1=min(pixels.shape[0],ceil(b.max_y*sy)+2)
        if x0==0 or y0==0 or x1==pixels.shape[1] or y1==pixels.shape[0]:return 'region_near_page_edge'
        raw=(pixels[y0:y1,x0:x1]<255).any(2).astype(np.uint8)
        ink=np.rot90(raw,k).copy();expected=text.replace(' ','').replace('\t','')
        # Preserve original OCR extent independently of the two-pixel crop halo.
        # core 单独表示真正原框范围，halo 不算原字面证据。
        # 旋转 core 与 ink 采用同样 k，保证外部组件归属比较一致。
        # nominal 使用随文字方向切换的原框厚度，避免 90 度字高取错。
        # 分辨率低于可判断范围时拒绝，不把粗像素当可靠字形。
        core=np.zeros_like(raw)
        core[floor(b.min_y*sy)-y0:ceil(b.max_y*sy)-y0,
             floor(b.min_x*sx)-x0:ceil(b.max_x*sx)-x0]=1
        core=np.rot90(core,k).copy()
        nominal=(b.max_x-b.min_x)*sx if k%2 else (b.max_y-b.min_y)*sy
        if nominal<7:return 'insufficient_glyph_resolution'
        # 完整连通域先分为候选字形墨迹与受保护工程线。
        # 足够长且薄的线从 keep 移除，但保留在 protected 防止被字框吞掉。
        # 触裁剪边组件必须另证原图延伸，否则整个区域拒绝。
        # 不会把附着字母从工程线切开重建，也不推断丢失背景。
        count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
        keep=ink.copy();protected=np.zeros_like(ink)
        for label,(x,y,w,h,area)in enumerate(stats[1:],1):
            # Only strokes well longer than a text-line height may be classed
            # as geometry. Attached letters make the component thick and are
            # rejected rather than reconstructed or severed from that stroke.
            long_line=(w>nominal*3 and h<nominal*.18)or(h>nominal*3 and w<nominal*.18)
            if long_line:
                keep[labels==label]=0;protected[labels==label]=1
            elif x==0 or y==0 or x+w==ink.shape[1] or y+h==ink.shape[0]:
                component=labels==label
                if not _exterior_line_component(component,core,pixels,x0,y0,k,nominal):
                    return 'unexplained_edge_component'
                keep[component]=0;protected[component]=1
        # 按旋正图的非空列形成投影字槽，允许一个字有内部断开笔画。
        # 投影槽数必须与预期字符数一致，不忽略多余标记或漏失字符。
        # 没有墨迹说明候选关联失败，不生成空蒙版通过。
        # 此计数仍不证明每个字槽内的笔画只属于该字母。
        columns=np.flatnonzero(keep.any(0))
        if not len(columns):return 'no_glyph_components'
        starts=[int(columns[0])];ends=[]
        for left,right in zip(columns,columns[1:]):
            if right-left>1:ends.append(int(left)+1);starts.append(int(right))
        ends.append(int(columns[-1])+1)
        if len(starts)!=len(expected):return 'projection_glyph_count_mismatch'
        glyphs=[];baselines=[];letter_heights=[]
        for character,left,right in zip(expected,starts,ends):
            ys,_=np.nonzero(keep[:,left:right]);top=int(ys.min());bottom=int(ys.max())+1
            h=bottom-top;w=right-left;area=int(keep[top:bottom,left:right].sum())
            # 字符槽内部再次做连通域检查，普通字母不得有无法解释的碎片。
            # i/j 可有点与主干两个组件，但小点须在上且面积受限。
            # 字形高度、宽度和填充率共同拒绝实心方块或异常长线。
            # I/i/l 的细竖形状有特例，不能把所有高填充率都误判为矩形。
            # 基线单独收集，最后检查整行字形是否保持合理对齐。
            if character.isalpha():
                parts,_,part_stats,_=cv2.connectedComponentsWithStats(keep[top:bottom,left:right],8)
                if parts-1>1 and character not in 'ij':return 'unexplained_glyph_fragment'
                if parts-1>2:return 'unexplained_glyph_fragment'
                if parts-1==2:
                    small,large=sorted(part_stats[1:],key=lambda row:row[4])
                    if small[4]>large[4]*.45 or small[1]+small[3]>large[1]:
                        return 'unexplained_glyph_fragment'
                if h<nominal*.40 or h>nominal*1.25 or w>nominal*1.7 or w<max(1,nominal*.04):
                    return 'glyph_shape_inconsistent'
                if area/(w*h)>.93 and character not in 'Iil':return 'solid_unknown_component'
                baselines.append(bottom);letter_heights.append(h)
            else:
                parts,_,part_stats,_=cv2.connectedComponentsWithStats(keep[top:bottom,left:right],8)
                # 标点有自己的组件数量和位置条件，不能按字母高度判断。
                # 冒号必须是上下分离且面积相近的两点，未知几何点不随便忽略。
                # 标点宽高受文本名义高度约束，过大块视作无法解释。
                # 保护线若落入任何字框内，即使它已从 keep 排除也拒绝。
                if parts-1!=(2 if character==':'else 1):return 'unexplained_punctuation_component'
                if character==':':
                    first,second=sorted(part_stats[1:],key=lambda row:row[1])
                    if first[1]+first[3]>=second[1] or max(first[4],second[4])>3*min(first[4],second[4]):
                        return 'punctuation_shape_inconsistent'
                if w>nominal*.40 or h>nominal*(.8 if character==':'else .4):
                    return 'punctuation_shape_inconsistent'
            # A bounding mask may not enclose even a protected pale line.
            if protected[top:bottom,left:right].any():return 'glyph_intersects_protected_line'
            glyphs.append((left,top,right,bottom))
        # 行基线不一致可能表示错配、混入其它标记或 OCR 框不完整。
        # 每个字框周围一像素再次查墨迹，避免与线或未知组件接触。
        # 框内空洞也不能含 protected 线，逐字矩形不会越过保护对象。
        # 这些几何条件仍不能排除与字母连通的短线，所以生产资格已撤回。
        if baselines and max(baselines)-min(baselines)>max(2,max(letter_heights)*.35):
            return 'glyph_baseline_inconsistent'
        for left,top,right,bottom in glyphs:
            # Account for all other ink around this exact glyph. Counter/hole
            # pixels inside the glyph rectangle are already proved empty of
            # protected strokes; no full OCR-box mask is ever returned.
            ring=ink[max(0,top-1):bottom+1,max(0,left-1):right+1].copy()
            ring[1:1+bottom-top,1:1+right-left]=0
            if ring.any():return 'glyph_touches_unknown_or_geometry'
            # 把字槽四角从旋正局部图映回原裁剪方向，再加整页偏移。
            # 每个 k 使用对应反变换，输出始终为原源坐标单位。
            # 四分之一像素边留量照样需要下游证据检验，不能视为任意扩张。
            # 没有写回 region.polygon，原 OCR 几何保持可追溯。
            # 当前多边形是历史候选记录，不能塞进 native_verified 冒充独立证据。
            points=[]
            for u,v in ((left-.25,top-.25),(right+.25,top-.25),(right+.25,bottom+.25),(left-.25,bottom+.25)):
                if k==0:x,y=u,v
                elif k==1:x,y=raw.shape[1]-v,u
                elif k==2:x,y=raw.shape[1]-u,raw.shape[0]-v
                else:x,y=v,raw.shape[0]-u
                points.append(((x0+x)/sx,(y0+y)/sy))
            polygons.append(tuple(points))
    return tuple(polygons)
