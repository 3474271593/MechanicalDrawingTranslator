# mechanical_drawing_translator/drawing_frame_detection.py
"""Evidence-bearing raster frames; preserve strict frames before skew recovery.

No page-specific coordinates, OCR identities, or PDF-paper fallback are used.
A missing side stays unknown. New bounds are inscribed inside four measured
near-axis sides so scan skew cannot permit annotation beyond the observed line.
"""
from dataclasses import dataclass
from itertools import product
import cv2
import numpy as np
from .models import TextBounds


# 图框必须有实际四边观测，不能从纸张尺寸或 OCR 范围推断。
# 未知时 bounds 为 None，并保存明确的拒绝原因。
# confidence 是该几何检测路径的固定诊断分值，不是专业验收概率。
# 文档和页身份随证据保存，防止把上一页图框贴到本页。
# 证据只用于限制标注范围，不用于删除任何工程线或背景。
# 不可变记录使预览和持久化能引用同一个测得边界。
@dataclass(frozen=True)
class FrameEvidence:
    bounds: TextBounds | None
    confidence: float
    method: str
    evidence: tuple[str, ...]
    rejections: tuple[str, ...]
    document_id: str
    page_index: int


def _intersection(a, b):
    """Intersect homogeneous lines, refusing parallel or degenerate input."""
    # 用齐次直线叉积求交点，不先把近垂直线写成斜率形式。
    # 齐次分母接近零意味着平行或退化，不能把无限远点当角点。
    # 上层遇到 None 会跳过该四边组合，保持未知而非强行闭合。
    p = np.cross(a, b)
    return None if abs(p[2]) < 1e-8 else p[:2] / p[2]


def _line_support(ink, a, b, radius=3):
    """Observed evidence along the whole proposed side, including gaps."""
    # 沿整条候选边按近像素距离取样，包含 Hough 片段间的缺口。
    # 邻域半径容忍扫描线厚与轻微拟合误差，不补画缺失边。
    # 每个样点必须在原始墨迹里找到支撑，越界点不会被当作观测。
    # 返回平均值与最差局部支撑，避免半条实线掩盖另外半条缺失。
    # 四个分段及两个端部同时检查，角点附近也需有实际线迹。
    n = max(2, int(np.linalg.norm(b-a)))
    xy = np.rint(a + np.linspace(0,1,n)[:,None]*(b-a)).astype(int)
    h,w=ink.shape; present=np.zeros(n,dtype=bool)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            x=xy[:,0]+dx;y=xy[:,1]+dy
            ok=(x>=0)&(x<w)&(y>=0)&(y<h)
            present[ok] |= ink[y[ok],x[ok]] != 0
    # Aggregate support alone can conceal an unobserved half-side. All four
    # quarters need independent observations, not just a high global average.
    quarters=[float(v.mean()) for v in np.array_split(present,4)]
    end=max(2,int(n*.075))
    local_support=quarters+[float(present[:end].mean()),float(present[-end:].mean())]
    return float(present.mean()), min(local_support)


def _skew_frame(pixels, page):
    # 保留所有非纯白通道，淡扫描框也参与几何检测。
    # 这个二值图只是检测副本，原 RGB 样本不被改动。
    # 形态学开运算先降低字符短笔画影响，再以 Hough 提长线候选。
    # 核长较短以容忍小倾斜，不靠长水平核把斜边整体滤掉。
    # Hough 的角分辨率和断口容忍只提出候选，最终还需原墨迹复验。
    h,w=pixels.shape[:2]
    ink=((pixels<255).any(axis=2).astype(np.uint8))*255
    # Remove letter strokes before Hough. A short opening tolerates modest
    # scan inclination, unlike the previous 40%-page horizontal opening.
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,20),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((20,1),np.uint8))
    sparse=cv2.bitwise_or(horizontal,vertical)
    proposed=cv2.HoughLinesP(sparse,1,np.pi/1800,100,
                            minLineLength=min(h,w)*.45,maxLineGap=25)
    # 未观测到长边时保守返回未知，不能退回媒体纸张冒充图框。
    # 候选分成上下左右，且要求边位于靠近页边的合理带内。
    # 接近水平或垂直的斜率范围很窄，任意零件斜边不进入四框求解。
    # 长度相对整页约束，避免小标题格或零件矩形被当外框。
    if proposed is None:return None,(),('no_long_side_segments',)
    sides={'top':[],'bottom':[],'left':[],'right':[]}
    for coords in proposed.reshape(-1,4):
        a=np.array(coords[:2],dtype=float);b=np.array(coords[2:],dtype=float)
        dx,dy=b-a;length=float(np.linalg.norm(b-a));center=(a+b)/2
        if abs(dy)<abs(dx)*.035 and abs(dx)>=w*.60:
            side='top' if center[1]<h*.15 else 'bottom' if center[1]>h*.85 else None
        elif abs(dx)<abs(dy)*.035 and abs(dy)>=h*.60:
            side='left' if center[0]<w*.15 else 'right' if center[0]>w*.85 else None
        else:continue
        # A raster edge is not a drawing boundary, even when the renderer
        # leaves a one-pixel transparent/black strip at the media edge.
        # 贴着栅格最边缘的线可能只是渲染黑边或透明边，不算图框。
        # 归一化齐次直线系数，使交点计算不受片段长度尺度干扰。
        # 每个方向保留最长的有限候选，重复检出同一边不增加可信度。
        # 四个方向缺任意一个都明确列出 unobserved，不能用三边外推第四边。
        if min(center[0],w-1-center[0],center[1],h-1-center[1])<3:continue
        if side:
            line=np.cross(np.r_[a,1.],np.r_[b,1.]);line/=np.linalg.norm(line[:2])
            sides[side].append((length,line,a,b))
    missing=tuple('unobserved_'+k for k,v in sides.items() if not v)
    if missing:return None,(),missing
    for key in sides:
        # Longest representatives are deterministic and bounded; line duplicates
        # do not count as independent sides or raise the confidence.
        sides[key]=sorted(sides[key],key=lambda x:-x[0])[:6]
    candidates=[]
    for top,bottom,left,right in product(*(sides[k] for k in ('top','bottom','left','right'))):
        corners=[_intersection(top[1],left[1]),_intersection(top[1],right[1]),
                 _intersection(bottom[1],right[1]),_intersection(bottom[1],left[1])]
        if any(p is None for p in corners):continue
        tl,tr,br,bl=corners
        # 四条观测直线先形成四个真实交点，再取其内接轴对齐框。
        # 左上用较大值、右下用较小值，保证斜框每一边都在外侧。
        # 额外四像素内缩避开线厚与抗锯齿，绝不外扩成注释区。
        # 面积与宽高门禁要求它像整图外框，而不是内部视图窗口。
        # 角点平行退化或不在页内的组合直接拒绝。
        x0=max(tl[0],bl[0])+4;x1=min(tr[0],br[0])-4
        y0=max(tl[1],tr[1])+4;y1=min(bl[1],br[1])-4
        if not (0<x0<x1<w and 0<y0<y1<h and x1-x0>w*.75 and y1-y0>h*.75):continue
        observed=[];valid=True
        for segment,a,b in ((top,tl,tr),(bottom,bl,br),(left,tl,bl),(right,tr,br)):
            # Hough may split one observed side into several segments. Require
            # raw ink at both corner neighborhoods and in all four quarters,
            # instead of treating a fragment endpoint as the end of the frame.
            # Hough 可能只给出一段长边，不能把片段端点当整框终点。
            # 在推算的两角之间重新观测整边，平均支撑和局部最低支撑都要过线。
            # 边的四个分区与两端若缺证据，宁可不认图框。
            # 这些门禁降低完整矩形零件轮廓造成的假阳性。
            average,quarter=_line_support(ink,a,b)
            if average<.80 or quarter<.60:valid=False;break
            observed.append(average)
        if not valid:continue
        # Reject view/part boxes which leave significant measured ink outside.
        # 外框应包住几乎全部观测内容，框外大量墨迹提示选成了局部视图框。
        # 按斜四边形生成内部掩码，而不是按内接矩形判断外围内容。
        # 窄膨胀只容忍边中心线两侧的厚度，不能吞掉大块框外注释。
        # 框外墨迹比例用全页实际墨迹作分母，空白纸面积不会稀释风险。
        # 超过阈值的候选不返回，用户需要知道图框仍不可靠。
        mask=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(mask,np.rint(corners).astype(np.int32),1)
        # The measured centerline has ink on both sides; its antialiasing and
        # thickness are border evidence, not exterior content. Ignore only a
        # narrow raster tolerance, never a large surrounding annotation area.
        mask=cv2.dilate(mask,np.ones((11,11),np.uint8))
        exterior=np.count_nonzero((ink!=0)&(mask==0))
        ratio=exterior/max(1,np.count_nonzero(ink))
        if ratio>.015:continue
        area=(x1-x0)*(y1-y0)
        candidates.append((area,x0,y0,x1,y1,tuple(observed),ratio,corners))
    if not candidates:return None,(),('four_side_geometry_or_content_guard_failed',)
    # 合法候选取内接面积最大者，保持选择规则确定。
    # 像素坐标分别按 X/Y 比例转换回源页单位。
    # 证据存四边支撑、外围比例与栅格角点，便于复核失败与成功。
    # 返回内接边界不会改动 page 的原始宽高。
    _,x0,y0,x1,y1,support,outside,corners=max(candidates,key=lambda c:c[0])
    bounds=TextBounds(x0*page.width/w,y0*page.height/h,x1*page.width/w,y1*page.height/h)
    evidence=('four_independent_observed_sides',
              'four_intersections_with_observed_corner_neighborhoods',
              'quarterwise_side_support',
              'side_support='+','.join(f'{v:.4f}' for v in support),
              f'exterior_ink_ratio={outside:.6f}',
              'raster_corners='+repr([[round(float(v),3) for v in p] for p in corners]),
              'bounds_inscribed_inside_all_observed_sides')
    return bounds,evidence,()


def detect_frame_evidence(pixels, page):
    """Return an observed frame in source units, or explicit uncertainty.

    This accepts an immutable RGB raster already composited onto white, as
    BlankSpace does. It never changes the source pixels or page geometry.
    Existing reliable strict bounds win unchanged over new evidence paths.
    """
    # 入口严格接收三通道 RGB 检测图，透明白纸合成由 BlankSpace 完成。
    # 过小栅格无法可靠形成长线证据，返回未知而不是猜测。
    # 先尝试旧严格闭框，已可靠边界优先保持，避免无必要位置漂移。
    # 只有旧方法失败才启用斜框恢复，新增路径不替换已知可行结果。
    # 两个路径失败时 bounds 仍为 None，拒绝信息可随布局持久化。
    identity=(page.document_id,page.page_index)
    if not isinstance(pixels,np.ndarray) or pixels.ndim!=3 or pixels.shape[2]!=3:
        raise ValueError('frame detection requires RGB pixels')
    if min(pixels.shape[:2])<30:
        return FrameEvidence(None,0.,'unknown',(),('insufficient_raster_extent',),*identity)
    strict=_strict_closed_frame(pixels,page)
    if strict is not None:
        return FrameEvidence(strict,.98,'strict_closed_frame',
                             ('legacy_four_side_and_peripheral_or_title_grid_guards',),(),*identity)
    bounds,evidence,rejections=_skew_frame(pixels,page)
    return FrameEvidence(bounds,.93 if bounds is not None else 0.,
                         'observed_skew_four_sides' if bounds is not None else 'unknown',
                         evidence,rejections,*identity)


def _strict_closed_frame(pixels,page):
    """Recognize a large closed rectangular frame, never infer one from text.

    Morphological long strokes propose horizontal borders. Both vertical
    sides must independently cover at least 80% of the proposed height.
    Ambiguous/scanned broken frames fall back to conservative page bounds.
    Returned coordinates use the existing source units, not raster pixels.
    """
    import cv2
    # Match clear()'s any-visible-ink policy. Low-resolution antialiasing
    # can turn a faint but continuous scanned frame into RGB253; dropping
    # it changes drawing boundaries with zoom and permits outside labels.
    # Long strokes, four-sided support and title-grid guards still apply.
    ink=((pixels<255).any(axis=2).astype(np.uint8))*255
    height,width=ink.shape
    # 严格路径用可见墨迹范围估计水平长线长度。
    # 这使窄竖图嵌在宽横向 PDF 纸内时仍能找到真实框。
    # 完全白纸没有任何观测，不能被判为一个覆盖整页的闭框。
    # 连通域只负责提出横线，其它方向必须独立查墨迹支撑。
    ys,xs=np.nonzero(ink)
    if not len(xs):return None
    ink_left,ink_right=int(xs.min()),int(xs.max())
    ink_top,ink_bottom=int(ys.min()),int(ys.max())
    content_width=ink_right-ink_left+1
    # Propose strokes relative to visible ink as well as paper: portrait
    # scans are often embedded in a much wider landscape PDF sheet.
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(20,int(content_width*.40))),np.uint8))
    count,_,stats,_=cv2.connectedComponentsWithStats(horizontal)
    lines=[(x,y,w,h) for x,y,w,h,area in stats[1:] if w>=content_width*.40 and h<height*.025]
    candidates=[]
    for x0,y0,w0,h0 in lines:
        for x1,y1,w1,h1 in lines:
            # 上下候选需有足够高度间距并大体左右对齐。
            # 随后重新选择墨迹支撑最强的左右位置，容忍窄线厚和扫描偏差。
            # 两条竖边各自覆盖至少八成高度，横线长不能替代竖边证据。
            # 此门禁是闭合结构约束，不把文字排列当一条推测边。
            if y1-y0<height*.35 or min(w0,w1)<content_width*.75:continue
            if abs(x0-x1)>width*.04 or abs(x0+w0-x1-w1)>width*.04:continue
            left=max(x0,x1);right=min(x0+w0,x1+w1)-1
            # Small raster tolerance accommodates line thickness and antialiasing.
            def support(x):
                return ink[y0:y1+1,max(0,x-3):min(width,x+4)].any(axis=1).mean()
            left=max(range(max(0,min(x0,x1)-3),min(width,max(x0,x1)+5)),key=support)
            right=max(range(max(0,min(x0+w0,x1+w1)-5),min(width,max(x0+w0,x1+w1)+3)),key=support)
            if support(left)<.8 or support(right)<.8:continue
            # A part outline can itself be a perfect rectangle. Only a
            # large peripheral enclosure is credible as a drawing frame;
            # uncertain interior rectangles must not restrict annotations.
            # 纯矩形零件也可能四边闭合，因此必须再判断是否是外围大包围框。
            # 外围要求宽高、面积和位置共同成立，避免内部大零件误限注释区。
            # 较小内嵌框只有存在独立标题栏网格才可被接受。
            # 标题栏依据下部多条横分隔和内部竖分隔，不能只凭一个长矩形。
            # encloses_ink 要求它几乎包住全部实际内容，非任意局部表格。
            peripheral=not ((right-left)<width*.65 or (y1-y0)<height*.60
                or (right-left)*(y1-y0)<width*height*.45
                or left>width*.25 or y0>height*.25
                or right<width*.75 or y1<height*.75)
            # A smaller inset enclosure needs independent title-grid
            # evidence. A plain rectangular part is never enough. Require
            # the enclosure to contain essentially all ink, several lower
            # horizontal dividers and interior vertical title-cell splits.
            encloses_ink=(left<=ink_left+5 and right>=ink_right-5
                          and y0<=ink_top+5 and y1>=ink_bottom-5)
            lower=[(x,y,w,h) for x,y,w,h in lines
                   if y0+(y1-y0)*.70<y<y1-5
                   and x>=left-4 and x+w<=right+5]
            title_grid=False
            # 标题栏竖分隔须位于框内部，外框两侧不能重复算成分格。
            # 核长相对标题栏高度选择，短小字母竖笔不充当分格线。
            # 至少两个内部竖分隔与多个低位横分隔共同构成支持。
            # 不满足外围或标题栏两类证据的候选全部拒绝。
            if encloses_ink and len(lower)>=2:
                top=min(y for x,y,w,h in lower)
                vertical=cv2.morphologyEx(ink[top:y1+1,left:right+1],cv2.MORPH_OPEN,
                    np.ones((max(12,int((y1-top)*.50)),1),np.uint8))
                _,_,vstats,_=cv2.connectedComponentsWithStats(vertical)
                splits=[x for x,y,w,h,a in vstats[1:]
                        if 8<x<(right-left)-8 and h>=(y1-top)*.50]
                title_grid=len(splits)>=2
            if not peripheral and not title_grid:continue
            candidates.append(((right-left)*(y1-y0),left,y0+h0,right,y1))
    if not candidates:return None
    _,left,top,right,bottom=max(candidates)
    # 最终按源页单位输出，并向内留三像素安全距离。
    # X/Y 比例独立，兼容渲染像素取整造成的细微长宽差。
    # 没有候选时返回 None；上层负责记录 unknown，而非纸张替代框。
    # 历史 docstring 的回退字样不能理解为检测出可靠媒体边框。
    sx=page.width/width;sy=page.height/height
    return TextBounds((left+3)*sx,(top+3)*sy,(right-3)*sx,(bottom-3)*sy)
