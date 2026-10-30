# mechanical_drawing_translator/raster_pdf_evidence.py
"""Fixed PDF scales supplement strict raster evidence, never the display raster.

Both independent rasterizations must explain the same glyphs. A mask enlarged
to cover canonical antialiasing is accepted only when its extra support is white
or already-accounted glyph ink in BOTH renders, and intersects no foreign source
region. This is OCR-derived component evidence, not semantic certification.
"""
from dataclasses import dataclass
from math import ceil,floor,isfinite,sqrt
import re
from types import MappingProxyType
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage,QPainter,QColor


# 证据包显式区分 native_verified 与 raster_masks，不能混合授权级别。
# 原生集合供独立背景路径核验；栅格蒙版只保留研究诊断。
# 拒绝原因按 group 索引，widths 记录实际双尺度栅格尺寸。
# 字典复制后包为只读，防止任务返回后外部容器被继续修改。
# frozen 不会使 QImage 深度不可变，调用者仍应使用私有副本。
@dataclass(frozen=True)
class PreviewMaskEvidence:
    background: QImage
    native_verified: frozenset
    raster_masks: dict
    rejections: dict
    widths: tuple

    def __post_init__(self):
        object.__setattr__(self,'native_verified',frozenset(self.native_verified))
        object.__setattr__(self,'raster_masks',MappingProxyType(dict(self.raster_masks)))
        object.__setattr__(self,'rejections',MappingProxyType(dict(self.rejections)))
        object.__setattr__(self,'widths',tuple(self.widths))


# 工作图透明合成白纸，只用于像素研究，不改变原文件颜色合同。
# 逐行按 Qt stride 读取，避免把行对齐 padding 当像素。
# 末尾 copy 脱离局部 RGB QImage 的存储生命周期。
# 这个 RGB 数组不能替代 native_raster_pdf 的原始通道嵌入。
def _pixels(image):
    if image.hasAlphaChannel():
        paper=QImage(image.size(),QImage.Format.Format_RGB32);paper.fill(QColor('white'))
        painter=QPainter(paper);painter.drawImage(0,0,image);painter.end();image=paper
    rgb=image.convertToFormat(QImage.Format.Format_RGB888)
    return np.frombuffer(rgb.constBits(),dtype=np.uint8,count=rgb.sizeInBytes()).reshape(
        rgb.height(),rgb.bytesPerLine())[:,:rgb.width()*3].reshape(rgb.height(),rgb.width(),3).copy()


# 每个尺度独立运行投影候选，必要时再试旧孤立词规则。
# 不同渲染尺度不能共享一份组件结果，否则不构成一致性比较。
# 两个路径都是 OCR 派生证据，失败字符串须保留为拒绝原因。
# 双尺度一致只能支持几何稳定，不能证明组件内无短工程线。
def _evidence(entry,obstacles,pixels,sx,sy):
    from .raster_mask_evidence import _inspect
    from .raster_text_components import projected_glyph_polygons
    value=projected_glyph_polygons(entry,obstacles,pixels,sx,sy)
    return _inspect(entry,obstacles,pixels,sx,sy) if isinstance(value,str) else value


def _box(poly):
    return min(x for x,y in poly),min(y for x,y in poly),max(x for x,y in poly),max(y for x,y in poly)


# 外部源区域与蒙版正交叠就拒绝，即使其本身没有译文。
# 只排除本 entry 的原 region_ids，不能把所有英文都当可擦对象。
# 连续源坐标交叠与字体或渲染 DPI 无关，避免证据随显示缩放漂移。
def _foreign_overlap(box,entry,obstacles):
    l,t,r,b=box
    return any(o.id not in entry.region_ids and min(r,o.bounds.max_x)>max(l,o.bounds.min_x)
               and min(b,o.bounds.max_y)>max(t,o.bounds.min_y) for o in obstacles)


def _support_is_explained(box,pixels,proved,sx,sy):
    """Every added ink pixel must belong to a glyph proved at this exact scale."""
    l,t,r,b=box
    x0=max(0,floor(l*sx));y0=max(0,floor(t*sy))
    x1=min(pixels.shape[1],ceil(r*sx));y1=min(pixels.shape[0],ceil(b*sy))
    # 扩大支持框内每个非白像素起初都视为未知。
    # 只有落在本尺度已解释字形框中的部分才能从未知集合移除。
    # 另一尺度的组件不能直接解释本尺度墨迹，避免相互循环论证。
    # 此检查不推断或重建底纹，仍有未知像素就拒绝扩张范围。
    unknown=(pixels[y0:y1,x0:x1]<255).any(axis=2)
    for poly in proved:
        pl,pt,pr,pb=_box(poly)
        a=max(0,floor(pl*sx)-x0);c=min(x1-x0,ceil(pr*sx)-x0)
        d=max(0,floor(pt*sy)-y0);e=min(y1-y0,ceil(pb*sy)-y0)
        if a<c and d<e:unknown[d:e,a:c]=False
    return not unknown.any()


def _rotation_alignment(canonical,reference,entries,geometry):
    """Check non-target long-stroke landmarks in the same visible canvas.

    PDFium applies /Rotate during render. Matching dimensions alone cannot
    prove agreement with Qt; independently rendered non-target drawing strokes
    must agree in both directions, without an unexplained whole-pixel shift.
    """
    import cv2
    # 旋转 PDF 不只检查宽高，还需非目标工程长线作为独立对齐路标。
    # 把参考图缩到 canonical 尺寸后比较，目标字区先掩去避免自证。
    # 长线开运算降低文字抗锯齿差异影响，但不能制造不存在的路标。
    # 路标不足直接拒绝，不能只凭 Rotate 值信任不同渲染器坐标一致。
    height,width=canonical.shape[:2]
    reduced=cv2.resize(reference,(width,height),interpolation=cv2.INTER_AREA)
    a=(canonical<245).any(2).astype(np.uint8)
    b=(reduced<245).any(2).astype(np.uint8)
    sx=width/geometry.width;sy=height/geometry.height
    for entry in entries:
        box=entry.bounds
        x0=max(0,floor(box.min_x*sx)-2);x1=min(width,ceil(box.max_x*sx)+2)
        y0=max(0,floor(box.min_y*sy)-2);y1=min(height,ceil(box.max_y*sy)+2)
        a[y0:y1,x0:x1]=0;b[y0:y1,x0:x1]=0
    horizontal=np.ones((1,max(8,width//40)),np.uint8)
    vertical=np.ones((max(8,height//40),1),np.uint8)
    a=(cv2.morphologyEx(a,cv2.MORPH_OPEN,horizontal)|cv2.morphologyEx(a,cv2.MORPH_OPEN,vertical)).astype(bool)
    b=(cv2.morphologyEx(b,cv2.MORPH_OPEN,horizontal)|cv2.morphologyEx(b,cv2.MORPH_OPEN,vertical)).astype(bool)
    if min(int(a.sum()),int(b.sum()))<max(40,min(width,height)*2):return 'rotated_pdf_landmarks_unavailable'
    # 双向距离检查避免只覆盖一方部分线迹也得到高一致性。
    # 距离容忍面向渲染边缘差异，不允许整体一像素偏移被忽略。
    # 额外试探邻域平移，如果移位明显更匹配就说明原坐标尚未对齐。
    # 不在这里自动补偿偏移，否则会改写源坐标合同。
    to_b=cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,3)
    to_a=cv2.distanceTransform((~a).astype(np.uint8),cv2.DIST_L2,3)
    if (to_b[a]<=1.5).mean()<.97 or (to_a[b]<=1.5).mean()<.97:return 'rotated_pdf_landmarks_misaligned'
    zero=int((a&b).sum());best=zero
    for dy in (-2,-1,0,1,2):
        for dx in (-2,-1,0,1,2):
            if not dx and not dy:continue
            x0=max(0,dx);x1=min(width,width+dx);y0=max(0,dy);y1=min(height,height+dy)
            best=max(best,int((a[y0:y1,x0:x1]&b[y0-dy:y1-dy,x0-dx:x1-dx]).sum()))
    if best>zero*1.05 and best-zero>20:return 'rotated_pdf_landmarks_require_shift'
    return None


# 两尺度都须解释出非空且等量字形，才继续比较几何。
# 容差由两尺度最粗像素及小的源单位下限共同决定。
# 按最近矩形匹配而不是 OpenCV 标签顺序，兼容直角文字的组件编号差异。
# 匹配必须同时通过边距差和 IoU，防止错配到相邻字母。
# 任何一字失败拒绝整组，不把残缺掩码作为成功结果。
def _consistent_masks(entry,obstacles,first,second,first_pixels,second_pixels,geometry,canonical):
    if not first or len(first)!=len(second):return 'dual_scale_glyph_count_disagreement'
    sx1=first_pixels.shape[1]/geometry.width;sy1=first_pixels.shape[0]/geometry.height
    sx2=second_pixels.shape[1]/geometry.width;sy2=second_pixels.shape[0]/geometry.height
    tolerance=max(.25,1.5/min(sx1,sy1,sx2,sy2))
    boxes=[_box(p) for p in first];remaining=[_box(p) for p in second]
    dx=geometry.width/canonical.width();dy=geometry.height/canonical.height()
    masks=[]
    for a in sorted(boxes,key=lambda b:((b[1]+b[3])/2,(b[0]+b[2])/2)):
        # Nearest rectangle matching works for upright and 90-degree text;
        # relying on component label ordering would vary with rasterization.
        index=min(range(len(remaining)),key=lambda i:sum(abs(x-y) for x,y in zip(a,remaining[i])))
        b=remaining.pop(index)
        # IoU 在源单位计算，不受两份数组像素大小直接影响。
        # 不合法面积或几何差异过大都说明组件对应关系不足。
        # 每次匹配从 remaining 移除一个框，避免多个字重复消费同一证据。
        intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
        if max(abs(x-y) for x,y in zip(a,b))>tolerance or union<=0 or intersection/union<.5:
            return 'dual_scale_glyph_geometry_disagreement'
        # A one-canonical-pixel envelope covers antialiasing differences. It is
        # NOT guessed background: the entire envelope is proved below twice.
        # 包络向 canonical 像素网格对齐，并各外留一像素处理抗锯齿差异。
        # 扩大不是背景重建；新增支持范围必须在两个独立尺度中都被解释。
        # 页外、其它源框、未解释墨迹任意出现就拒绝。
        # 通过几何一致性仍不能给 OCR 候选独立语义擦除权。
        box=(floor((min(a[0],b[0])-dx)/dx)*dx,floor((min(a[1],b[1])-dy)/dy)*dy,
             ceil((max(a[2],b[2])+dx)/dx)*dx,ceil((max(a[3],b[3])+dy)/dy)*dy)
        if box[0]<0 or box[1]<0 or box[2]>geometry.width or box[3]>geometry.height:
            return 'dual_scale_mask_outside_page'
        if _foreign_overlap(box,entry,obstacles):return 'dual_scale_mask_neighbor_collision'
        if (not _support_is_explained(box,first_pixels,first,sx1,sy1)
                or not _support_is_explained(box,second_pixels,second,sx2,sy2)):
            return 'dual_scale_support_contains_unexplained_ink'
        l,t,r,bt=box;masks.append(((l,t),(r,t),(r,bt),(l,bt)))
    return tuple(masks)


def multiscale_raster_pdf_evidence(path,geometry,entries,source_image,obstacles=(),*,
                                  background=None,native_verified=frozenset(),
                                  expected_source_sha=None,cancelled=None):
    """Return a source-bound bundle; old canonical successes are never replaced."""
    # 此公开函数保留历史双尺度研究流程，正常 MaskEvidenceTask 未调用。
    # 源 SHA 将实验绑定到实际 PDF 字节，而非文件名或显示缩略图。
    # 背景副本与 native_verified 分别保留，附加研究失败不应撤销独立原生证据。
    # 取消、源内容改变或 canonical 缺失则撤销整包，防止陈旧证据污染。
    # 接口存在不能表述为当前产品自动擦除能力已启用。
    from .project_store import source_sha256
    cancelled=cancelled or (lambda:False)
    rejections={};masks={};widths=();before=None
    base=(background.copy() if background is not None and not background.isNull() else source_image.copy())
    # 统一完成出口负责决定证据撤销范围与失败原因。
    # 普通 supplemental 失败只清栅格结果，原生证据可保持。
    # 若尚未读到源哈希，则无法证明数据对应关系，应全部撤销。
    # 完成前再次检测源是否变化，封堵异常路径漏掉末尾哈希的空隙。
    # 返回图像总是复制，避免撤销逻辑把修改过的合成背景泄漏出去。
    def finish(reason=None):
        if reason is not None:
            revoke=reason in {'source_sha_mismatch','source_changed_during_evidence','canonical_image_unavailable','cancelled'}
            if before is None:revoke=True
            elif not revoke:
                try:changed=source_sha256(path)!=before
                except OSError:changed=True
                if changed:reason='source_changed_during_evidence';revoke=True
            # Ordinary supplemental rejection must not remove independent native
            # evidence. Source/cancellation failures revoke the whole bundle.
            preserved=(background.copy() if background is not None and not background.isNull() else source_image.copy())
            return PreviewMaskEvidence(source_image.copy() if revoke else preserved,
                                       frozenset() if revoke else native_verified,{},
                                       {e.group_id:reason for e in entries},widths)
        return PreviewMaskEvidence(base,native_verified,masks,rejections,widths)
    if source_image.isNull():return finish('canonical_image_unavailable')
    try:
        before=source_sha256(path)
        if expected_source_sha is not None and before!=expected_source_sha:return finish('source_sha_mismatch')
        if cancelled():return finish('cancelled')
        # 先检查每项文档页身份、OCR 来源与有限置信度。
        # NaN 置信度不能通过普通范围判断，必须显式有限性检查。
        # 限定英文字母句段研究范围，不把数字、公差等混合字面送去组件擦除。
        # canonical 已支持的旧候选不被双尺度替换，避免扩大研究改动范围。
        # 不产生 todo 时无需再渲染两张大图。
        canonical=_pixels(source_image);todo=[]
        for entry in entries:
            if entry.document_id!=geometry.document_id or entry.page_index!=geometry.page_index:
                rejections[entry.group_id]='entry_page_identity_mismatch';continue
            if any(r.source!='ocr' or r.confidence is None or not isfinite(r.confidence)
                   or not .995<=r.confidence<=1 for r in entry.regions):
                rejections[entry.group_id]='insufficient_ocr_confidence';continue
            if (len(entry.source_text)>500 or not all(re.fullmatch(r'[A-Za-z]+(?:[ \t]+[A-Za-z]+)*[.,:]?',r.source_text.strip()) for r in entry.regions)):
                rejections[entry.group_id]='unsupported_region_text_pattern';continue
            low=_evidence(entry,obstacles,canonical,canonical.shape[1]/geometry.width,canonical.shape[0]/geometry.height)
            if not isinstance(low,str):
                rejections[entry.group_id]='canonical_evidence_already_supported';continue
            todo.append(entry)
        if not todo:
            if source_sha256(path)!=before:return finish('source_changed_during_evidence')
            return finish()
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(str(path)) as pdf:
            page=pdf[geometry.page_index]
            try:
                w,h=page.get_size();media=page.get_mediabox();crop=page.get_cropbox()
                # 研究路径支持标准直角 Rotate，但仍限制零媒体原点与相同 CropBox。
                # 宽高必须与当前显示源几何一致，未知变换不做自动猜测。
                # 页面旋转有值时还需后续非文字路标双向校验。
                # 这些研究兼容条件不能反向放宽生产原生对象去字路径的门禁。
                rotation=page.get_rotation()
                if (rotation not in {0,90,180,270} or media!=crop or abs(media[0])+abs(media[1])>.01
                        or abs(w-geometry.width)>.1 or abs(h-geometry.height)>.1):
                    return finish('unsupported_pdf_page_transform')
                # PDFium rounds each raster side upward. Reserve that rounding
                # before allocation instead of rendering just beyond the cap.
                # 先为 PDFium 向上取整预留余量，再按单边和总像素预算求尺度。
                # 两尺度差不足时不声称获得独立证据，直接报告预算内不可用。
                # 每次 render 后仍实际复核尺寸，公式不能替代真实分配检查。
                # 页面与 bitmap 由 finally 释放，避免多页实验累积底层内存。
                limit=min(5999/max(w,h),sqrt(23988000/(w*h)))
                scales=(min(3.0,limit),min(4.0,limit))
                if scales[1]-scales[0]<.1:return finish('independent_scales_unavailable_within_budget')
                renders=[];scale_masks=[]
                for scale in scales:
                    if cancelled():return finish('cancelled')
                    bitmap=page.render(scale=scale)
                    try:pixels=np.asarray(bitmap.to_pil().convert('RGB')).copy()
                    finally:bitmap.close()
                    if max(pixels.shape[:2])>6000 or pixels.shape[0]*pixels.shape[1]>24000000:
                        return finish('render_exceeds_pixel_budget')
                    renders.append(pixels)
                    scale_masks.append({entry.group_id:_evidence(entry,obstacles,pixels,pixels.shape[1]/w,pixels.shape[0]/h) for entry in todo})
                widths=tuple(p.shape[1] for p in renders)
                if rotation:
                    disagreement=_rotation_alignment(canonical,renders[0],entries,geometry)
                    if disagreement is not None:return finish(disagreement)
                # 每个目标分别组合两个尺度的结果，拒绝原因完整保留双侧诊断。
                # 只在两尺度都返回多边形时进入一致性匹配。
                # 取消检查位于渲染和逐项处理边界，尽快停止后续实验工作。
                # 扫描文字高置信也不改 provenance，更不等于机器译文可正式导出。
                for entry in todo:
                    if cancelled():return finish('cancelled')
                    a,b=(values[entry.group_id] for values in scale_masks)
                    if isinstance(a,str) or isinstance(b,str):
                        rejections[entry.group_id]='dual_scale_rejected:'+str(a if isinstance(a,str) else 'accepted')+'|'+str(b if isinstance(b,str) else 'accepted')
                        continue
                    value=_consistent_masks(entry,obstacles,a,b,*renders,geometry,source_image)
                    if isinstance(value,str):rejections[entry.group_id]=value
                    else:masks[entry.group_id]=value
            finally:page.close()
        if source_sha256(path)!=before:return finish('source_changed_during_evidence')
        if cancelled():return finish('cancelled')
        # 只有完成源身份复核后才在私有背景上画研究白框。
        # 源坐标多边形按实际 base 尺寸换算，不改变原布局坐标。
        # 返回 masks 与合成背景供历史诊断使用，生产布局会忽略 OCR 擦除授权。
        # 异常最后归入 evidence_unavailable，不以异常后半成品创建新证明。
        painter=QPainter(base);painter.setPen(QColor('white'));bx=base.width()/geometry.width;by=base.height()/geometry.height
        for polygons in masks.values():
            for poly in polygons:
                l,t,r,b=_box(poly);painter.fillRect(QRectF(l*bx,t*by,(r-l)*bx,(b-t)*by),QColor('white'))
        painter.end()
        return finish()
    except Exception:
        # A malformed/unavailable private PDF or image never creates new proof.
        return finish('pdf_raster_evidence_unavailable')
