# mechanical_drawing_translator/annotation_preview.py
"""Conservative source-coordinate preview with local placement and explicit review."""
from dataclasses import replace
from math import ceil,cos,sin,radians
import re
import numpy as np
from PySide6.QtGui import QImage,QPainter,QColor
from .annotation_models import AnnotationItem,AnnotationLayout,MaskDecision
from .font_resources import ensure_font_supports,load_annotation_font
from .mask_safety import analyze_mask_safety
from .models import TextBounds
from .translation_results import validate_translation_results,preview_translation_available


# 所有碰撞函数共享最终旋转矩形的轴对齐包围框。
# 这里计算的是几何包络，不是逐字轮廓，也不会生成擦除证据。
# 正负角都通过绝对投影宽高统一处理，中心保持不变。
# 旋转围绕框中心，与 annotation_painter 的变换顺序对应。
def footprint(bounds,angle):
    """Axis-aligned enclosure of final rotated ink rectangle, in source units."""
    a=radians(angle);w=bounds.max_x-bounds.min_x;h=bounds.max_y-bounds.min_y
    rw=abs(w*cos(a))+abs(h*sin(a));rh=abs(w*sin(a))+abs(h*cos(a))
    x=(bounds.min_x+bounds.max_x)/2;y=(bounds.min_y+bounds.max_y)/2
    return TextBounds(x-rw/2,y-rh/2,x+rw/2,y+rh/2)


# 缓存只保存图框检测结果，避免同一源图反复做形态学与 Hough。
# 键包含图像 cacheKey、文档、页和源尺寸，不能跨页复用。
# 容量有界，防止多页操作长期积累整批图框对象。
# 它不是遮挡证据缓存，更不能用旧缓存保留擦除资格。
_FRAME_EVIDENCE_CACHE={}
def _cached_frame_evidence(pixels,page,image_identity):
    from .drawing_frame_detection import detect_frame_evidence
    key=(image_identity,page.document_id,page.page_index,page.width,page.height)
    cached=_FRAME_EVIDENCE_CACHE.get(key)
    if cached is not None:return cached
    evidence=detect_frame_evidence(pixels,page)
    _FRAME_EVIDENCE_CACHE[key]=evidence
    while len(_FRAME_EVIDENCE_CACHE)>16:
        _FRAME_EVIDENCE_CACHE.pop(next(iter(_FRAME_EVIDENCE_CACHE)),None)
    return evidence


class BlankSpace:
    """Conservative pooled ink integral: even a thin dark pixel blocks its tile."""
    def __init__(self,image,page):
        # 空图一律不可判为空白，否则缺失背景会被误当安全空间。
        # 透明图先合成白纸只用于布局检测，不覆盖原始 RGBA 导出数据。
        # 图像身份在合成前取得，便于同一原图稳定命中图框缓存。
        # 转换 RGB 后必须尊重 bytesPerLine，Qt 可能在行尾填充对齐字节。
        self.page=page;self.valid=not image.isNull()
        image_identity=image.cacheKey()
        if not self.valid:return
        if image.hasAlphaChannel():
            paper=QImage(image.size(),QImage.Format.Format_RGB32);paper.fill(QColor('white'))
            painter=QPainter(paper);painter.drawImage(0,0,image);painter.end();image=paper
        rgb=image.convertToFormat(QImage.Format.Format_RGB888)
        pixels=np.frombuffer(rgb.constBits(),dtype=np.uint8,count=rgb.sizeInBytes()).reshape(rgb.height(),rgb.bytesPerLine())[:,:rgb.width()*3].reshape(rgb.height(),rgb.width(),3)
        # 池化网格最长边约 1500 格，控制整页候选查询的内存和计算量。
        # 每格以 any 保留任意非白墨迹，细线不会被平均成白色。
        # 这是一种保守拒绝：同格有墨迹时，格内其它空白也暂不使用。
        # 浅灰和彩色淡线都算占用，避免灰度四舍五入隐藏工程信息。
        # 纸张外补为占用，边缘不足一格的区域不会虚构可用空间。
        tile=max(1,ceil(max(rgb.width(),rgb.height())/1500))
        # All channels must be white; grayscale rounding can hide pale colors.
        dark=(pixels<255).any(axis=2)
        self.frame_evidence=_cached_frame_evidence(pixels,page,image_identity)
        self.effective_bounds=self.frame_evidence.bounds
        ph=(-dark.shape[0])%tile;pw=(-dark.shape[1])%tile
        dark=np.pad(dark,((0,ph),(0,pw)),constant_values=True)
        grid=dark.reshape(dark.shape[0]//tile,tile,dark.shape[1]//tile,tile).any(axis=(1,3))
        # 积分图提供常数次矩形查询，不在每个候选上扫描全部像素。
        # 前置零行零列使四角相减公式可覆盖靠边矩形。
        # 累计使用 uint32；网格已受最长边约束，避免普通字节累计溢出。
        # sx/sy 把源单位映射到池化格，X/Y 分开处理非正方形缩放。
        # 积分图用于候选背景空白，不具备判断笔画属于英文的语义。
        self.integral=np.pad(grid.astype(np.uint32).cumsum(0,dtype=np.uint32).cumsum(1,dtype=np.uint32),((1,0),(1,0)))
        self.sx=image.width()/page.width/tile;self.sy=image.height()/page.height/tile

    def clear(self,box):
        # 先检查纸张范围，再检查已可靠测得的有效图框。
        # 图框未知不伪装成检测成功，纸张界限仍然有效。
        # 左上向下取整、右下向上取整，保守包含每个可能覆盖的池化格。
        # 候选恰好碰到暗格也拒绝，不通过采样跳过细尺寸线。
        # 返回 True 仅说明这张检测图上的该矩形纯白，不直接授权擦英文。
        if not self.valid:return False
        if box.min_x<0 or box.min_y<0 or box.max_x>self.page.width or box.max_y>self.page.height:return False
        b=self.effective_bounds
        if b is not None and (box.min_x<b.min_x or box.min_y<b.min_y or box.max_x>b.max_x or box.max_y>b.max_y):return False
        x0=max(0,int(box.min_x*self.sx));y0=max(0,int(box.min_y*self.sy))
        x1=min(self.integral.shape[1]-1,ceil(box.max_x*self.sx));y1=min(self.integral.shape[0]-1,ceil(box.max_y*self.sy))
        q=self.integral
        return int(q[y1,x1])+int(q[y0,x0])-int(q[y0,x1])-int(q[y1,x0])==0

    @staticmethod
    def _closed_frame(pixels,page):
        """Compatibility API; detection retains evidence instead of paper fallback."""
        from .drawing_frame_detection import detect_frame_evidence
        return detect_frame_evidence(pixels,page).bounds



def _pad(box,gap):return TextBounds(box.min_x-gap,box.min_y-gap,box.max_x+gap,box.max_y+gap)


# 这是带版本与图框诊断的统一预览入口。
# 内部算法返回布局后，用 replace 补上当前几何身份，原对象不被修改。
# 图框证据含方法、置信度、观测和拒绝，未知原因也会保存。
# 源图缺失时记录 missing_image，不用纸张尺寸伪造可靠图框。
# 版本供项目恢复和缓存判断使用，与翻译模型版本分开。
def create_preview_layout(engine,page,entries,results,image,mask_requested,obstacles,mask_background=None,verified_mask_groups=frozenset(),display_mode=None,supplemental_masks=None,supplemental_reasons=None):
    from .layout_identity import LAYOUT_ALGORITHM_VERSION
    layout=_create_preview_layout(engine,page,entries,results,image,mask_requested,obstacles,
                                  mask_background,verified_mask_groups,display_mode,supplemental_masks,supplemental_reasons)
    frame=BlankSpace(image,page).frame_evidence if not image.isNull() else None
    metadata=((f'frame_method:{frame.method}',f'frame_confidence:{frame.confidence:.3f}',
               *frame.evidence,*frame.rejections) if frame is not None else ('frame_unknown:missing_image',))
    return replace(layout,layout_version=LAYOUT_ALGORITHM_VERSION,
                   effective_bounds=frame.bounds if frame is not None else None,frame_evidence=metadata)


def _create_preview_layout(engine,page,entries,results,image,mask_requested,obstacles,mask_background=None,verified_mask_groups=frozenset(),display_mode=None,supplemental_masks=None,supplemental_reasons=None):
    # 模式解析保留历史数据兼容，当前正式界面只有英文和中英对照。
    # 结果先按完整目标集合校验，再应用当前英文资格投影。
    # 这样历史被过滤项仍须满足身份契约，不会成为隐藏的坏缓存。
    # 预览允许 machine 来源，但返回布局始终带 preview_only 门禁。
    # 验证失败是契约错误，不应默默把另一页结果移到当前页。
    mode=display_mode or 'legacy'
    if mode not in {'legacy','chinese','bilingual','english'}:raise ValueError('unsupported display mode')
    validated=validate_translation_results(entries,results.values(),allowed_provenance=frozenset({'machine','manual_test','human_reviewed'}))
    for e in entries:
        if e.document_id!=page.document_id or e.page_index!=page.page_index:raise ValueError('preview target does not belong to this page')
    # Caller-supplied group IDs are not independent erasure proof for OCR.
    # Only complete native PDF objects may consume an independently rendered
    # background; stale/forged OCR verified or supplemental masks are ignored.
    # 外部传入的 verified 只作候选索引，必须再确认完整原生对象关联。
    # OCR、高置信框、双尺度组件和 supplemental 蒙版都不能替代原生证据。
    # 共享区域的片段无法只隐藏对应字面，不能拿全框擦掉剩余原文。
    # 这一收紧在分派模式前进行，防止旧兼容分支绕过当前安全规则。
    verified_mask_groups=frozenset(e.group_id for e in entries
        if e.group_id in verified_mask_groups and _complete_native_entry(e))
    # 英文直接返回空项目布局且关闭 mask_requested。
    # 无需先生成中文再隐藏，避免旧蒙版滞留在输出副本。
    # 字体族仍填入统一模型字段，但不会实际绘制中文或覆盖层。
    if mode=='english':
        return AnnotationLayout(page.document_id,page.page_index,page.width,page.height,page.coordinate_unit,
            load_annotation_font(),False,(),True,page.width,mode)
    # Recheck imported/cached targets at the final drawing boundary too. A
    # stale candidate containing only engineering numbers is not English prose.
    all_entries=entries
    if mode!='legacy':
        from .source_text_policy import is_translatable_english
        entries=tuple(e for e in entries if is_translatable_english(e.source_text))
    if mode=='chinese':
        layout=_chinese_replacement_layout(engine,page,entries,validated,image,obstacles,
                                           mask_background,verified_mask_groups,supplemental_masks,supplemental_reasons)
        return _retain_preserved_sources(engine,layout,all_entries,validated)
    # 双语先独立完成邻近中文位置，再叠加可选遮挡资格。
    # 遮挡失败不能退回重排中文，也不能把有位置的译文隐藏。
    # 保留源记录在布局构造后补回，维持全部原 group 与显示索引关联。
    # 开关关闭时 apply_bilingual_masks 会清空旧蒙版及证据收据。
    if mode=='bilingual':
        from .joint_annotation_layout import build_bilingual_layout
        layout=build_bilingual_layout(engine,page,entries,validated,image,obstacles)
        layout=_retain_preserved_sources(engine,layout,all_entries,validated)
        return apply_bilingual_masks(engine,layout,entries,mask_requested,mask_background,
                                     verified_mask_groups,obstacles)
    blank=BlankSpace(image,page)
    mask_blank=BlankSpace(mask_background,page) if mask_background is not None else None
    # legacy 是双语联合算法的可行种子，也承接旧项目兼容。
    # 有完整 OCR 障碍时优先使用全部障碍，避免只保护送译文字。
    # 没有显式障碍才用目标框保守替代，且统一扩大间距。
    # 背景图与独立去原生字背景分开，普通空白不能升级为擦除证明。
    occupied=[_pad(r.bounds,1.5) for r in obstacles] or [_pad(e.bounds,1.5) for e in entries]
    items=[]
    for index,e in enumerate(entries,1):
        r=validated.get(e.group_id)
        warnings=list(r.warnings if r else ('translation_result_missing',))
        target=r.translated_text if r else e.source_text
        available=preview_translation_available(r)
        box=e.bounds;rotation=engine._dominant_rotation(e);placement='inline'
        size=min(14.0,max(engine.min_font_size,(box.max_y-box.min_y)*.78))
        lines=();text_bounds=box;decision=MaskDecision.NOT_REQUESTED;masks=()
        # 无可用译文仍创建关联项，点击可查看原文与失败原因。
        # 已明确无需翻译的记录为 preserved，其它失败为 unavailable。
        # 两者均没有绘制行，不以红色原文伪装成功中文。
        # 有译文才检查字体、长度和安全局部位置。
        # 长度上限控制异常文本成本，不截去长译文尾部。
        if not available:
            if r and 'reviewed94:no_translation_required' in r.warnings:
                placement='preserved'
            else:
                placement='unavailable';warnings.append('未成功翻译，保留英文；点击区域查看原因')
        else:
            try:
                ensure_font_supports(target)
                if len(target)>2000:raise ValueError('译文过长')
                # 局部候选同时避让原框、已放中文和像素墨迹。
                # 成功后将旋转占用框加入 occupied，防止后项覆盖它。
                # 失败时保留完整 target 于 item，只清空绘制行并提示待布局。
                # 异常捕获限于可预期字体/适配问题，不伪造一个可显示位置。
                placed=engine._fit_local(target,box,rotation,page,occupied,size,blank)
                if placed is not None:
                    text_bounds,lines,size=placed
                    occupied.append(_pad(footprint(text_bounds,rotation),1))
                else:
                    placement='needs_layout_review';lines=()
                    warnings.append('needs_layout_review：邻近区域无法安全放置，请人工调整；完整译文保留在列表')
            except (ValueError,RuntimeError) as exc:
                placement='needs_layout_review';lines=();rotation=0;text_bounds=box
                warnings.append(f'无法安全放置完整中文，点击原区域查看：{exc}')
        item=AnnotationItem(e.group_id,e.region_ids,e.source_text,target,r.provenance if r else 'machine',
            engine._source_polygons(e),text_bounds,size,lines,rotation,masks,decision,tuple(warnings),
            None,placement,index)
        if available and placement == 'inline' and mask_requested:
            shared=e.coordinate_precision=='shared_region_bounds'
            other_boxes=[o.bounds for o in obstacles if o.id not in e.region_ids]
            # 原生证明和独立背景缺一不可，不能凭周围四角白色批准整框。
            # 其它原文障碍先排除本组区域，再检查邻近与共享框风险。
            # 源多边形扩展分析后还要在独立背景上确认内部空白。
            # 边界没有穿线不保证内部没有短线，所以两级检查不能合并。
            # 最终复用已经确定的中文框，遮挡结论不回写中文位置。
            if e.group_id not in verified_mask_groups or mask_blank is None:
                item=replace(item,mask_decision=MaskDecision.UNSUPPORTED_BACKGROUND,
                    warnings=(*item.warnings,'未获得可验证的纯文字背景证据，已自动保留英文（扫描/OCR区域不猜测擦除）'))
            elif shared or any(engine._overlap_area(_pad(box,3),o)>0 for o in other_boxes):
                item=replace(item,mask_decision=MaskDecision.UNSAFE_GEOMETRY,
                    warnings=(*item.warnings,'区域与工程信息相邻或共享坐标，已自动保留英文'))
            else:
                proposed=replace(item,mask_polygons=item.source_polygons,mask_decision=MaskDecision.PENDING_ANALYSIS)
                one=AnnotationLayout(page.document_id,page.page_index,page.width,page.height,page.coordinate_unit,
                    load_annotation_font(),True,(proposed,),True,page.width)
                tested=analyze_mask_safety(one,mask_background).items[0]
                safe_polygons=tested.mask_polygons
                for poly in safe_polygons:
                    f=TextBounds(min(x for x,y in poly),min(y for x,y in poly),max(x for x,y in poly),max(y for x,y in poly))
                    if not mask_blank.clear(f) or any(engine._overlap_area(f,o)>0 for o in other_boxes):
                        tested=replace(tested,mask_polygons=(),mask_decision=MaskDecision.UNSAFE_GEOMETRY,
                            warnings=(*tested.warnings,'原文字框内部仍有工程图形或其他文字，已保留英文'))
                        break
                # Keep independently verified text placement when masking is unsafe.
                item=replace(tested,text_bounds=text_bounds,fallback_text_bounds=None)
        items.append(item)
    return AnnotationLayout(page.document_id,page.page_index,page.width,page.height,page.coordinate_unit,
        load_annotation_font(),mask_requested,tuple(items),True,page.width,mode)


# 当前源策略过滤掉的历史项仍需要保留不可变关联。
# 不把数字、编号或视图标签当新翻译成功，也不删除它们的来源。
# 补回项 target 等于 source，lines 和蒙版为空，明确 preserved。
# 后续导出例外还会重新计算源策略，不能只信 placement 字段。
# 顺序按原 entries 重建，display_index 不因过滤发生错配。
def _retain_preserved_sources(engine,layout,entries,results):
    """Routing changes remove ink, never the immutable source/group association."""
    from .source_text_policy import english_translation_eligibility
    items={item.group_id:item for item in layout.items};ordered=[]
    for index,entry in enumerate(entries,1):
        item=items.get(entry.group_id)
        if item is None:
            result=results.get(entry.group_id);policy=english_translation_eligibility(entry.source_text)
            item=AnnotationItem(entry.group_id,entry.region_ids,entry.source_text,entry.source_text,
                result.provenance if result else 'machine',engine._source_polygons(entry),entry.bounds,
                engine.min_font_size,(),engine._dominant_rotation(entry),(),MaskDecision.NOT_REQUESTED,
                (*(result.warnings if result else ()), 'source_text_preserved','source_policy:'+policy.reason),
                placement='preserved',display_index=index)
        ordered.append(replace(item,display_index=index))
    return replace(layout,items=tuple(ordered))


# 这是历史中文原位兼容路径的字体候选，不增加正式产品模式。
# 直角文字交换可用宽高，斜向字不猜测原位替换。
# 先保留历史候选，再尝试实际字形紧框，以维持已有可行结果。
# 去掉字体留白不等于减小字形字号，最低仍为六源单位。
# 候选去重包含 lines、字号、尺寸与基线偏移，避免重复验证。
# 返回中心对齐原锚点的框，是否可擦仍由独立证据与碰撞判断。
def _replacement_fits(engine,target,box,rotation):
    """Keep the old fit first, then fit real glyph ink without reducing size."""
    from .glyph_text_geometry import tight_text_fit
    width=box.max_x-box.min_x;height=box.max_y-box.min_y
    if abs(rotation%180-90)<.01:width,height=height,width
    elif min(abs(rotation%180),abs(rotation%180-180))>.01:
        raise ValueError('斜向原文暂不能保证原位替换安全')
    floor=max(6.0,engine.min_font_size)
    initial=max(floor,min(14.0,max(engine.min_font_size,height*.78)));seen=set()
    sizes=[initial]
    while sizes[-1]>floor:sizes.append(max(floor,sizes[-1]-1))
    # Preserve every historical candidate first. A rectangle fitting the OCR
    # box may still fail the independent ink/neighbor checks below; enumerate
    # smaller tight-ink alternatives rather than abandoning that source word.
    for fit,starts in ((engine._fit_text,(initial,)),(tight_text_fit,sizes)):
        for start in starts:
            for margin in (0,1,2):
                try:
                    value=(fit(engine,target,width+margin*2,height+margin*2,start)
                           if fit is tight_text_fit else fit(target,width+margin*2,height+margin*2,start))
                except ValueError:continue
                lines,size,w,h=value[:4];origin=value[4]if len(value)>4 else None
                identity=(lines,size,w,h,origin)
                if size<floor or identity in seen:continue
                seen.add(identity);cx=(box.min_x+box.max_x)/2;cy=(box.min_y+box.max_y)/2
                yield TextBounds(cx-w/2,cy-h/2,cx+w/2,cy+h/2),lines,size,origin


# 每个关联区域都必须明确来自 pdf_native。
# 有 SourceSpan 时要求逐区域从零到原文字长的完整覆盖。
# 字符范围相同但数量重复也不行，因此同时比较集合和长度。
# 无 spans 是旧数据兼容，后续仍需原生对象文本精确匹配。
# 这里不因人工审核或置信度很高而接受 OCR 代替原生对象。
def _complete_native_entry(entry):
    """A full native object association is necessary, never OCR confidence."""
    if not entry.regions or any(r.source!='pdf_native'for r in entry.regions):return False
    if entry.coordinate_precision=='shared_region_bounds':return False
    return not entry.spans or (len(entry.spans)==len(entry.regions)and
        {(s.region_id,s.start,s.end)for s in entry.spans}==
        {(r.id,0,len(r.source_text))for r in entry.regions})


def apply_bilingual_masks(engine,layout,entries,requested,background,verified,obstacles):
    """Optional erasure changes only source polygons, never caption geometry.

    Native-object removal is the independent evidence. A white-looking scan
    or a saved SAFE flag does not substitute for it. Recheck every expanded
    polygon against the object-free background and other source associations;
    this catches short strokes as well as long engineering lines.
    """
    # 双语可选蒙版只改变 source 的视觉覆盖，不改变 caption 几何。
    # 从当前布局页尺寸构造背景查询，避免混入屏幕坐标。
    # 背景为空时所有遮挡申请保守失败，中文位置仍可照常保留。
    # 当前 entry 映射用于重新核对 group，而不是信保存项目的收据。
    entries={e.group_id:e for e in entries}
    from .annotation_models import PageGeometry
    page=PageGeometry(layout.document_id,layout.page_index,layout.page_width,layout.page_height,layout.coordinate_unit)
    blank=BlankSpace(background,page) if background is not None and not background.isNull() else None
    items=[]
    for item in layout.items:
        # 每项先清理旧蒙版、SAFE 和收据，关闭开关可立即去除残留。
        # 无绘制行或非 inline 项无需申请遮挡，保持无墨迹语义。
        # 只有完整原生 entry、当前 verified 与当前背景同时满足才分析。
        # 独立证明失败只加诊断，不更改 item 的中文文字框或译文。
        clean=replace(item,mask_polygons=(),mask_decision=MaskDecision.NOT_REQUESTED,mask_evidence=())
        if not requested or not item.lines or item.placement!='inline':items.append(clean);continue
        entry=entries.get(item.group_id)
        reason='native_object_evidence_missing'
        if entry is not None and _complete_native_entry(entry) and entry.group_id in verified and blank is not None:
            candidate=replace(clean,mask_polygons=clean.source_polygons,mask_decision=MaskDecision.PENDING_ANALYSIS)
            tested=analyze_mask_safety(replace(layout,mask_requested=True,items=(candidate,)),background).items[0]
            # 除原始 OCR 障碍，还把其它翻译组 bounds 作为独立阻挡。
            # 即使障碍列表不完整，也不能擦掉相邻目标的工程字面。
            # 对每一个扩大后的蒙版都做纯白和交叠复验。
            # SAFE 是多边形全部通过后的整体结论，不允许只擦一部分组。
            # 成功收据明确指出原生移除背景重验，便于追查授权路径。
            others=[r.bounds for r in obstacles if r.id not in entry.region_ids]
            others.extend(e.bounds for e in entries.values() if e.group_id!=entry.group_id)
            safe=tested.mask_decision==MaskDecision.SAFE
            for poly in tested.mask_polygons:
                b=TextBounds(min(x for x,y in poly),min(y for x,y in poly),max(x for x,y in poly),max(y for x,y in poly))
                safe=safe and blank.clear(b) and not any(engine._overlap_area(b,o)>0 for o in others)
            if safe:
                items.append(replace(tested,mask_evidence=('native_object_removed_background_revalidated',)))
                continue
            reason='remaining_ink_or_neighbor_geometry'
        elif entry is not None and not _complete_native_entry(entry):reason='ocr_derived_erasure_not_authorized'
        items.append(replace(clean,mask_decision=MaskDecision.UNSUPPORTED_BACKGROUND,
            warnings=(*clean.warnings,'mask_evidence_rejected:'+reason)))
    return replace(layout,mask_requested=bool(requested),items=tuple(items))


# 保留旧中文原位恢复接口，正式双模式 UI 不直接暴露该模式。
# 原位失败保持英文并记录 needs_layout_review，不挪远冒充替换。
# supplemental 参数为历史兼容，当前不消费栅格蒙版权限。
# 人工位置或旧 SAFE 不能让扫描/OCR 项获得本路径擦除资格。
def _chinese_replacement_layout(engine,page,entries,results,image,obstacles,background,verified,supplemental_masks=None,supplemental_reasons=None):
    """Replace at the source anchor only after independently proving safe ink.

    A translated nearby caption would make this a bilingual view. Failed
    replacement therefore keeps original ink and records a visible review
    state instead of drawing a displaced substitute.
    """
    verified=frozenset(e.group_id for e in entries if e.group_id in verified and _complete_native_entry(e))
    blank=BlankSpace(background,page) if background is not None else None
    items=[];occupied=[]
    for index,e in enumerate(entries,1):
        r=results.get(e.group_id)
        target=r.translated_text if r else e.source_text
        item=AnnotationItem(e.group_id,e.region_ids,e.source_text,target,r.provenance if r else 'machine',
            engine._source_polygons(e),e.bounds,engine.min_font_size,(),engine._dominant_rotation(e),(),
            MaskDecision.UNSUPPORTED_BACKGROUND,tuple(r.warnings if r else ('translation_result_missing',)),
            placement='needs_layout_review',display_index=index)
        if not preview_translation_available(r):
            item=replace(item,placement='preserved' if r and 'reviewed94:no_translation_required' in r.warnings else 'unavailable')
        # 缺少独立背景或完整原生证明时，不枚举会误导的原位候选。
        # 共享范围意味着同一框可能还有其它字，必须保留整组英文。
        # 原因同时写入面向用户的提示和机器可检索的 rejection 标签。
        # 已有译文留在列表供检查，不等于原位功能通过。
        elif (blank is None or not blank.valid or e.group_id not in verified
              or e.coordinate_precision=='shared_region_bounds'):
            item=replace(item,warnings=(*item.warnings,'needs_layout_review：中文原位替换缺少可靠文字背景证据，保留英文；可切换中英对照',
                'mask_evidence_rejected:'+('native_object_evidence_missing'if _complete_native_entry(e)else'ocr_derived_erasure_not_authorized')))
        else:
            try:
                ensure_font_supports(target)
                if len(target)>2000:raise ValueError('译文过长')
                other=[o.bounds for o in obstacles if o.id not in e.region_ids]
                other.extend(x.bounds for x in entries if x.group_id!=e.group_id)
                other.extend(occupied)
                # 每个字号候选都重新分析扩展遮挡和实际旋转文字占用。
                # 原位框、扩大后的蒙版、其它源框及已放中文都必须互不相交。
                # 不能只检验蒙版，也不能只检验中文；两者覆盖范围不同。
                # 只有全部成立才提交 item 并预留占用，否则尝试下一个候选。
                # 耗尽候选时保留初始无墨迹项，异常不写入新擦除多边形。
                for bounds,lines,size,origin in _replacement_fits(engine,target,e.bounds,item.rotation_degrees):
                    proposed=replace(item,text_bounds=bounds,font_size=size,lines=lines,text_origin=origin,
                        mask_polygons=item.source_polygons,mask_decision=MaskDecision.PENDING_ANALYSIS,placement='inline')
                    one=AnnotationLayout(page.document_id,page.page_index,page.width,page.height,page.coordinate_unit,
                        load_annotation_font(),True,(proposed,),True,page.width,'chinese')
                    tested=analyze_mask_safety(one,background).items[0]
                    f=_pad(footprint(bounds,item.rotation_degrees),1)
                    safe=tested.mask_decision==MaskDecision.SAFE and blank.clear(f)
                    for poly in tested.mask_polygons:
                        p=TextBounds(min(x for x,y in poly),min(y for x,y in poly),max(x for x,y in poly),max(y for x,y in poly))
                        safe=safe and blank.clear(p) and not any(engine._overlap_area(p,o)>0 for o in other)
                    if safe and not any(engine._overlap_area(f,o)>0 for o in other):
                        item=tested;occupied.append(f);break
                else:raise ValueError('字号或原位空间不足，或附近存在尺寸、线稿、其他文字或越界风险')
            except (ValueError,RuntimeError) as exc:
                item=replace(item,warnings=(*item.warnings,f'needs_layout_review：中文无法安全原位替换，保留英文：{exc}'))
        items.append(item)
    return AnnotationLayout(page.document_id,page.page_index,page.width,page.height,page.coordinate_unit,
        load_annotation_font(),True,tuple(items),True,page.width,'chinese')
