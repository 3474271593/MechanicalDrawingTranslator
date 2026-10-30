# mechanical_drawing_translator/joint_annotation_layout.py
"""Bounded nearby caption candidates and deterministic page-wide reassignment.

English and engineering ink are immutable obstacles. The old feasible greedy
assignment seeds the optimizer: adding a pending caption may move existing
captions, but a transaction is accepted only after every displaced caption has
another safe slot. All searches use fixed operation budgets, not wall clocks.
"""
from dataclasses import dataclass,replace
from math import hypot
import numpy as np

from .annotation_models import AnnotationLayout,MaskDecision
from .annotation_preview import BlankSpace,_pad,footprint
from .font_resources import ensure_font_supports,load_annotation_font
from .glyph_text_geometry import tight_text_fit
from .models import TextBounds
from .translation_results import preview_translation_available


# 联合求解只处理中文邻近位置，原英文和工程墨迹始终是障碍。
# 候选预算与修复节点预算固定，避免拥挤大图无限搜索。
# 版本是算法诊断标签；持久缓存身份另由 layout_identity 管理。
# 候选增加不会自动提升翻译质量、审核来源或遮挡资格。
LAYOUT_VERSION='joint-nearby-observed-gap-v3'
MAX_CANDIDATES=96
MAX_REPAIR_NODES=192


# 候选同时保存未旋转文字框和带保护边的旋转占用框。
# lines、size 和 origin 必须与该框对应，不能在选定后另行换行。
# rank 是稳定排序依据，使同输入不依赖墙钟时间得到确定结果。
# 不可变对象可在试探分配中复用，回滚不需要撤销原对象修改。
@dataclass(frozen=True)
class _Candidate:
    bounds: TextBounds
    lines: tuple[str,...]
    size: float
    occupied: TextBounds
    rank: tuple
    origin: tuple[float,float] | None = None


# 严格正交叠才算框冲突，边缘接触本身不产生面积。
# 实际候选已有保护边，允许边缘相接不等于让字形贴住工程线。
# 这里比较的是轴对齐占用，旋转需在创建候选时完成。
def _overlap(a,b):
    return min(a.max_x,b.max_x)>max(a.min_x,b.min_x) and min(a.max_y,b.max_y)>max(a.min_y,b.min_y)


# 距离是两个矩形边缘之间的欧氏空隙。
# 对大原文框更公平，不会因中心相距远就误判为关联过远。
# 四十源单位的上限约束关联关系，不是四十屏幕像素。
def _distance(source,box):
    return hypot(max(source.min_x-box.max_x,box.min_x-source.max_x,0),
                 max(source.min_y-box.max_y,box.min_y-source.max_y,0))


# 已有可行种子也转换为同一种候选结构。
# 保护边和距离排序与新候选一致，避免混用两套碰撞语义。
# 旧 text_origin 随候选保留，否则重放会改变实际字形位置。
def _candidate(item,source):
    box=_pad(footprint(item.text_bounds,item.rotation_degrees),1)
    return _Candidate(item.text_bounds,item.lines,item.font_size,box,
                      (_distance(source,box),-item.font_size,len(item.lines),0,0),item.text_origin)


# 向量化只负责批量预筛，最终接受仍调用 BlankSpace.clear。
# 坐标数组第三列 side 只用于排序，不参与尺度转换。
# 页面钳制后继续检查真实边界，尺寸过大的框不能靠 clip 变安全。
# 如果检测到可靠图框，同一批次也必须完全落在图框之内。
# 积分图相减转换 int64，避免无符号减法下溢产生伪空白。
# 批量筛选不把原图降采样，暗格仍来自任意非白源像素。
def _blank_positions(blank,page,positions,rw,rh):
    """Vectorized rejection only; accepted rectangles are rechecked by clear()."""
    if not blank.valid:return ()
    values=np.asarray(positions,dtype=np.float64)
    values[:,0]=np.clip(values[:,0],2.0,max(2.0,page.width-rw-2))
    values[:,1]=np.clip(values[:,1],2.0,max(2.0,page.height-rh-2))
    left=values[:,0]-1;top=values[:,1]-1
    right=values[:,0]+rw+1;bottom=values[:,1]+rh+1
    keep=(left>=1)&(top>=1)&(right<=page.width-1)&(bottom<=page.height-1)
    frame=blank.effective_bounds
    if frame is not None:
        keep&=(left>=frame.min_x)&(top>=frame.min_y)&(right<=frame.max_x)&(bottom<=frame.max_y)
    values=values[keep];left=left[keep];top=top[keep];right=right[keep];bottom=bottom[keep]
    x0=np.clip((left*blank.sx).astype(np.int64),0,blank.integral.shape[1]-1)
    y0=np.clip((top*blank.sy).astype(np.int64),0,blank.integral.shape[0]-1)
    x1=np.clip(np.ceil(right*blank.sx).astype(np.int64),0,blank.integral.shape[1]-1)
    y1=np.clip(np.ceil(bottom*blank.sy).astype(np.int64),0,blank.integral.shape[0]-1)
    q=blank.integral
    ink=q[y1,x1].astype(np.int64)+q[y0,x0].astype(np.int64)-q[y0,x1].astype(np.int64)-q[y1,x0].astype(np.int64)
    return values[ink==0]


# 常规对齐网格可能跨过很窄的标题栏空格，观察网格提供补充起点。
# 只在原锚点附近四十单位内检查池化格边界，避免全页搜索。
# 网格宽高把两侧保护边计入，空白不足的格不会被当作可用。
# 返回的是源坐标起点，不是池化索引。
# 观察到空白只支持放置中文，不授权移除原有英文。
def _observed_blank_positions(blank,page,source,rw,rh):
    """Inspect integral-tile origins in the same forty-unit neighborhood.

    The legacy alignment grid can step over a short empty title cell. This
    fallback tests every pooled-tile boundary in a bounded local window; each
    accepted rectangle still passes clear(), source, frame and distance gates.
    Pooling uses any visible ink, so this does not downsample away linework.
    """
    if not blank.valid:return ()
    sx,sy=blank.sx,blank.sy;q=blank.integral
    nx=int(np.ceil((rw+2)*sx));ny=int(np.ceil((rh+2)*sy))
    xs=np.arange(max(0,int((source.min_x-40-rw-1)*sx)),
                 min(q.shape[1]-nx,int((source.max_x+40)*sx)+1))
    ys=np.arange(max(0,int((source.min_y-40-rh-1)*sy)),
                 min(q.shape[0]-ny,int((source.max_y+40)*sy)+1))
    if not len(xs) or not len(ys):return ()
    # At most the existing 1500-cell page grid, limited again by source bounds.
    x0=xs[None,:];y0=ys[:,None]
    ink=q[y0+ny,x0+nx].astype(np.int64)+q[y0,x0].astype(np.int64)-q[y0,x0+nx].astype(np.int64)-q[y0+ny,x0].astype(np.int64)
    iy,ix=np.nonzero(ink==0)
    positions=[(float(xs[x]/sx+1),float(ys[y]/sy+1),4) for y,x in zip(iy,ix)]
    positions.sort(key=lambda p:(_distance(source,TextBounds(p[0]-1,p[1]-1,p[0]+rw+1,p[1]+rh+1)),p[1],p[0]))
    # Keep spatially distinct origins, rather than flooding the fixed solver
    # pool with equivalent positions from one large connected white region.
    # 按距离及坐标稳定排序后，以空间桶去掉近似起点。
    # 候选池应覆盖多个实际空隙，不能全被同一片白区的相邻点占满。
    # 三单位桶是搜索多样性启发式，不放宽最终安全检查。
    # 上限防止大量等价空白把冲突修复预算耗在同一位置。
    selected=[];seen=set()
    for p in positions:
        bucket=(int(p[0]//3),int(p[1]//3))
        if bucket in seen:continue
        seen.add(bucket);selected.append(p)
        if len(selected)>=MAX_CANDIDATES:break
    return selected


# 有限字号、宽度、侧边和滑移量形成候选，不做逐像素穷举。
# 候选始终包含完整译文，无法适配的宽度直接跳过。
# tight 仅使用实际字形边界，observed 仅扩展可尝试起点。
# 二者都不能放松空白、原框、图框、纸张或距离约束。
# 旋转包围框先计算，接下来所有空白查询都针对最终占用。
def _enumerate_candidates(engine,page,entry,text,blank,source_boxes,*,tight=False,observed=False):
    """Slide complete rotated rectangles along all four nearby source sides.

    Fixed finite font/width/distance grids avoid pixel-scale exhaustive search.
    Candidate acceptance still uses the pooled full-resolution ink integral;
    sampling positions never samples away thin lines or unknown background.
    """
    b=entry.bounds;rotation=engine._dominant_rotation(entry)
    sw=b.max_x-b.min_x;sh=b.max_y-b.min_y
    # 初始字号由原文高度估计，同时尊重引擎最小字号。
    # 字号集合保留尝试顺序，并去除相同值以减少无效重复。
    # 多个宽度让较短单行与较窄多行都有机会进入相邻空隙。
    # 最大宽度按纸张比例限制，不因长译文横跨远处不相关区域。
    # 每种适配失败只影响当前组合，不删除条目或截断文字。
    initial=min(14.0,max(engine.min_font_size,sh*.78))
    sizes=tuple(dict.fromkeys((initial,max(engine.min_font_size,initial-2),
                              max(engine.min_font_size,min(initial,8)),engine.min_font_size)))
    normal=min(page.width*.30,max(sw,page.width*.12))
    widths=tuple(dict.fromkeys((normal,min(page.width*.38,normal*1.5),normal*.70,
                               max(engine.min_font_size*2,min(sw,normal)*.65))))
    found={}
    for size in sizes:
        for width in widths:
            try:
                if tight:
                    lines,fitted,w,h,origin=tight_text_fit(engine,text,width,page.height*.25,size)
                else:
                    lines,fitted,w,h=engine._fit_text(text,width,page.height*.25,size)
                    origin=None
            except ValueError:continue
            f=footprint(TextBounds(0,0,w,h),rotation)
            rw=f.max_x-f.min_x;rh=f.max_y-f.min_y
            # Edge-, centre- and trailing-edge alignment with bounded sliding
            # lets a caption enter the same title cell without crossing ink.
            # 沿原框首端、中心与末端对齐，再做有限横纵滑移。
            # 四个方向都围绕同一个英文锚点，不产生远端侧栏。
            # gap 是占用外框与原框间的源坐标距离。
            # observed 模式替换起点来源，但后续接受流程完全相同。
            xbase=(b.min_x,(b.min_x+b.max_x-rw)/2,b.max_x-rw)
            ybase=(b.min_y,(b.min_y+b.max_y-rh)/2,b.max_y-rh)
            xvalues=tuple(dict.fromkeys(x+shift for x in xbase for shift in (0,-4,4,-8,8,-16,16,-28,28)))
            yvalues=tuple(dict.fromkeys(y+shift for y in ybase for shift in (0,-4,4,-8,8,-16,16,-28,28)))
            positions=[]
            for gap in (3,5,7,9,12,18,30,39):
                positions.extend((x,b.max_y+gap,0) for x in xvalues)
                positions.extend((x,b.min_y-rh-gap,1) for x in xvalues)
                positions.extend((b.max_x+gap,y,2) for y in yvalues)
                positions.extend((b.min_x-rw-gap,y,3) for y in yvalues)
            if observed:positions=_observed_blank_positions(blank,page,b,rw,rh)
            if not len(positions):continue
            for x,y,side in _blank_positions(blank,page,positions,rw,rh):
                # Preserve the old page clamp, then validate the final footprint.
                x=max(2.0,min(x,page.width-rw-2));y=max(2.0,min(y,page.height-rh-2))
                bounds=TextBounds(x+rw/2-w/2,y+rh/2-h/2,x+rw/2+w/2,y+rh/2+h/2)
                box=_pad(footprint(bounds,rotation),1)
                if box.min_x<1 or box.min_y<1 or box.max_x>page.width-1 or box.max_y>page.height-1:continue
                # 先保证页内，再检验近邻、像素纯白和所有源框。
                # 源框列表可能含 manual 预留，因此人工锁定位置也被保护。
                # 候选坐标按千分之一去重只压缩重复搜索，不修改实际保存坐标。
                # rank 先距离，再字号、行数、方向和横向对齐偏差。
                # 不能把可读性排序放在安全过滤之前，以免高分危险框进入分配。
                distance=_distance(b,box)
                if distance>40 or not blank.clear(box):continue
                if any(_overlap(box,o) for o in source_boxes):continue
                key=tuple(round(v,3) for v in (bounds.min_x,bounds.min_y,bounds.max_x,bounds.max_y,fitted))
                rank=(distance,-fitted,len(lines),side,abs((x+rw/2)-(b.min_x+b.max_x)/2))
                value=_Candidate(bounds,lines,fitted,box,rank,origin)
                if key not in found or rank<found[key].rank:found[key]=value
    # Keep alternatives spread over space and size. Ninety-six near-identical
    # positions in one shared gap would make reassignment needlessly brittle.
    # 排序之后再按空间、宽度和字号分桶，保留不同可行布局形状。
    # 固定候选数是可维护的成本上限，不表示找到了全局最优排版。
    # 拥挤场景仍可能因有限预算留下待布局项，应如实保留原因。
    # 保留下来的每个候选都已经通过背景和源框检查。
    candidates=sorted(found.values(),key=lambda c:(c.rank,c.bounds.min_x,c.bounds.min_y))
    selected=[];buckets=set()
    for candidate in candidates:
        box=candidate.occupied
        bucket=(round(box.min_x/6),round(box.min_y/6),round((box.max_x-box.min_x)/12),round(candidate.size))
        if bucket in buckets:continue
        buckets.add(bucket);selected.append(candidate)
        if len(selected)>=MAX_CANDIDATES:break
    return tuple(selected)


# 求解从已有可行分配开始，只尝试增加尚未分配的中文。
# 冲突图按当前选中框惰性生成，避免预建所有候选的平方组合。
# 试探使用字典副本，失败不会改变已提交的 assignment。
# 一次增广只有把所有被挤出的项安置好才可提交。
# 因此不能为了新增一条中文而丢掉此前已有的成功条目。
# 固定轮次、深度与节点预算使结果不受机器瞬时速度影响。
def _solve_candidates(candidates,assignment,expand=None):
    """Conflict-graph augmentations; no accepted step drops a placed caption."""
    assignment=dict(assignment)
    # Source groups interact only when their candidate rectangles overlap.
    # Build this graph lazily against the currently chosen rectangles rather
    # than materializing a quadratic graph of every candidate combination.
    # budget 是本次增广共享计数，递归分支不能各自无限扩张。
    # expand 按需为冲突项补齐候选，未碰到的成功项不用枚举新位置。
    # protected 防止重新挤走当前递归链已经确定的项形成循环。
    # 先考虑阻挡数量少的候选，再用稳定 rank 选择。
    # 零阻挡候选可直接返回新的分配，不需继续试探。
    def augment(group,state,protected,depth,budget):
        if budget[0]<=0:return None
        budget[0]-=1
        if expand is not None:expand(group)
        choices=[]
        for candidate in candidates[group]:
            blockers=tuple(g for g,c in state.items() if g!=group and _overlap(candidate.occupied,c.occupied))
            if any(g in protected for g in blockers):continue
            choices.append((len(blockers),candidate.rank,candidate,blockers))
        choices.sort(key=lambda x:(x[0],x[1]))
        seen=set();attempts=0
        for count,_,candidate,blockers in choices:
            if count==0:
                result=dict(state);result[group]=candidate;return result
            if depth<=0 or count>3:continue
            # Distinct blocker sets and physical slots get a fair share of the
            # fixed budget, instead of revisiting one congested slot repeatedly.
            # 同一阻挡集合与近似空间格只尝试一次，避免重复无效修复。
            # 最多三个阻挡和有限 attempts 限制复杂冲突分支。
            # 被挤出的项按候选数少者优先安置，减少稀缺空隙被抢占。
            # 任一 displaced 无法重放，整个 trial 失败而不部分接受。
            # 这是局部有界修复，不能声称一般组合优化的全局最优。
            signature=(blockers,round(candidate.occupied.min_x/8),round(candidate.occupied.min_y/8))
            if signature in seen:continue
            seen.add(signature);attempts+=1
            if attempts>18:break
            trial={g:c for g,c in state.items() if g not in blockers}
            trial[group]=candidate
            for displaced in sorted(blockers,key=lambda g:(len(candidates[g]),g)):
                trial=augment(displaced,trial,protected|{group},depth-1,budget)
                if trial is None:break
            if trial is not None:return trial
        return None
    # 两轮让前一轮新增空间关系有机会帮助其它待布局项。
    # pending 按候选数及 group 索引排序，保证相同数据重复执行稳定。
    # 提交前明确要求原已分配集合是新集合的子集。
    # 本轮没有新增就停止，不重复消耗相同失败分支的预算。
    for _ in range(2):
        changed=False
        pending=sorted((g for g in candidates if g not in assignment),key=lambda g:(len(candidates[g]),g))
        for group in pending:
            if not candidates[group]:continue
            trial=augment(group,assignment,frozenset(),3,[MAX_REPAIR_NODES])
            if trial is not None and set(assignment)<=set(trial):
                assignment=trial;changed=True
        if not changed:break
    return assignment


def build_bilingual_layout(engine,page,entries,results,image,obstacles):
    """Return one complete item per eligible input, without masking English.

    ``results`` has already passed identity/provenance validation at the preview
    boundary. Manual reservations are identified by ``manual:<group>`` obstacle
    IDs and applied by the existing workflow after automatic construction.
    """
    # 上层已经验证所有结果，再按本轮实际 entries 缩小种子映射。
    # 过滤历史原文保留项不代表它们可以绕过上层身份验证。
    # legacy 调用跳过本双语分支，避免递归并保留历史可行布局作为种子。
    # 联合优化以原来成功的集合为基线，不从空白随机重排整页。
    from .annotation_preview import create_preview_layout
    groups={entry.group_id for entry in entries}
    # The preview boundary validates all historical results before applying
    # today's English routing projection. Excluded historical records may be
    # present in this already-validated mapping; keep them out of the seed only.
    results={group:result for group,result in results.items() if group in groups}
    # Legacy dispatch bypasses this new bilingual branch while retaining the
    # exact previously feasible placement algorithm as the optimization seed.
    baseline=create_preview_layout(engine,page,entries,results,image,False,obstacles,display_mode='legacy')
    blank=None
    source_boxes=[_pad(o.bounds,1.5) for o in obstacles] or [_pad(e.bounds,1.5) for e in entries]
    # 人工位置由 manual:<group> 障碍显式预留，优化器不重新接管它。
    # 自动阶段先清掉该项自己的自动框，之后工作流替换为用户原项。
    # 锁定不是安全豁免：人工项仍应由工作流按当前源图与边界复验。
    # 其它项的候选始终避开人工预留区域。
    locked={o.id[len('manual:'):] for o in obstacles if o.id.startswith('manual:')}
    options={};assignment={};items=list(baseline.items);expanded=set()
    # 仅对需要替换或首次安置的项惰性扩展候选。
    # 全页 BlankSpace 也延迟构造，减少无冲突页的额外工作。
    # 检查完整译文字体和长度后才生成候选。
    # 历史适配、紧字形适配、观察空格适配依次补充，安全标准不变。
    # 已有 previous 放到首位保留原位置的可行路径。
    # 预期的字体/适配异常只使该项无新候选，不改变原始译文。
    def expand(index):
        nonlocal blank
        if index in expanded:return
        expanded.add(index)
        entry=entries[index];item=items[index]
        if blank is None:blank=BlankSpace(image,page)
        try:
            ensure_font_supports(item.translated_text)
            if len(item.translated_text)>2000:raise ValueError('translation too long')
            choices=list(_enumerate_candidates(engine,page,entry,item.translated_text,blank,source_boxes))
            # Keep the historical pool first. A second bounded pool fits actual
            # glyph ink into short title cells without shrinking below six or
            # weakening any blank-space, frame, source or distance guard.
            choices.extend(_enumerate_candidates(engine,page,entry,item.translated_text,blank,source_boxes,tight=True))
            if not choices:
                choices.extend(_enumerate_candidates(engine,page,entry,item.translated_text,blank,source_boxes,tight=True,observed=True))
        except (ValueError,RuntimeError):choices=[]
        previous=assignment.get(index)
        if previous is not None:choices.insert(0,previous)
        options[index]=tuple(choices)
    for index,(entry,item) in enumerate(zip(entries,items)):
        result=results.get(entry.group_id)
        # 人工锁项不留一个同时参与分配的自动矩形，避免重复占空间。
        # 无可用翻译项沿用原状态，不参与联合修复。
        # 已有 inline 成功项直接作为种子，只有冲突到它才延迟枚举替代框。
        # 这样多数正常页面保持稳定位置，也减少无用字体度量。
        if entry.group_id in locked:
            # Existing workflow substitutes the immutable user item afterwards.
            # Do not leave an unassigned automatic rectangle consuming space.
            items[index]=replace(item,text_bounds=entry.bounds,lines=(),placement='needs_layout_review')
            continue
        if not preview_translation_available(result):continue
        if item.placement=='inline' and item.lines:
            previous=_candidate(item,entry.bounds)
            assignment[index]=previous
            options[index]=(previous,)
        else:
            expand(index)
    # Existing successful labels only need alternative positions if an actual
    # conflict augmentation reaches them. Uncontested pages retain the seed
    # without enumerating thousands of unused positions per existing label.
    assignment=_solve_candidates(options,assignment,expand)
    # 只有已提交分配写回布局，试探中的候选不会泄漏。
    # 成功后删除过时待布局警告，追加算法版本及实际邻近距离。
    # 基线原点与框、行和字号同时替换，保持绘制一致。
    # 该阶段强制无蒙版，后续可选遮挡不会重新决定中文位置。
    # 未分配项分别记录人工预留、预算冲突或无安全位置，避免混为翻译失败。
    for index,candidate in assignment.items():
        item=items[index]
        warnings=tuple(w for w in item.warnings if not w.startswith('needs_layout_review'))
        warnings=(*warnings,f'layout:{LAYOUT_VERSION}',f'layout_distance:{candidate.rank[0]:.3f}')
        items[index]=replace(item,text_bounds=candidate.bounds,lines=candidate.lines,font_size=candidate.size,
                             placement='inline',warnings=warnings,text_origin=candidate.origin,mask_decision=MaskDecision.NOT_REQUESTED,
                             mask_polygons=(),fallback_text_bounds=None)
    for index,item in enumerate(items):
        if index not in assignment and item.placement=='needs_layout_review':
            reason=('manual_position_reserved' if item.group_id in locked else
                    'candidate_conflict_budget' if options.get(index) else 'no_safe_nearby_slot')
            items[index]=replace(item,warnings=(*item.warnings,f'layout:{LAYOUT_VERSION}',f'layout_reason:{reason}'))
    return AnnotationLayout(page.document_id,page.page_index,page.width,page.height,page.coordinate_unit,
                            load_annotation_font(),False,tuple(items),True,page.width,'bilingual')
