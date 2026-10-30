# mechanical_drawing_translator/translation_convergence.py
"""Conservative known-source recovery and mechanical legend composition.

No reference bank or answers are loaded here. Whitespace restoration requires
one existing complete reviewed source, and never edits original OCR objects.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib,re,unicodedata
from .translation_phrase_memory import PhraseEnhancement,PhraseEvidence
from .translation_targets import TargetEntry,SelectionResult
from .semantic_grouping import SourceSpan

def key(text):return ' '.join(unicodedata.normalize('NFKC',text).upper().split())
def compact(text):return re.sub(r'\s+','',key(text))

# 复用只读取当前传入的词表视图，
# 不读取参考库或历史输出答案。
# 完整源去空白键可定位OCR粘连，
# 但多个不同中文时始终拒绝。
# 短单词不参与长句空白恢复。
class ConvergenceReuse:
    def __init__(self,glossary):
        self.complete={};self.exact={};self.material={}
        for p in glossary.reviewed_phrases():
            if len(p.targets)!=1:continue
            self.exact.setdefault(key(p.source),[]).append(p)
            # 空白恢复源至少两个词且十二字符，
            # 避免单个短工程代号误匹配。
            # 材料名称只从双语完整显式定义提取，
            # 代码是英中定义对齐的身份锚。
            # 任意括号说明不授予材料新词义。
            if len(key(p.source).split())>=2 and len(compact(p.source))>=12:
                self.complete.setdefault(compact(p.source),[]).append(p)
            # Explicit code (material) definitions in both languages. The code
            # anchors the alignment; arbitrary parenthetic clauses are excluded.
            m=re.fullmatch(r'(?:OR\s+)?([A-Z0-9.-]+)\s*\(([A-Z][A-Z ]{2,30})\)',key(p.source))
            if m:
                t=re.fullmatch(r'(?:或\s*)?'+re.escape(m[1])+r'\s*[（(]([^()（）]+)[）)]',p.targets[0])
                if t:self.material.setdefault(m[2],[]).append((p,t[1]))

    @staticmethod
    def unique(rows):
        return rows[0] if rows and len({p.targets[0] for p in rows})==1 else None

    @staticmethod
    def result(source,target,p,kind):
        ev=PhraseEvidence(source,target,p.entry_id,p.origin,kind)
        return PhraseEnhancement(target,(ev,),int(p.origin=='v1'),0,0,0,0)

    # 仅非空白字符相同且普通key不同，
    # 才报告需要恢复的已知完整短语。
    # 不修复拼写、数字或丢失条件，
    # 也不把目标中文拿来反推英文。
    def spacing_phrase(self,source):
        p=self.unique(self.complete.get(compact(source),()))
        return p if p and key(source)!=key(p.source) else None

    # 优先唯一完整表达空白变体，
    # 随后才处理有限机械图例。
    # 图形标记与等号关系原样保留，
    # 右侧必须是已知CONSTRUCTING DIMENSIONS。
    # 材料别名也要求唯一中文含义，
    # 没有命中就继续其它服务路线。
    def resolve(self,source):
        if len(source)>2000:return None
        p=self.spacing_phrase(source)
        if p:return self.result(source,p.targets[0],p,'convergence_spacing_only_complete_source')
        # Preserve the actual graphical marker and equality relation; only the
        # complete already-known right hand predicate is translated by reuse.
        m=re.fullmatch(r'\s*(\[\s*\]|\(\s*\))\s*-?\s*dimensions\s*=\s*([A-Za-z ]+)\s*',source,re.I)
        if m:
            p=self.unique(self.exact.get(key(m[2]),()))
            if p and key(m[2])=='CONSTRUCTING DIMENSIONS':
                return self.result(source,m[1]+'尺寸 = '+p.targets[0],p,'convergence_dimension_legend')
        rows=self.material.get(key(source),())
        if rows and len({target for _,target in rows})==1:
            p,target=rows[0];return self.result(source,target,p,'convergence_explicit_material_definition')
        return None

# 有限德语提示不是通用语言识别，
# 只作为拒绝英文模型误译的证据。
# RC/CL等不透明尺寸或热处理字段，
# 在语义身份未确定前保留原样。
# 这些拒绝不产生德语或缩写翻译能力。
def unsupported_input_warning(source):
    # Finite, distinctive language cues seen on actual non-English drawings.
    # This is not general language identification or German translation support.
    if re.search(r'\b(?:SCHNITT|FUEHRUNGSHUELSE|FÜHRUNGSHÜLSE|ERODIERRADIEN|SCHIEBERBLOCK|HALTERAHMEN|LAUFRICHTUNG|STUECK|SCHMIERNUTEN|MASSTAB|SIEHE)\b',key(source)):
        return 'convergence_unsupported_german_source_preserved'
    # Opaque dimensional/heat-treatment fields must not become ordinary words,
    # chemicals or satellites. Preserve them until their role is established.
    if re.fullmatch(r'\s*[()]?[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)[()]?\s*CL\s*',source,re.I) or re.fullmatch(r'\s*RC\s*\d+(?:\s*-\s*\d+)?\s*',source,re.I):
        return 'convergence_opaque_engineering_field_preserved'
    return None

def repair_entity_count(source,target):
    """An explicitly located physical feature count is not a count of types.

    This finite correction keeps the existing complete translation. It neither
    supplies missing instructions nor expands to arbitrary nouns/conditions.
    """
    # 实体计数修正仅针对明确HAVE句法，
    # 含否定、条件或类型词时完全跳过。
    # 英文必须恰好一个可支持的计数，
    # 中文只修对应“种”到“处”。
    # 不会补写不存在的加工要求。
    normalized=key(source)
    if re.search(r"\b(?:NOT|NO|EXCEPT|UNLESS|IF|WHEN|WITHOUT|NEVER|CANNOT|DON['’]T|DOESN['’]T|TYPES?|KINDS?)\b",normalized):return target
    matches=list(re.finditer(r'\bHAVE (TWO|THREE|FOUR) SLUG RETENTION FEATURES LOCATED\b',normalized))
    if len(matches)!=1:return target
    number={'TWO':'两','THREE':'三','FOUR':'四'}[matches[0][1]]
    wrong=number+'种防跳屑特征'
    n=target.count(wrong)
    if n==1 or (n==2 and number=='两' and 'BOTH RETENTION FEATURES' in normalized):
        return target.replace(wrong,number+'处防跳屑特征')
    return target

def recover_reviewed_selection(selection,glossary):
    resolver=ConvergenceReuse(glossary)
    entries=selection.candidate_entries
    # Rebuild atomic regions only when all their original non-whitespace
    # characters already belong to candidates. Shared sentence spans may thus
    # be reunited, but excluded/unknown characters cannot be silently borrowed.
    # 区域恢复先计算已有候选字符覆盖，
    # reviewed94组的原区域独立保护。
    # 仅所有非空白字符已属于候选的原区，
    # 才可作为完整恢复节点。
    # 禁止借入被筛除或未知字符，
    # 也拒绝核验字面与原字面不同的区域。
    coverage={};protected=set()
    for e in entries:
        if e.paragraph_id.startswith('reviewed94:'):protected.update(e.region_ids)
        for r in e.regions:
            spans=[s for s in e.spans if s.region_id==r.id] if e.spans else [SourceSpan(r.id,0,len(r.source_text))]
            coverage.setdefault(r.id,set()).update(i for s in spans for i in range(s.start,s.end))
    nodes=[]
    for r in selection.raw_regions:
        if r.id not in coverage or not r.source_text.strip():continue
        if r.id in protected or abs(r.rotation_degrees or 0)>5:continue
        if any(not c.isspace() and i not in coverage.get(r.id,set()) for i,c in enumerate(r.source_text)):continue
        if r.verified_source_text and compact(r.source_text)!=compact(r.verified_source_text):continue
        nodes.append(r)
    nodes.sort(key=lambda r:(r.document_id,r.page_index,r.bounds.min_y,r.bounds.min_x))
    # 仅同文档同页且由上向下相邻，
    # 以字高缩放垂距和水平对齐阈值。
    # 半宽重叠与有限对齐偏差共同限定，
    # 几何只提议连行，不证明语言完整。
    # 旋转大于五度的节点此前已排除。
    def follows(a,b):
        if a.document_id!=b.document_id or a.page_index!=b.page_index:return False
        h=max(a.bounds.max_y-a.bounds.min_y,b.bounds.max_y-b.bounds.min_y,1)
        overlap=min(a.bounds.max_x,b.bounds.max_x)-max(a.bounds.min_x,b.bounds.min_x)
        width=min(a.bounds.max_x-a.bounds.min_x,b.bounds.max_x-b.bounds.min_x)
        align=min(abs(a.bounds.min_x-b.bounds.min_x),abs(a.bounds.max_x-b.bounds.max_x),abs((a.bounds.min_x+a.bounds.max_x-b.bounds.min_x-b.bounds.max_x)/2))
        return 0<b.bounds.min_y-a.bounds.min_y<=2.5*h and overlap>=.5*max(width,1) and align<=2*h
    paths=[]
    from .knowledge_v2 import safety_warnings
    for start,r in enumerate(nodes):
        possible=[k for k in resolver.complete if k.startswith(compact(r.source_text))]
        if not possible:continue
        found=[];overflow=False
        # 搜索只沿已知完整源的前缀延伸，
        # 两区以上才构成连行恢复。
        # 合成英文必须有唯一已知中文，
        # 并重新通过知识安全警告检查。
        # 最多八区、两千字符与十六路径，
        # 超预算便放弃这一组而非猜首选。
        def visit(indices,source,keys):
            nonlocal overflow
            k=compact(source)
            if len(indices)>=2 and k in resolver.complete:
                p=resolver.unique(resolver.complete[k])
                if p and not safety_warnings(p.source,p.targets[0]):found.append(indices)
            if len(found)>16:overflow=True;return
            if len(indices)>=8 or len(source)>2000:return
            last=indices[-1]
            for nxt in range(last+1,len(nodes)):
                if not follows(nodes[last],nodes[nxt]):continue
                joined=source+' '+nodes[nxt].source_text
                remaining=[v for v in keys if v.startswith(compact(joined))]
                if remaining:visit((*indices,nxt),joined,remaining)
                if overflow:return
        visit((start,),r.source_text,possible)
        if not overflow:paths.extend(found)
    # A unique complete reviewed paragraph dominates its complete subclauses.
    # Alternative overlapping readings remain ambiguous and are all refused.
    # 唯一完整大段支配其严格子段，
    # 重叠但互不包含的解释全部拒绝。
    # 每个节点只能属于一个保留路径，
    # 避免同一原文被翻译或展示两次。
    paths=list(dict.fromkeys(paths))
    paths=[p for p in paths if not any(set(p)<set(q) for q in paths)]
    degree={}
    for p in paths:
        for i in p:degree[i]=degree.get(i,0)+1
    groups=[]
    # 组前后若还有条件、否定或续接词，
    # 不能截断后宣称完整说明。
    # 包围框内任何其它原区域也阻断，
    # 防止跨列跨段读序绕过中间文字。
    # 新SourceSpan覆盖每个真实原区全字面。
    for p in paths:
        if any(degree[i]!=1 for i in p):continue
        regions=tuple(nodes[i] for i in p);ids={r.id for r in regions}
        continuation=r'^\s*(?:DO\s+NOT|NOT|NO|EXCEPT|IF|UNLESS|UNTIL|WHEN|WITH|WITHOUT|AND|OR|PROVIDED)\b'
        if any(r.id not in ids and (follows(r,regions[0]) or follows(regions[-1],r))
               and re.match(continuation,r.source_text,re.I) for r in selection.raw_regions):continue
        x0=min(r.bounds.min_x for r in regions);x1=max(r.bounds.max_x for r in regions)
        y0=min(r.bounds.min_y for r in regions);y1=max(r.bounds.max_y for r in regions)
        if any(r.id not in ids and r.document_id==regions[0].document_id and r.page_index==regions[0].page_index
               and r.bounds.max_x>x0 and r.bounds.min_x<x1 and r.bounds.max_y>y0 and r.bounds.min_y<y1
               for r in selection.raw_regions):continue
        spans=tuple(SourceSpan(r.id,0,len(r.source_text)) for r in regions)
        pid='convergence2:'+hashlib.sha256('|'.join(r.id for r in regions).encode()).hexdigest()
        groups.append(TargetEntry(regions,'translatable','完整已知专业说明的原始区域全覆盖恢复',spans,pid,('convergence_complete_reviewed_source_regions_v2',)))
    # Do not split any old candidate while leaving part of its source behind.
    # 恢复组不能切走旧候选的一部分，
    # 循环移除所有残留部分源的组合。
    # 替换同时更新候选与展示全集，
    # 按group_id去重保持一组只输出一次。
    # raw_regions、分类决定与统计保持原值，
    # 这一步只改变句界关联而非OCR识别。
    while groups:
        covered={rid for g in groups for rid in g.region_ids}
        incomplete=set().union(*(set(e.region_ids) for e in (*entries,*selection.all_entries) if covered.intersection(e.region_ids) and not set(e.region_ids)<=covered))
        if not incomplete:break
        groups=[g for g in groups if not incomplete.intersection(g.region_ids)]
    if not groups:return selection
    by_region={rid:g for g in groups for rid in g.region_ids}
    def apply(rows):
        out=[];seen=set()
        for e in rows:
            replacements=[by_region[rid] for rid in e.region_ids if rid in by_region]
            for v in replacements or [e]:
                if v.group_id not in seen:out.append(v);seen.add(v.group_id)
        return tuple(out)
    return SelectionResult(selection.raw_regions,selection.decisions,apply(entries),apply(selection.all_entries),selection.counts)
