# mechanical_drawing_translator/reviewed94.py
"""User-reviewed V2 addendum: retain evidence and keep instance authority bounded."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from .resource_paths import package_resource

# Reviewed94是独立用户审核增量，
# 不是全部94条一律批准的词库。
# 正确、错误、无需翻译与未定，
# 在导入时各保留真实资格。
# 原人工签名为空也不会由机器补填。
VERSION = 'reviewed94.1'

def _text(value):
    return value.strip() if isinstance(value, str) else ''

def import_review_rows(rows):
    """Normalize typed extracted cells, without inventing reviewer identity or approval."""
    # 导入拷贝每行不覆盖原表数据，
    # unit_id必须保留原product身份且唯一。
    # 只接受用户表的有限决策枚举，
    # 空英文或无页码不具备可追溯范围。
    result=[]; seen=set()
    for row in rows:
        record=dict(row)
        identifier=record.get('unit_id')
        if not isinstance(identifier,str) or not identifier.startswith('product:') or identifier in seen:
            raise ValueError('invalid/duplicate original review ID')
        seen.add(identifier)
        decision=record.get('decision')
        if decision not in {'正确','错误','无需翻译','不确定','待填写'}:
            raise ValueError('invalid review judgment')
        if not _text(record.get('source')) or not isinstance(record.get('page'),int):
            raise ValueError('missing source/page')
        target=record.get('target')
        valid_target=isinstance(target,str) and bool(re.search(r'[\u3400-\u9fff]',target))
        # “错误”且有中文目标才算可用纠正，
        # 没有可靠目标保持pending_invalid_target。
        # “正确”仅接受现有表达，
        # “无需翻译”明确保留原文。
        # 未定或待填写不能自动转为批准。
        record['qualification']=('accepted_existing' if decision=='正确' else
            'preserve_original' if decision=='无需翻译' else 'pending' if decision in {'不确定','待填写'} else
            'accepted_correction' if valid_target else 'pending_invalid_target')
        result.append(record)
    # Same target across distinct sources on one page can be a paragraph shared
    # by several fragments. Keep it out of automatic per-entry import.
    # 同页多个不同源共享一个中文，
    # 可能是跨行段落授权而非各碎片答案。
    # 这些行改为shared_context，
    # 防止每个碎片重复贴整段中文。
    # 后续只能按完整组证据恢复一次。
    targets={}
    for r in result:
        if r['qualification']=='accepted_correction':
            key=(r['pdf'],r['page'],_text(r['target']))
            targets.setdefault(key,[]).append(r)
    for members in targets.values():
        if len({r['source'] for r in members})>1:
            for r in members:r['qualification']='shared_context'
    return result

@dataclass(frozen=True)
# 候选保留实际授权行号和范围，
# action区分翻译与保留原文决定。
# source可带单独核实的corrected_source，
# 不会据目标中文自动生成英文修正。
class ReviewedCandidate:
    target: str
    source: str
    evidence_id: str
    rows: tuple[int,...]
    scope: str
    action: str = 'translate'

class ReviewedKnowledge:
    def __init__(self,data):
        if data.get('schema')!='MDT.reviewed94.1': raise ValueError('invalid reviewed addendum schema')
        # 资源同时绑定父V2指纹与原表摘要，
        # 增量有自己的稳定JSON指纹。
        # exact与instance分别建立索引，
        # 实例授权不能晋升成全局精确项。
        self.parent=data['parent']; self.source_sha256=data['source_sha256']
        self.groups=data.get('groups',[])
        self.records=data['records']; self._exacts={}; self._instances={}
        seen=set()
        # ID唯一且范围只能exact/instance，
        # 英文、中文及来源行号都不可缺。
        # 全局精确英文有多项便拒绝歧义，
        # 实例项还需PDF内容摘要与页码。
        for r in self.records:
            if r['id'] in seen or r['scope'] not in {'exact','instance'}:
                raise ValueError('invalid/duplicate reviewed knowledge')
            seen.add(r['id'])
            if not _text(r.get('source')) or not _text(r.get('target')) or not r.get('rows'):
                raise ValueError('incomplete reviewed knowledge')
            if r['scope']=='exact':
                if r['source'] in self._exacts: raise ValueError('ambiguous exact correction')
                self._exacts[r['source']]=r
            else:
                if len(r.get('pdf_sha256',''))!=64 or not isinstance(r.get('page'),int):
                    raise ValueError('instance missing content identity')
                self._instances.setdefault((r['page'],r['source']),[]).append(r)
        self.fingerprint=hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()

    @staticmethod
    # 有独立corrected_source才采用修正送译，
    # 否则使用审核时记录的原英文。
    # 此函数只构造候选值，
    # 工程守卫与机器来源仍由服务负责。
    def _candidate(r):
        return ReviewedCandidate(r['target'],r.get('corrected_source') or r['source'],
                                 r['id'],tuple(r['rows']),r['scope'],r.get('action','translate'))

    def exact(self,source):
        r=self._exacts.get(source)
        return self._candidate(r) if r else None

    def resolve_entries(self,entries,contexts,*,use_verified_source=False):
        if len(entries)!=len(contexts):raise ValueError('reviewed context count mismatch')
        # Hash once per request, never authorize by a reusable basename or by
        # stale file metadata. This reads bytes only, no additional PDF pages.
        # 每请求对真实PDF字节最多哈希一次，
        # 不按可复用文件名或mtime授予范围。
        # 该读取不打开新增PDF页面内容，
        # 仅核实已有实例授权的文件身份。
        fingerprints={}; results=[]
        for entry,context in zip(entries,contexts):
            candidate=self.exact(entry.translation_source_text if use_verified_source else entry.source_text)
            # 先保留可能的全局exact候选，
            # 实例再要求原页、原字面与原框。
            # 调用者context的页必须等于当前页，
            # 不能借别页相同句子的审核决定。
            records=self._instances.get((entry.page_index+1,entry.source_text),())
            if records and context is not None and context.page==entry.page_index+1:
                actual={k:getattr(entry.bounds,k) for k in ('min_x','min_y','max_x','max_y')}
                if context.bounds==actual:
                    path=str(Path(context.pdf).resolve())
                    if path not in fingerprints:
                        try:fingerprints[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        except OSError:fingerprints[path]=''
                    # 文件SHA256必须完全相同，
                    # 坐标只容忍0.001序列化舍入。
                    # 多条匹配时不选第一个，
                    # 保持明确的唯一实例授权边界。
                    matches=[r for r in records if r['pdf_sha256']==fingerprints[path] and
                        all(abs(r['bounds'][k]-actual[k])<=.001 for k in actual)]
                    if len(matches)==1:candidate=self._candidate(matches[0])
            results.append(candidate)
        return tuple(results)

    def merge_entries(self,entries,pdf_path,*,preserve_semantic_units=False):
        """Recover only explicitly reviewed groups on the original PDF content.

        Original regions/spans remain intact. A complete group is emitted once;
        partial or ambiguous membership fails closed, without copying its answer.
        Optional semantic_units are the original inputs for inspection only:
        the reviewed whole-paragraph answer remains one translation authority.
        """
        if not entries or not self.groups:return tuple(entries)
        # 组级恢复先限定当前页，
        # 再核实PDF实际内容身份。
        # 没有组、读取失败或无适用组，
        # 都保持输入条目原样返回。
        applicable=[g for g in self.groups if g['page']==entries[0].page_index+1]
        if not applicable:return tuple(entries)
        try:fingerprint=hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
        except OSError:return tuple(entries)
        from .translation_targets import TargetEntry
        from .semantic_grouping import SourceSpan
        replacements={};consumed=set()
        for group in applicable:
            if group['pdf_sha256']!=fingerprint:continue
            # 每个组成员须在原条目唯一找到，
            # 英文、文档、页与边框同时相符。
            # 缺一项、重复成员或已消费成员，
            # 都使整个组保持拒绝而非局部导入。
            indices=[]
            for member in group['members']:
                matches=[i for i,e in enumerate(entries) if e.source_text==member['source'] and
                    e.page_index==group['page']-1 and e.document_id==entries[0].document_id and
                    all(abs(getattr(e.bounds,k)-member['bounds'][k])<=.001 for k in member['bounds'])]
                if len(matches)!=1:break
                indices.append(matches[0])
            if len(indices)!=len(group['members']) or len(set(indices))!=len(indices) or consumed.intersection(indices):continue
            members=[entries[i] for i in indices]
            # 区域按原ID去重不改内容，
            # spans保留原范围或补真实全区范围。
            # semantic_units可保存原句供详情，
            # 整段中文授权仍只属于完整组一次。
            regions=tuple(dict((r.id,r) for e in members for r in e.regions).values())
            spans=tuple(s for e in members for s in (e.spans or tuple(SourceSpan(r.id,0,len(r.source_text)) for r in e.regions)))
            merged=TargetEntry(regions,'translatable','用户审核的完整句/标题组合',spans=spans,
                paragraph_id='reviewed94:'+group['id'],boundary_notes=('reviewed94_group:'+group['id'],),
                semantic_units=tuple(members) if preserve_semantic_units else ())
            replacements[min(indices)]=merged;consumed.update(indices)
        return tuple(replacements[i] if i in replacements else e for i,e in enumerate(entries) if i not in consumed or i in replacements)

    def recover_selection(self,selection,pdf_path,*,preserve_semantic_units=False):
        from .translation_targets import SelectionResult
        merged=self.merge_entries(selection.candidate_entries,pdf_path,
                                  preserve_semantic_units=preserve_semantic_units)
        if merged==selection.candidate_entries:return selection
        existing={e.group_id:e for e in selection.candidate_entries}
        # 候选恢复与展示全集同时调整，
        # 不影响的原条目保持原位。
        # non_translatable不因组覆盖被强行改判，
        # 合并组按group_id只展示一次。
        # 原raw_regions、decisions与counts保留。
        affected={rid for e in merged if e.group_id not in existing for rid in e.region_ids}
        displayed=[];emitted=set()
        for original in selection.all_entries:
            if original.category=='non_translatable' or not affected.intersection(original.region_ids):
                displayed.append(original);continue
            for entry in merged:
                if set(entry.region_ids).intersection(original.region_ids) and entry.group_id not in emitted:
                    displayed.append(entry);emitted.add(entry.group_id)
        return SelectionResult(selection.raw_regions,selection.decisions,merged,tuple(displayed),selection.counts)

def load_packaged():
    from .knowledge_v2_runtime import MANIFEST_SHA256
    path=package_resource('translation', 'knowledge_v2', 'reviewed94.json')
    # 包内增量先核对独立固定摘要，
    # 再核对父V2清单指纹。
    # 换一份父资源不能沿用旧增量授权，
    # 字节不匹配明确报错而非静默跳过。
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ADDENDUM_SHA256:raise ValueError('reviewed94 knowledge hash mismatch')
    data=json.loads(raw)
    if data['parent']!=MANIFEST_SHA256:raise ValueError('reviewed94 parent V2 mismatch')
    return ReviewedKnowledge(data)

ADDENDUM_SHA256 = '2d072fa4c15b3391f9b8947b7dfcf8985acf11c1a12edb361c6187102691de97'
