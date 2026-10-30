# mechanical_drawing_translator/translation_aligned_edit.py
"""Conservative source-aligned editing of globally reusable atomic terms.

This is an inference-time locator/editor, not a trained APE model or a sentence
verifier. Exact memory answers and instance approvals never become terms here.
The caller must validate the full, original English against the edited Chinese.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from .translation_backend import AlignedGeneration
from .translation_glossary import Glossary


@dataclass(frozen=True)
# 术语定位保留源字符起止与词条ID，
# 不保存也不推测图纸字符坐标。
# LocalEdit再携带目标文本区间，
# before/after供审计展示真实替换。
# EditProposal只是待验证的新中文。
class TermSpan:
    source: str
    start: int
    end: int
    target: str
    entry_id: str
    origin: str


@dataclass(frozen=True)
class LocalEdit:
    term: TermSpan
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    before: str
    after: str


@dataclass(frozen=True)
class EditProposal:
    text: str
    edits: tuple[LocalEdit, ...]


class AlignedTermEditor:
    def __init__(self, glossary: Glossary):
        # 候选只来自全局合资格术语，
        # 精确整句和局部实例不升格成词。
        # 同英文的全部读法必须只有同一中文，
        # 目标还限定一至十六个汉字。
        # runtime挂起条目连编辑词表也不进入。
        rows = {}
        runtime = getattr(glossary, 'runtime', None)
        self._hold = getattr(runtime, 'policy_hold', None)
        for p in glossary.reviewed_phrases():
            rows.setdefault(' '.join(p.source.upper().split()), []).append(p)
        self._patterns = []
        for key, readings in rows.items():
            atoms = [p for p in readings if p.kind == 'term' and len(set(p.targets)) == 1]
            if not atoms:
                continue
            p = atoms[0]
            if (self._hold and self._hold(p.source)) or not re.fullmatch(r'[\u4e00-\u9fff]{1,16}', p.targets[0]):
                continue
            if any(set(q.targets) != {p.targets[0]} for q in readings):
                continue
            pattern = re.compile(r'(?<![A-Za-z0-9])' +
                                 r'\s+'.join(re.escape(w) for w in key.split()) +
                                 r'(?![A-Za-z0-9])', re.I)
            self._patterns.append((pattern, p))

    # 输入含东亚语言或超两千字即拒绝，
    # 英文子串按字母数字边界完整匹配。
    # 最长术语先占据重叠源跨度，
    # 不把一个专业词拆成多个短词重复改。
    # 返回值仍按原文顺序排列。
    def terms(self, source: str) -> tuple[TermSpan, ...]:
        if (not isinstance(source, str) or len(source) > 2000
                or re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', source)
                or (self._hold and self._hold(source))):
            return ()
        found = [TermSpan(m[0], m.start(), m.end(), p.targets[0], p.entry_id, p.origin)
                 for pattern, p in self._patterns for m in pattern.finditer(source)]
        longest = []
        for item in sorted(found, key=lambda p: (-(p.end-p.start), p.start)):
            if not any(item.start < x.end and item.end > x.start for x in longest):
                longest.append(item)
        return tuple(sorted(longest, key=lambda p: p.start))

    def edit(self, source: str, value: AlignedGeneration) -> EditProposal | None:
        terms = self.terms(source)
        target = value.generation.text
        # 输入输出pieces数与跨度数量先对齐，
        # 注意力必须每目标piece一行。
        # 任一维度缺失、空序列或截断输出，
        # 均不能支撑安全的局部编辑。
        # 原generation仍可走服务原验证路径。
        ns, nt = len(value.source_tokens), len(value.target_tokens)
        if (not terms or value.generation.hit_output_limit
                or len(value.source_spans) != ns or len(value.target_spans) != nt
                or len(value.attention) != nt or not ns or not nt):
            return None
        # 每权重必须有限且在零到一之间，
        # 各行概率质量允许小幅数值误差。
        # 跨度须在真实文本内部且不重叠，
        # 不会手工修补不一致的模型对齐。
        for row in value.attention:
            if (len(row) != ns or any(not math.isfinite(v) or not 0 <= v <= 1 for v in row)
                    or not .95 <= sum(row) <= 1.05):
                return None
        for spans, text in ((value.source_spans, source), (value.target_spans, target)):
            if any(not 0 <= a <= b <= len(text) for a,b in spans):
                return None
            if any(spans[i][1] > spans[i+1][0] for i in range(len(spans)-1)):
                return None
        edits = []
        for term in terms:
            # A declared correct rendering may be elsewhere by legal reordering;
            # never remove it or force a duplicate just to satisfy a local slot.
            # 已声明的正确词可能合法换序，
            # 整句任意位置存在就不强制再插一遍。
            # 源pieces覆盖必须恰好等于术语字面，
            # 不接受吞入相邻对象或修饰词的分片。
            if term.target in target:
                continue
            indices = [i for i,(a,b) in enumerate(value.source_spans)
                       if a < term.end and b > term.start]
            if not indices:
                continue
            a,b = value.source_spans[indices[0]][0], value.source_spans[indices[-1]][1]
            if source[a:b].strip().upper() != term.source.upper():
                continue
            # 计算目标piece指向当前源术语的总质量，
            # 强对齐块要求连续且不超过八piece。
            # 块外对该术语的注意力不得过高，
            # 防止同一术语被拆散在多个目标位置。
            mass = [sum(row[i] for i in indices) for row in value.attention]
            chosen = [j for j,v in enumerate(mass) if v >= .75]
            if (not chosen or len(chosen) > 8 or chosen != list(range(chosen[0], chosen[-1]+1))
                    or any(v > .25 for j,v in enumerate(mass) if j not in chosen)):
                continue
            # Reverse agreement prevents consuming a token shared with another
            # source object. Every source piece must be accounted for locally.
            if any(sum(row[i] for row in value.attention) < .1 or
                   sum(value.attention[j][i] for j in chosen) < .75 *
                   sum(row[i] for row in value.attention) for i in indices):
                continue
            if any(sum(value.attention[j][i] for j in chosen) > .25 *
                   sum(row[i] for row in value.attention)
                   for i in range(ns) if i not in indices):
                continue
            # 反向一致性限制共享对象的误吞，
            # 每个源piece都应主要落入同一目标块。
            # 被替换中文仅允许短纯汉字段，
            # 禁止擦掉数字、标点、材料ID，
            # 以及否定、情态、顺序和连接词。
            start, end = value.target_spans[chosen[0]][0], value.target_spans[chosen[-1]][1]
            before = target[start:end]
            # Never erase punctuation, numbers, materials IDs, polarity, modal,
            # order, or connection/attachment words as a terminology operation.
            if (not re.fullmatch(r'[\u4e00-\u9fff]{1,16}', before)
                    or re.search(r'[不无未非须应必可允禁先后前再且和及或于在从向到对把将]', before)):
                continue
            if any(start < e.target_span[1] and end > e.target_span[0] for e in edits):
                continue
            # 目标编辑区间不得相互重叠，
            # 每条完整原句最多两个局部修改。
            # 从右向左应用保证旧跨度不位移，
            # 返回全部before/after证据。
            # 服务还须对整个源目标独立检查。
            edits.append(LocalEdit(term, (term.start, term.end), (start,end), before, term.target))
            if len(edits) == 2:
                break
        if not edits:
            return None
        text = target
        for e in sorted(edits, key=lambda e:e.target_span[0], reverse=True):
            text = text[:e.target_span[0]] + e.after + text[e.target_span[1]:]
        return EditProposal(text, tuple(edits))
