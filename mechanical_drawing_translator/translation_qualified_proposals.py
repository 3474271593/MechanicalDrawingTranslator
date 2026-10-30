# mechanical_drawing_translator/translation_qualified_proposals.py
"""Full-coverage proposals from qualified knowledge, never evaluation answers.

Complete reviewed expressions keep their original scope. D6/V1 instruction
atoms reuse the existing finite parser. The only new lexical normalization is
the English function abbreviation INCL. / INCLUDING; it adds no technical
meaning. Explicit AND/THEN joins are accepted only when every operand is known.
An unsupported assessment is distinct from a verified candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import re
import unicodedata

from .mechanical_instructions import compose_instruction, parse_instruction, validate_instruction
from .translation_convergence import compact
from .translation_phrase_memory import PhraseEnhancement, PhraseEvidence


# 仅INCL.与INCLUDING作为功能缩写归一，
# 不引入新的专业名词译法。
# 其它内部标点与非空白字符仍重要，
# 源规范化不修改实际OCR记录。
def _key(text: str) -> str:
    text = ' '.join(unicodedata.normalize('NFKC', text).upper().split())
    return re.sub(r'(?<![A-Z0-9])INCL\.(?![A-Z0-9])', 'INCLUDING', text)


def _target(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).rstrip('。.')


@dataclass(frozen=True)
# 局部约束保存原文字符位置与来源，
# 只证明某个已知片段被看见。
# 它不认证其余未覆盖英文，
# 不能让未知整句因含已知词而通过。
class ScopedConstraint:
    source: str
    start: int
    end: int
    targets: tuple[str, ...]
    entry_id: str | None
    origin: str


@dataclass(frozen=True)
# 操作数必须完整消费对应源区间，
# targets列明可接受整段中文。
# instruction区分有限类型AST与已知整句，
# 两种证据采取不同目标验证方式。
class CoverageOperand:
    source: str
    start: int
    end: int
    translated_text: str
    targets: tuple[str, ...]
    evidence: tuple[PhraseEvidence, ...]
    instruction: bool = False


@dataclass(frozen=True)
class CoveragePlan:
    operands: tuple[CoverageOperand, ...]
    relation: str


@dataclass(frozen=True)
# supported与warnings同时解释覆盖，
# False配空警告属于无覆盖。
# True配空警告才表示此语法未见问题，
# 仍不替代数值、术语或人工审核。
class ProposalAssessment:
    supported: bool
    warnings: tuple[str, ...]


class QualifiedProposalEngine:
    """One reusable knowledge index; no model calls or mutable runtime data."""

    def __init__(self, glossary, phrase_memory=None):
        self.glossary = glossary
        runtime = getattr(glossary, 'runtime', None)
        self._hold = getattr(runtime, 'policy_hold', None)
        self._knowledge_ids = {}
        self._evidence_targets = {}
        self._count_repairs = set()
        # 索引只收exact和term已知来源，
        # runtime挂起在构建时排除。
        # phrase_memory只可提供同ID满分非近邻，
        # 保留现有范围内的D7覆盖决策。
        # 不调用enhance或任何模型补词。
        rows = {}
        for phrase in glossary.reviewed_phrases():
            if phrase.kind not in {'exact', 'term'} or not phrase.targets:
                continue
            if self._held(phrase.source):
                continue
            knowledge_id = None
            if phrase_memory is not None:
                # Keep the existing scoped runtime choice, exactly as the core
                # composer does. A near match or a different entry cannot grant
                # a new target to this complete source expression.
                resolution = phrase_memory.resolve_clause(phrase.source)
                if (resolution is not None and resolution.entry_id == phrase.entry_id
                        and resolution.score == 1.0 and resolution.match_kind != 'near_clause'):
                    targets = (phrase.targets if resolution.translated_text == phrase.targets[0]
                               else (resolution.translated_text,))
                    phrase = replace(phrase, targets=targets, origin=resolution.origin)
                    knowledge_id = resolution.knowledge_id
            if knowledge_id is None and runtime is not None and hasattr(runtime, 'knowledge_for_ref'):
                knowledge = runtime.knowledge_for_ref('v1_raw:' + phrase.entry_id)
                if knowledge is not None:
                    knowledge_id = knowledge['id']
            # 证据原始首选译文单独保存，
            # 有限计数修复只新增受控可接受实现。
            # 原资源与原审核文字不被改写，
            # 采用修复时另有machine证据说明。
            identity = (phrase.source, phrase.entry_id)
            self._knowledge_ids[identity] = knowledge_id
            self._evidence_targets[identity] = phrase.targets[0]
            # Final mode inherits the existing convergence count correction.
            # Its narrowly source-bound realization must survive the later
            # complete-target gate; raw source resources remain immutable.
            from .translation_convergence import repair_entity_count
            repaired = tuple(repair_entity_count(phrase.source, t) for t in phrase.targets)
            if repaired != phrase.targets:
                self._count_repairs.add(identity)
                phrase = replace(phrase, targets=tuple(dict.fromkeys((*repaired, *phrase.targets))))
            rows.setdefault(_key(phrase.source), []).append(phrase)
        # 同英文多条记录只接受明确包含关系，
        # 一条术语的同义集合可被保留。
        # 不相关记录不能取并集扩充译法，
        # 有冲突key直接登记拒绝集合。
        self._index = {}
        self._conflicts = set()
        for key, phrases in rows.items():
            # Multiple declared synonyms within one term are admissible. An
            # unrelated row cannot silently enlarge that equivalence class.
            accepted = {_target(t) for t in phrases[0].targets}
            if all({_target(t) for t in p.targets} <= accepted for p in phrases):
                self._index[key] = phrases[0]
            else:
                self._conflicts.add(key)
        # 去空白索引要求唯一完整token范围，
        # 数字表达禁止拼接或重新切分。
        # 不同英文去空白后相同视作歧义，
        # 不能借空格修复猜其中一种意义。
        compact_sources = {}
        for key in rows:
            compact_sources.setdefault(compact(key), set()).add(key)
        self._compact_index = {}
        self._compact_conflicts = set()
        for joined, keys in compact_sources.items():
            # Whitespace cannot disambiguate different complete token scopes.
            # Numeric components are never concatenated or resegmented.
            if len(keys) != 1 or any(key in self._conflicts for key in keys):
                self._compact_conflicts.add(joined)
                continue
            key = next(iter(keys))
            phrase = self._index.get(key)
            if phrase is not None and phrase.kind == 'exact' and not re.search(r'\d', key):
                self._compact_index[joined] = phrase
        # 局部匹配按完整字面跨度建正则，
        # 内部标点与空白边界仍被保留。
        # 唯一开放INCL./INCLUDING等价，
        # 没有模糊OCR或词子串修复。
        self._patterns = []
        for key, phrase in self._index.items():
            # Literal full spans only. Internal punctuation and whitespace are
            # retained; no fuzzy OCR or substring token matching is permitted.
            parts = key.split(' ')
            body = r'\s+'.join(re.escape(p) for p in parts)
            body = body.replace('INCLUDING', r'(?:INCLUDING|INCL\.)')
            self._patterns.append((re.compile(r'(?<![A-Z0-9])' + body + r'(?![A-Z0-9])', re.I), phrase))

    def _held(self, source):
        return bool(self._hold and self._hold(source))

    # 先只清理当前源区间两端空白，
    # start/end继续指向真实源字符位置。
    # 冲突key优先拒绝，
    # 不会在空白兼容路径选择另一义项。
    def _operand(self, source, start, end):
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        text = source[start:end]
        normalized = _key(text)
        if normalized in self._conflicts:
            return None
        phrase = self._index.get(normalized)
        whitespace_only = False
        if phrase is None:
            joined = compact(normalized)
            if joined in self._compact_conflicts:
                return None
            phrase = self._compact_index.get(joined)
            if phrase is not None:
                # Existing function-word canonicalization is shared with the
                # ordinary index. Beyond that, every non-whitespace character
                # must match one unique complete qualified expression.
                # 空白修复也必须保留AND/THEN次数与顺序，
                # 不能把连接词粘连造成的新范围忽略。
                # 非空白字符相同仍不足以重新划分句义，
                # 需要同一完整合资格表达的唯一证据。
                expected_joins = re.findall(r'\s+(AND|THEN)\s+', _key(phrase.source))
                actual_joins = re.findall(r'\s+(AND|THEN)\s+', normalized)
                if actual_joins != expected_joins:
                    return None
                whitespace_only = True
        if phrase is not None:
            evidence = PhraseEvidence(phrase.source if whitespace_only else text,
                                      self._evidence_targets[(phrase.source, phrase.entry_id)],
                                      phrase.entry_id, phrase.origin,
                                      'qualified_complete_source',
                                      self._knowledge_ids.get((phrase.source, phrase.entry_id)))
            evidence = (evidence,)
            if whitespace_only:
                evidence += (PhraseEvidence(text, phrase.targets[0], None, 'machine',
                                            'qualified_whitespace_only', 'qualified-proposals.1'),)
            if (phrase.source, phrase.entry_id) in self._count_repairs:
                evidence += (PhraseEvidence(text, phrase.targets[0], None, 'machine',
                                            'qualified_existing_count_repair', 'convergence.2'),)
            return CoverageOperand(text, start, end, phrase.targets[0], phrase.targets, evidence)
        # Reuse the existing typed action/object parser, including its original
        # qualified atoms and refusal rules; do not invent a second word grammar.
        # 没有完整已知整句时才借有限指令AST，
        # 仅single操作可成为叶子。
        # 不会另造第二套技术词义解析器，
        # 原mechanical_instructions拒绝规则仍有效。
        ast = parse_instruction(text, self.glossary)
        if ast is None or ast.relation != 'single':
            return None
        composition = compose_instruction(text, self.glossary)
        if composition is None:
            return None
        return CoverageOperand(text, start, end, composition.translated_text,
                               (composition.translated_text,), composition.evidence, True)

    def plan(self, source: str) -> CoveragePlan | None:
        if not isinstance(source, str) or not source.strip() or len(source) > 2000 or self._held(source):
            return None
        complete = self._operand(source, 0, len(source))
        if complete is not None:
            return CoveragePlan((complete,), 'single')
        # 整串已知表达优先于拆AND/THEN，
        # 未知组合只允许单一种连接符。
        # 不带括号的条件附着范围不明时拒绝，
        # 前导禁止和中途禁止THEN也保持禁区。
        # 分隔符超过16个超出搜索预算。
        separators = list(re.finditer(r'\s+(AND|THEN)\s+', source, re.I))
        if not separators or len(separators) > 16 or len({m[1].upper() for m in separators}) != 1:
            return None
        # Qualifier attachment across an unparenthesized chain is deliberately
        # unsupported. BEFORE inside a complete reviewed operand is preserved.
        if re.search(r'\b(?:IF|UNLESS|EXCEPT|ONLY|WHEN|WITHOUT)\b', source, re.I):
            return None
        if re.match(r'\s*(?:DO NOT|MUST NOT|SHALL NOT)\b', source, re.I):
            return None
        if re.search(r'\b(?:DO NOT|MUST NOT|SHALL NOT)\b.+\bTHEN\b', source, re.I):
            return None
        relation = separators[0][1].lower()

        @lru_cache(None)
        # 记忆化搜索必须覆盖直到源串末端，
        # 每个前段都成为完整已知操作数。
        # 路径不超过八叶子、十六候选，
        # 超出预算不会只保留首个解。
        # 含义比较还包括原源跨度，
        # 同中文但不同分段也属于不确定。
        def visit(start):
            found = []
            leaf = self._operand(source, start, len(source))
            if leaf is not None:
                found.append((leaf,))
            for separator in separators:
                if separator.start() <= start:
                    continue
                leaf = self._operand(source, start, separator.start())
                if leaf is None:
                    continue
                for rest in visit(separator.end()):
                    if len(rest) < 8:
                        found.append((leaf, *rest))
                    if len(found) > 16:
                        return tuple(found)
            return tuple(found)

        paths = visit(0)
        if not paths or len(paths) > 16:
            return None
        meanings = {tuple((_target(p.translated_text), p.start, p.end) for p in path) for path in paths}
        return CoveragePlan(paths[0], relation) if len(meanings) == 1 else None

    def propose(self, source: str) -> PhraseEnhancement | None:
        plan = self.plan(source)
        if plan is None:
            return None
        values = [p.translated_text for p in plan.operands]
        # Independent complete declarations keep a visible clause boundary;
        # joining with 和 can incorrectly attach the second process as a noun
        # modifier of a dimension statement. THEN remains explicitly ordered.
        # 并列完整断言用分号保留边界，
        # 避免“和”把第二个过程当名词修饰。
        # THEN用明确先/再保留顺序，
        # 语法连接本身另记machine证据。
        # 合成新句不会获得human_reviewed。
        target = (values[0] if plan.relation == 'single' else '；'.join(values)
                  if plan.relation == 'and' else '先' + '，再'.join(values))
        evidence = tuple(e for p in plan.operands for e in p.evidence)
        evidence += (PhraseEvidence(source, target, None, 'machine',
                                    'qualified_full_coverage_grammar', 'qualified-proposals.1'),)
        return PhraseEnhancement(target, evidence, sum(e.origin == 'v1' for e in evidence), 0)

    # 局部constraints只做证据检索，
    # 最长完整短语支配内嵌短词。
    # 防止某整句内部的词义，
    # 又被独立短名词强加另一译法。
    # 不允许它替代plan的全覆盖判断。
    def constraints(self, source: str) -> tuple[ScopedConstraint, ...]:
        """Local evidence only; these hits never certify the remaining sentence."""
        if not isinstance(source, str) or len(source) > 2000 or self._held(source):
            return ()
        matches = []
        for pattern, phrase in self._patterns:
            for m in pattern.finditer(source):
                matches.append(ScopedConstraint(m[0], m.start(), m.end(), phrase.targets,
                                                phrase.entry_id, phrase.origin))
        # A maximal complete phrase dominates nested words, preventing unrelated
        # noun senses from imposing additional constraints inside that phrase.
        result = []
        for item in sorted(matches, key=lambda p: (-(p.end - p.start), p.start)):
            if not any(item.start < other.end and item.end > other.start for other in result):
                result.append(item)
        return tuple(sorted(result, key=lambda p: p.start))

    # 完整已知表达按声明targets比较，
    # 类型指令则调用原AST角色验证。
    # 两者都不使用任意中文相似度，
    # 未声明的流畅译文仍需专业复核。
    def _matches(self, operand, value):
        if any(_target(value) == _target(t) for t in operand.targets):
            return True
        return operand.instruction and not validate_instruction(operand.source, value, self.glossary)

    def assess(self, source: str, translated: str) -> ProposalAssessment:
        plan = self.plan(source)
        if plan is None:
            return ProposalAssessment(False, ())
        # 单叶子与多叶子的验证范围分开，
        # THEN目标只接受明确顺序连接。
        # AND可用分号或有限并列连词，
        # 但各段仍必须逐项完全匹配。
        # 目标分隔超过32个保守未确认。
        text = _target(translated)
        if plan.relation == 'single':
            valid = self._matches(plan.operands[0], text)
        else:
            if plan.relation == 'then':
                text = text.removeprefix('先')
                pattern = r'[,，]?(?:再|然后|随后|接着)'
            else:
                pattern = r'(?:[;；]|[,，]?(?:并且|并|和|及|且))'
            cuts = list(re.finditer(pattern, text))
            if len(cuts) > 32:
                return ProposalAssessment(True, ('qualified_target_scope_unconfirmed',))

            @lru_cache(None)
            # 递归按原操作数顺序消费目标全文，
            # 最后一项必须吃到目标末尾。
            # 不能把中间未知中文留在连接符间，
            # 也不能只找到几个相同词就放行。
            # 失败只标target_scope_unconfirmed，
            # 不捏造一个已证实专业错误的理由。
            def consume(index, start):
                operand = plan.operands[index]
                if index == len(plan.operands) - 1:
                    return self._matches(operand, text[start:])
                return any(m.start() > start and self._matches(operand, text[start:m.start()])
                           and consume(index + 1, m.end()) for m in cuts if m.start() >= start)

            valid = consume(0, 0)
        return ProposalAssessment(True, () if valid else ('qualified_target_scope_unconfirmed',))

    def validate(self, source: str, translated: str) -> tuple[str, ...]:
        """An empty result is meaningful only when assess(...).supported is true."""
        return self.assess(source, translated).warnings
