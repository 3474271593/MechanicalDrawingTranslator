# mechanical_drawing_translator/translation_phrase_memory.py
# 记忆复用以当前已审核词表为输入，
# 不会从评测参考反推新答案。
# 分词只为规则匹配与安全近邻比较，
# 真正模型分词仍由SentencePiece承担。
"""Conservative reuse of reviewed D6/V1 phrases inside non-exact sentences."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from .engineering_text import extract_immutable_literals, validate_immutable_literals
from .translation_domain_knowledge import DomainKnowledge
from .translation_glossary import Glossary, ReviewedPhrase
from .translation_generalization import GeneralizationKnowledge
from .translation_professional import ProfessionalKnowledge


# 否定、方向、顺序、数量界限是关键字，
# 不可借相似词替换或省略。
# 仅BE被列为可选功能词，
# 不能随意删除介词和条件词。
# 星号、度数及井号也作为匹配token，
# 不能为提高匹配率静默抹去。
_WORD_PATTERN = re.compile(r"[A-Z]+(?:'[A-Z]+)?|\d+(?:\.\d+)?|[°′″*#]")
_CRITICAL_WORDS = frozenset({
    "NO", "NOT", "ONLY", "ALL", "TOP", "BOTTOM", "FAR", "NEAR",
    "BEFORE", "AFTER", "ABOVE", "BELOW", "MIN", "MAX", "MUST",
})
_OPTIONAL_FUNCTION_WORDS = frozenset({"BE"})
_TAIL_WORDS = frozenset({"TO", "MM", "CM", "UM", "RA", "RMAX", "HRC"})


@dataclass(frozen=True)
class PhraseResolution:
    translated_text: str
    entry_id: str
    origin: str
    match_kind: str
    score: float
    knowledge_id: str | None = None


@dataclass(frozen=True)
# 证据保留原段、译段、词条与来源，
# knowledge_id可关联独立知识授权。
# 多个证据可组成整句机器结果，
# 每个片段的来源边界仍须保留。
class PhraseEvidence:
    source_segment: str
    translated_segment: str
    entry_id: str | None
    origin: str
    match_kind: str
    knowledge_id: str | None = None


@dataclass(frozen=True)
class PhraseEnhancement:
    translated_text: str
    evidence: tuple[PhraseEvidence, ...]
    v1_segment_count: int
    model_segment_count: int
    d7c_segment_count: int = 0
    d7d_segment_count: int = 0
    d7e_segment_count: int = 0


@dataclass(frozen=True)
class _Rule:
    entry_id: str
    source: str
    target: str
    origin: str
    tokens: tuple[str, ...]
    match_kind: str | None = None
    knowledge_id: str | None = None


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).upper()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_WORD_PATTERN.findall(_normalize(value)))


# 近邻桶先忽略允许的BE，
# 按内容token数与首词前缀索引。
# 这只减少候选扫描量，
# 真正接受仍需动态规划和字面量守卫。
# 同桶不代表两个英文可以互换。
def _match_key(tokens: tuple[str, ...]) -> tuple[int, str]:
    content = tuple(token for token in tokens if token not in _OPTIONAL_FUNCTION_WORDS)
    if not content:
        return 0, ""
    first = content[0]
    stem = first[:5] if first.isalpha() and len(first) >= 5 else first
    return len(content), stem


def _label_key(value: str) -> str:
    return " ".join(_normalize(value).split())


# 完全相同token零成本，
# 任何关键字变化都不可对齐。
# 数字和符号不允许形似替换，
# 较长纯字母词才允许有限共同前缀。
# 词尾变化最多三字符且成本0.2，
# 不是任意编辑距离纠错器。
def _word_cost(left: str, right: str) -> float | None:
    if left == right:
        return 0.0
    if left in _CRITICAL_WORDS or right in _CRITICAL_WORDS:
        return None
    if not left.isalpha() or not right.isalpha():
        return None
    if len(left) >= 5 and len(right) >= 5:
        common = 0
        for a, b in zip(left, right):
            if a != b:
                break
            common += 1
        if common >= 5 and len(left) - common <= 3 and len(right) - common <= 3:
            return 0.2
    return None


def _alignment_cost(left: tuple[str, ...], right: tuple[str, ...]) -> float | None:
    # 二维动态规划消费左右两串token，
    # 只有BE可插入或删除并付0.15成本。
    # 其它步骤必须有可接受替换成本，
    # 不可达路径保持正无穷。
    # 总成本超过0.5直接拒绝，
    # 不会只凭局部高相似片段命中。
    rows, columns = len(left) + 1, len(right) + 1
    infinity = float("inf")
    costs = [[infinity] * columns for _ in range(rows)]
    costs[0][0] = 0.0
    for i in range(rows):
        for j in range(columns):
            current = costs[i][j]
            if current == infinity:
                continue
            if i < len(left) and left[i] in _OPTIONAL_FUNCTION_WORDS:
                costs[i + 1][j] = min(costs[i + 1][j], current + 0.15)
            if j < len(right) and right[j] in _OPTIONAL_FUNCTION_WORDS:
                costs[i][j + 1] = min(costs[i][j + 1], current + 0.15)
            if i < len(left) and j < len(right):
                word_cost = _word_cost(left[i], right[j])
                if word_cost is not None:
                    costs[i + 1][j + 1] = min(
                        costs[i + 1][j + 1], current + word_cost,
                    )
    result = costs[-1][-1]
    return None if result == infinity or result > 0.5 else result


def _split_segments(source: str) -> tuple[tuple[str, str], ...]:
    # 按句点、分号与破折号拆片段，
    # 紧邻数字的分隔符保持在字面里。
    # 原分隔符与片段成对保留，
    # 重组时再映射成中文句读。
    # 不把小数点误认为句界。
    pieces = re.split(r"(?<!\d)([.;—]+)(?!\d)", source)
    result = []
    for index in range(0, len(pieces), 2):
        segment = pieces[index].strip()
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        if segment:
            result.append((segment, separator))
    return tuple(result)


def _translated_separator(separator: str) -> str:
    if "—" in separator or ";" in separator:
        return "；"
    if "." in separator:
        return "。"
    return ""


class PhraseMemory:
    def __init__(
        self, rules: tuple[_Rule, ...], *, domain_knowledge: DomainKnowledge | None = None,
        generalization_knowledge: GeneralizationKnowledge | None = None,
        professional_knowledge: ProfessionalKnowledge | None = None,
    ) -> None:
        self._rules = rules
        self._domain_knowledge = domain_knowledge
        self._generalization_knowledge = generalization_knowledge
        self._professional_knowledge = professional_knowledge
        # 精确、近邻和短标签各建独立索引，
        # 相同key保留全部候选而非覆盖。
        # 短标签仅从最多四token的V1产生，
        # 避免开放任意长句冒充字段头。
        # 索引构建不调用模型。
        exact: dict[tuple[str, ...], list[_Rule]] = {}
        near: dict[tuple[int, str], list[_Rule]] = {}
        labels: dict[str, list[_Rule]] = {}
        for rule in rules:
            exact.setdefault(rule.tokens, []).append(rule)
            near.setdefault(_match_key(rule.tokens), []).append(rule)
            if rule.origin == "v1" and len(rule.tokens) <= 4:
                labels.setdefault(_label_key(rule.source), []).append(rule)
        self._exact_index = {key: tuple(value) for key, value in exact.items()}
        self._near_index = {key: tuple(value) for key, value in near.items()}
        self._label_index = {key: tuple(value) for key, value in labels.items()}

    @classmethod
    def from_glossary(
        cls, glossary: Glossary, *, domain_knowledge: DomainKnowledge | None = None,
        generalization_knowledge: GeneralizationKnowledge | None = None,
        professional_knowledge: ProfessionalKnowledge | None = None,
    ) -> "PhraseMemory":
        rules = []
        # 只接收单一目标的已审核片段，
        # 多目标词义不凭首项任意选译。
        # D7C复用覆盖按原source_entry_id绑定，
        # 来源改记d7c并保留其独立identifier。
        # 旧V1文本与资源本体不被改写。
        for phrase in glossary.reviewed_phrases():
            if len(phrase.targets) != 1:
                continue
            override = (
                domain_knowledge.reuse_overrides.get(phrase.entry_id)
                if domain_knowledge is not None else None
            )
            rules.append(_Rule(
                phrase.entry_id,
                phrase.source,
                override.target if override is not None else phrase.targets[0],
                "d7c" if override is not None else phrase.origin,
                _tokens(phrase.source),
                "reuse_override" if override is not None else None,
                override.identifier if override is not None else None,
            ))
        if domain_knowledge is not None:
            for phrase in domain_knowledge.reviewed_phrases:
                rules.append(_Rule(
                    phrase.entry_id, phrase.source, phrase.targets[0], phrase.origin,
                    _tokens(phrase.source),
                ))
        return cls(
            tuple(rules), domain_knowledge=domain_knowledge,
            generalization_knowledge=generalization_knowledge,
            professional_knowledge=professional_knowledge,
        )

    def resolve_clause(self, source: str) -> PhraseResolution | None:
        stripped = source.strip()
        if not stripped:
            return None
        # 整句D7D模板优先，
        # 再检查D7E专业组合与D7C模板。
        # 这些解析需要完整原串被消费，
        # 不是只替换句内几个专业词。
        # 只有均未解决才搜索记忆近邻。
        if self._generalization_knowledge is not None:
            generalized = self._generalization_knowledge.resolve_template(stripped)
            if generalized is not None:
                return PhraseResolution(
                    generalized.translated_text, generalized.entry_id, "d7d",
                    generalized.match_kind, 1.0,
                )
        if self._professional_knowledge is not None:
            professional = self._professional_knowledge.resolve(stripped)
            if professional is not None:
                return PhraseResolution(
                    professional.translated_text, professional.entry_id, "d7e",
                    professional.match_kind, 1.0,
                    ",".join(professional.term_ids),
                )
        if self._domain_knowledge is not None:
            template = self._domain_knowledge.resolve_template(stripped)
            if template is not None:
                return PhraseResolution(
                    template.translated_text, template.entry_id, template.origin,
                    template.match_kind, 1.0,
                )
        # 精确桶存在时不混入其它近邻，
        # 匹配后还按规则原文保护工程字面。
        # score表示允许的词形代价，
        # 不代表专业翻译置信度或准确率。
        # 同最高分但中文不同必须拒绝，
        # 稳定ID排序不负责裁决不同含义。
        source_tokens = _tokens(stripped)
        candidates: list[PhraseResolution] = []
        exact_rules = self._exact_index.get(source_tokens, ())
        candidate_rules = exact_rules or self._near_index.get(_match_key(source_tokens), ())
        for rule in candidate_rules:
            cost = _alignment_cost(source_tokens, rule.tokens)
            if cost is None:
                continue
            literal_check = validate_immutable_literals(rule.source, stripped)
            if not literal_check.is_valid:
                continue
            exact = source_tokens == rule.tokens
            candidates.append(PhraseResolution(
                rule.target, rule.entry_id, rule.origin,
                rule.match_kind or ("exact_clause" if exact else "near_clause"),
                1.0 - cost / max(len(source_tokens), len(rule.tokens), 1),
                rule.knowledge_id,
            ))
        if candidates:
            candidates.sort(key=lambda item: (-item.score, item.entry_id))
            best = candidates[0]
            tied = [item for item in candidates if abs(item.score - best.score) < 1e-9]
            if len({item.translated_text for item in tied}) == 1:
                return best
            return None
        return self._resolve_structured_tail(stripped)

    # 字段尾部必须明确以冒号分隔，
    # 标签来自短V1标签索引。
    # 尾部至少含一个工程字面量，
    # 且英文词仅能是受控量纲与标记。
    # TO可转为“至”但数值仍原样保护，
    # 多候选字段不选第一个结果。
    def _resolve_structured_tail(self, source: str) -> PhraseResolution | None:
        # 结构尾只接收唯一标签候选，
        # 同标签的不同V1含义不能任选。
        # 尾部数字可以包含范围与公差，
        # 但不允许带未审的自由英文句。
        # 保留原尾字符串可追查每个符号，
        # 中文冒号仅改变字段呈现形式。
        candidates = []
        match = re.match(r"^(.+?)\s*:\s*(.+)$", source)
        if match is None:
            return None
        label, tail = match.group(1), match.group(2).strip()
        for rule in self._label_index.get(_label_key(label), ()):
            if not extract_immutable_literals(tail):
                continue
            words = {word for word in re.findall(r"[A-Za-z]+", tail.upper())}
            if not words.issubset(_TAIL_WORDS):
                continue
            translated_tail = re.sub(r"\s+TO\s+", " 至 ", tail, flags=re.I)
            translated = f"{rule.target}：{translated_tail}"
            if not validate_immutable_literals(source, translated).is_valid:
                continue
            candidates.append(PhraseResolution(
                translated, rule.entry_id, rule.origin, "structured_tail", 1.0,
            ))
        if len(candidates) == 1:
            return candidates[0]
        return None

    def enhance(
        self,
        source: str,
        translate_unresolved: Callable[[str], str | None],
    ) -> PhraseEnhancement | None:
        # 必须至少命中一个专业来源片段，
        # 不能把纯模型拆句称作记忆增强。
        # 所有子句先独立解析候选，
        # 随后按原顺序组装输出与证据。
        # 专业命中不为相邻未知句提供批准。
        segments = _split_segments(source)
        resolutions = tuple(self.resolve_clause(segment) for segment, _ in segments)
        if not any(
            resolution is not None and resolution.origin in {"v1", "d7c", "d7d", "d7e"}
            for resolution in resolutions
        ):
            return None
        evidence = []
        output = []
        v1_count = 0
        d7c_count = 0
        d7d_count = 0
        d7e_count = 0
        model_count = 0
        for (segment, separator), resolution in zip(segments, resolutions):
            if resolution is not None:
                translated = resolution.translated_text
                evidence.append(PhraseEvidence(
                    segment, translated, resolution.entry_id,
                    resolution.origin, resolution.match_kind, resolution.knowledge_id,
                ))
                if resolution.origin == "v1":
                    v1_count += 1
                elif resolution.origin == "d7c":
                    d7c_count += 1
                elif resolution.origin == "d7d":
                    d7d_count += 1
                elif resolution.origin == "d7e":
                    d7e_count += 1
            # 未知片段只经调用者提供的受控回调，
            # 返回None即放弃整个组合。
            # 不会丢掉失败片段只输出已知中文，
            # 避免表面覆盖掩盖条件或否定缺失。
            # 模型片段计数明确保留待审核来源。
            else:
                translated = translate_unresolved(segment)
                if translated is None:
                    return None
                evidence.append(PhraseEvidence(
                    segment, translated, None, "model", "model_segment",
                ))
                model_count += 1
            output.append(translated.strip())
            # 已有同样中文句读时避免重复追加，
            # 其它原分隔关系按映射保留。
            # 最终仍需服务对完整源句做守卫，
            # 片段各自通过不足以保证组合正确。
            translated_separator = _translated_separator(separator)
            if not (
                translated_separator
                and translated.rstrip().endswith(translated_separator)
            ):
                output.append(translated_separator)
        # 来源段数表示本次真正采用的片段，
        # 不是词库容量或人工正确实例数。
        # 同一段可能含多个术语，
        # 不能据段计数宣称术语95%达标。
        # 最终仍要求至少一个专业来源，
        # 纯模型输出不会冒名hybrid_memory。
        if v1_count == 0 and d7c_count == 0 and d7d_count == 0 and d7e_count == 0:
            return None
        return PhraseEnhancement(
            "".join(output), tuple(evidence), v1_count, model_count, d7c_count,
            d7d_count,
            d7e_count,
        )
