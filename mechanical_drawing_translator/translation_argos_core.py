# mechanical_drawing_translator/translation_argos_core.py
"""Compile qualified source evidence for bounded, fully consumed compositions.

This component cannot translate an unresolved fragment or approve a new answer.
It never calls a backend. Existing source entries and their accepted targets are
the sole technical vocabulary; new enumeration/join/equivalence syntax is machine
evidence. The service still applies its existing engineering and provenance gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata

from .translation_glossary import Glossary
from .translation_phrase_memory import PhraseEnhancement, PhraseEvidence, PhraseMemory


# Core是已知源片段的组合器，
# 不是CT2/Argos模型后端。
# 不调用模型、OCR或参考答案库，
# 未知叶子不能被它补写成中文。
# 英文和中文规范键分别用于索引，
# 不能据目标相似反向制造源证据。
def _key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).upper().split())


def _target_key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).rstrip("。.;；")


def _terminal_key(text: str) -> str:
    # Only terminal ordinary sentence punctuation can be relaxed. Operators,
    # direction marks, hashes, decimals and internal dots remain significant.
    return _key(text).rstrip(".;").rstrip()


# 枚举序号限定正整数与明确句点，
# 源目标都含相同编号才可剥取主体。
# 数字开头的尺寸不自动当列表号，
# 参考井号仍属于正文而非可删装饰。
# 续接、附属范围和开放尾词单列模式，
# 防止换行把条件语义切成独立句。
_SOURCE_LIST = re.compile(r"^([1-9]\d{0,2})\.\s+([A-Za-z#].*)$", re.S)
_TARGET_LIST = re.compile(r"^([1-9]\d{0,2})[.、]\s*([^\d.].*)$", re.S)
_CONTINUATION = re.compile(
    r"^(?:AND|OR|THEN|NOT|NO|DO NOT|IF|UNLESS|EXCEPT|AFTER|BEFORE|UNTIL|WHEN|WITHOUT|ONLY)\b", re.I,
)
_ATTACHED_SCOPE = re.compile(
    r"^(?:AND|OR|THEN|IF|UNLESS|EXCEPT|AFTER|BEFORE|UNTIL|WHEN|WITHOUT|ONLY|TO|OF|FOR|WITH|ON|IN|AS)\b", re.I,
)
_OPEN_END = re.compile(
    r"\b(?:TO|OF|FOR|BEFORE|AFTER|UNTIL|AND|OR|UNLESS|IF|WHEN|WITH|WITHOUT|EXCEPT|ON|IN|AS)$", re.I,
)
_GEOMETRY_NOUNS = frozenset({
    "SURFACE", "SURFACES", "FORM SURFACE", "FORM SURFACES", "TOP SURFACE",
    "BOTTOM SURFACE", "CHAMFER", "CHAMFERS", "RADIUS", "RADII",
})
_ACTIONS = frozenset({"GRIND", "POLISH"})
_CONFLICT = object()


@dataclass(frozen=True)
# 记录保存首选译文及明确同义集合，
# entry_id、origin与knowledge_id分开。
# kind决定是否可作术语原子，
# 完整exact不会自动拆成新技术词汇。
# _Piece是已经有证据的可组合叶子，
# 其目标不是最终人工批准结果。
class _Record:
    source: str
    target: str
    accepted: tuple[str, ...]
    entry_id: str
    origin: str
    knowledge_id: str | None
    kind: str


@dataclass(frozen=True)
class _Piece:
    target: str
    evidence: tuple[PhraseEvidence, ...]


class ArgosCoreComposer:
    """A reusable index built once from the supplied qualified glossary view."""

    def __init__(self, glossary: Glossary, phrase_memory: PhraseMemory) -> None:
        # 完整索引与枚举主体索引分开，
        # 末尾普通句读另有受控兼容索引。
        # _terms仅收term类型，
        # 没有先验技术词表可绕开传入资源。
        # runtime策略挂起在建索引时先执行，
        # 不能把已禁止条目埋入新组合。
        self._full: dict[str, list[_Record]] = {}
        self._full_terminal: dict[str, list[_Record]] = {}
        self._bodies: dict[str, list[_Record]] = {}
        self._body_terminal: dict[str, list[_Record]] = {}
        self._terms: dict[str, list[_Record]] = {}
        runtime = getattr(glossary, "runtime", None)
        self._policy_hold = getattr(runtime, "policy_hold", None)
        for phrase in glossary.reviewed_phrases():
            if not phrase.targets or any(not t.strip() for t in phrase.targets):
                continue
            if self._held(phrase.source):
                continue
            target, origin, knowledge_id = phrase.targets[0], phrase.origin, None
            # A reviewed D7 override keeps its original ID but deliberately has
            # a different target. Retain that already-scoped runtime decision;
            # never invoke enhance(), which can request model translations.
            # 复用当前已有D7范围决策，
            # 仅同entry_id、满分且非近邻可覆盖。
            # 不调用enhance避免暗中模型推理，
            # 不同词条不能借首选中文互换授权。
            # V2知识ID只从真实原引用取回，
            # 保留已迁移资源的审计链。
            resolution = phrase_memory.resolve_clause(phrase.source)
            if (resolution is not None and resolution.entry_id == phrase.entry_id
                    and resolution.score == 1.0 and resolution.match_kind != "near_clause"):
                target, origin, knowledge_id = (
                    resolution.translated_text, resolution.origin, resolution.knowledge_id,
                )
            if knowledge_id is None and runtime is not None:
                knowledge = runtime.knowledge_for_ref("v1_raw:" + phrase.entry_id)
                if knowledge is not None:
                    knowledge_id = knowledge["id"]
            accepted = phrase.targets if target == phrase.targets[0] else (target,)
            record = _Record(phrase.source, target, accepted, phrase.entry_id,
                             origin, knowledge_id, phrase.kind)
            self._add(self._full, _key(phrase.source), record)
            self._add(self._full_terminal, _terminal_key(phrase.source), record)
            if phrase.kind == "term":
                self._add(self._terms, _key(phrase.source), record)
            # 源与目标明确写同一整数序号，
            # 才能把完整句视作编号主体证据。
            # 只有源带号不授予删除编号的能力，
            # 目标编号不同时同样不建主体索引。
            # 主体仍绑定原完整条目的来源与ID。
            source_label = _SOURCE_LIST.fullmatch(phrase.source.strip())
            target_label = _TARGET_LIST.fullmatch(target.strip())
            # Both languages must explicitly carry the same integer list
            # label. A source-only leading number never licenses deletion.
            if source_label and target_label and source_label[1] == target_label[1]:
                body = _Record(source_label[2], target_label[2], (target_label[2],),
                               record.entry_id, record.origin, record.knowledge_id, record.kind)
                self._add(self._bodies, _key(body.source), body)
                self._add(self._body_terminal, _terminal_key(body.source), body)
        # 动作和几何对象只从term索引产生，
        # 普通exact不能当作任意词义库。
        # 同一术语声明的多个同义可接受，
        # 其它记录必须完全包含于该同义集合。
        # 不同记录的含义不能被自动取并集，
        # 避免静默扩大专业等价范围。
        self._atoms: dict[str, _Record] = {}
        for source in _ACTIONS | _GEOMETRY_NOUNS:
            terms = self._terms.get(source, ())
            if not terms:
                continue
            # A single term can explicitly accept multiple Chinese synonyms.
            # Different entries are not equivalent merely because a reviewer
            # might consider them so; every reading must be declared here.
            preferred = terms[0]
            accepted_keys = {_target_key(t) for t in preferred.accepted}
            all_readings = self._full.get(source, ())
            if all({_target_key(t) for t in row.accepted} <= accepted_keys for row in all_readings):
                self._atoms[source] = preferred

    @staticmethod
    def _add(index, key, record):
        index.setdefault(key, []).append(record)

    # 策略检查覆盖完整源与编号主体，
    # 编号包装不能让危险句重新准入。
    # 无runtime时保持原无策略钩子的路径，
    # 有挂起时返回专门冲突哨兵。
    def _held(self, source: str) -> bool:
        if self._policy_hold is None:
            return False
        if self._policy_hold(source):
            return True
        label = _SOURCE_LIST.fullmatch(source.strip())
        return bool(label and self._policy_hold(label[2]))

    @staticmethod
    # None表示没有可用记录，
    # _CONFLICT表示存在不可裁决含义。
    # 两者在上层不能混用，
    # 冲突必须阻止继续选择其它近似来源。
    # 只有所有读法属于首条声明同义集合，
    # 才可安全选择其确定首选译文。
    def _choose(rows):
        if not rows:
            return None
        preferred = rows[0]
        accepted = {_target_key(t) for t in preferred.accepted}
        if not all({_target_key(t) for t in row.accepted} <= accepted for row in rows):
            return _CONFLICT
        return preferred

    @staticmethod
    def _piece(source: str, record: _Record, *, label: str = "", bridge: bool = False) -> _Piece:
        target = (label + ". " if label else "") + record.target
        evidence = [PhraseEvidence(source, target, record.entry_id, record.origin,
                                  "argos_core_enumeration_bridge" if bridge else "argos_core_complete_source",
                                  record.knowledge_id)]
        return _Piece(target, tuple(evidence))

    def _known(self, source: str):
        key = _key(source)
        # 查找依次保留原完整表达优先，
        # 再尝试合法枚举主体与末句读兼容。
        # 任一已知key发生冲突就立即拒绝，
        # 不靠较宽索引掩盖原始歧义。
        # 新编号只在主体已有同号证据时桥接。
        for index, lookup, bridge in (
            (self._full, key, False), (self._bodies, key, True),
            (self._full_terminal, _terminal_key(source), False),
            (self._body_terminal, _terminal_key(source), True),
        ):
            if lookup not in index:
                continue
            record = self._choose(index[lookup])
            if record is _CONFLICT:
                return _CONFLICT
            return self._piece(source, record, bridge=bridge)
        label = _SOURCE_LIST.fullmatch(source.strip())
        if label:
            body = label[2]
            for index, lookup in ((self._bodies, _key(body)),
                                  (self._body_terminal, _terminal_key(body))):
                if lookup not in index:
                    continue
                record = self._choose(index[lookup])
                if record is _CONFLICT:
                    return _CONFLICT
                return self._piece(source, record, label=label[1], bridge=True)
        return None

    # 原子命令只支持GRIND/POLISH，
    # 对象限定明确几何名词全集。
    # 动作与对象都要已建合资格术语原子，
    # 不会凭词面猜新的对象中文。
    # 否定、全称和条件不在此窄语法。
    def _instruction(self, source: str):
        match = re.fullmatch(r"(GRIND|POLISH) (.+)", _key(source).removesuffix("."))
        if not match or match[2] not in _GEOMETRY_NOUNS:
            return None
        action, noun = self._atoms.get(match[1]), self._atoms.get(match[2])
        return (action, noun) if action is not None and noun is not None else None

    def _leaf(self, source: str):
        if self._held(source):
            return _CONFLICT
        known = self._known(source)
        if known is not None:
            return known
        instruction = self._instruction(source)
        if instruction:
            action, noun = instruction
            evidence = tuple(PhraseEvidence(p.source, p.target, p.entry_id, p.origin,
                                           "argos_core_reviewed_term", p.knowledge_id)
                             for p in (action, noun))
            return _Piece(action.target + noun.target, evidence)
        # Explicitly code-shaped opaque identity, never a material translation.
        # Plain words (STEEL) and arbitrary noun phrases are not identifiers.
        # OR EQUIVALENT左侧必须是代码形态，
        # 至少有点/连字符或字母数字混合。
        # 普通STEEL等材料名不被当作代码，
        # 不据此推导任何新的材料等价。
        # 透明代码保留字面再接有限等同语法。
        match = re.fullmatch(r"([A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)+|[A-Z]+\d[A-Z0-9]*) OR EQUIVALENT\.?", source.strip())
        if match:
            identifier = match[1]
            target = identifier + "或等同"
            return _Piece(target, (PhraseEvidence(identifier, identifier, None, "machine",
                                                  "argos_core_opaque_identity", "argoscore.1"),))
        return None

    @staticmethod
    # 短词尾句点可能属于工程缩写，
    # 未证实完整多词子句时不开放切分。
    # 首字母、带点缩写及明确缩写，
    # 即使前面有内容仍保持保护。
    # 句界证据来自完整源消费，
    # 不是为某份图纸硬编码答案例外。
    def _abbreviation_before(text: str, *, complete_clause: bool = False) -> bool:
        match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", text.rstrip())
        if match is None:
            return False
        word = match[1]
        # A short noun can end a genuine instruction, but a known isolated
        # abbreviation cannot authorize separating an unknown field group.
        # Only a completely consumed multiword clause licenses the short-word
        # exception. Initials and dotted/explicit abbreviations stay protected.
        return ("." in word or len(word) == 1 or word.lower() in {"incl", "etc", "vs"}
                or (len(word) <= 3 and not complete_clause))

    def _boundaries(self, source: str):
        # 边界包含起点和整串终点，
        # 换行属于软边界，句点分号可硬切。
        # 小数前数字阻止把尺寸切成两句，
        # 缩写后的换行也不自动成为语义句界。
        # 所有片段最终仍须有完整已知译法。
        cuts = {0: False, len(source): False}
        for separator in re.finditer(r"\r?\n+|;\s*|\.\s+", source):
            value = separator[0]
            before = source[:separator.end()].rstrip()
            complete_clause = False
            for start in cuts:
                if start >= separator.start():
                    continue
                candidate = source[start:separator.end()].strip()
                # This is source-consumption evidence, not an English word
                # exception list. DP. alone therefore cannot license DP. MIN.
                if len(_key(candidate).split()) < 2:
                    continue
                known = self._known(candidate)
                if ((known is not None and known is not _CONFLICT)
                        or self._instruction(candidate) is not None):
                    complete_clause = True
                    break
            if value.startswith("."):
                if (source[separator.start() - 1:separator.start()].isdigit()
                        or self._abbreviation_before(before, complete_clause=complete_clause)):
                    continue
            elif (value.startswith(("\n", "\r"))
                    and self._abbreviation_before(source[:separator.start()], complete_clause=complete_clause)):
                continue
            cuts[separator.end()] = value.startswith((".", ";"))
        return sorted(cuts), cuts

    def resolve(self, source: str) -> PhraseEnhancement | None:
        if not isinstance(source, str) or not source.strip() or len(source) > 2000 or self._held(source):
            return None
        # 先尝试整串已知叶子，
        # 避免拆分破坏原来完整审核范围。
        # 源最长两千字符限制单项成本，
        # 边界超过34个拒绝继续搜索。
        # 这些上限是计算预算而非质量阈值。
        complete = self._leaf(source)
        if complete is _CONFLICT:
            return None
        if complete is not None:
            return self._result(source, (complete,))
        positions, hard = self._boundaries(source)
        if len(positions) > 34:
            return None

        @lru_cache(None)
        # 递归按原始字符位置穷举全覆盖路径，
        # 记忆化只在本次resolve调用有效。
        # 软换行后的续接词不能单独起段，
        # 附属范围词与开放尾部也禁止脱离。
        # 每个子串都须成功成为有证据叶子，
        # 无翻译残片不会被丢弃。
        def paths(start):
            if start == len(source):
                return ((),)
            options = []
            for end in positions:
                if end <= start:
                    continue
                fragment = source[start:end].strip()
                if not fragment:
                    continue
                # A soft wrap cannot turn a local condition into an independent
                # paragraph statement. A complete existing source is tried first.
                if start and not hard[start] and _CONTINUATION.match(fragment):
                    continue
                if start and _ATTACHED_SCOPE.match(fragment):
                    continue
                if end < len(source) and (fragment.endswith(":")
                        or _OPEN_END.search(fragment.rstrip(".;"))):
                    continue
                leaf = self._leaf(fragment)
                if leaf is None or leaf is _CONFLICT:
                    continue
                for tail in paths(end):
                    if len(tail) < 16:
                        options.append((leaf, *tail))
                        if len(options) > 16:
                            return tuple(options)
            return tuple(options)

        # 每条路径最多16叶子，
        # 候选路径超过16即判范围不稳定。
        # 所有可行路径的中文分段序列须一致，
        # 相同词集合但段落意思不同仍拒绝。
        # 最后仅选唯一含义的一个证据路径，
        # 不是按最短或首条路径猜答案。
        options = paths(0)
        if not options or len(options) > 16:
            return None
        targets = {tuple(_target_key(p.target) for p in path) for path in options}
        if len(targets) != 1:
            return None
        return self._result(source, options[0])

    @staticmethod
    # 整句用可见中文边界连接已知叶子，
    # 各叶子的原始证据全部保留。
    # 另外追加argos_core_grammar机器证据，
    # 说明枚举与连接是新组合推导。
    # 分来源计数只描述证据构成，
    # 不能解释为新增专家批准数量。
    def _result(source: str, pieces: tuple[_Piece, ...]) -> PhraseEnhancement:
        target = "；".join(p.target.rstrip("。.;；") for p in pieces)
        evidence = tuple(e for p in pieces for e in p.evidence)
        evidence += (PhraseEvidence(source, target, None, "machine", "argos_core_grammar", "argoscore.1"),)
        return PhraseEnhancement(target, evidence, sum(e.origin == "v1" for e in evidence), 0,
                                 sum(e.origin == "d7c" for e in evidence),
                                 sum(e.origin == "d7d" for e in evidence),
                                 sum(e.origin == "d7e" for e in evidence))

    def validate(self, source: str, target: str) -> tuple[str, ...]:
        """Role-check only the fully supported atomic imperative source grammar.

        Empty warnings for an unsupported source express no coverage. Existing
        full reviewed clauses, conditions and sequences remain the service's
        responsibility; they are never certified by Chinese string equality.
        """
        # validate只覆盖窄原子动宾命令，
        # 其它完整句由服务原门禁负责。
        # 先尝试源自身允许的中文实现，
        # 避免单复数共享译法被别项抢判。
        # 再识别是否变成另一动作对象，
        # 未知中文返回target_unconfirmed。
        expected = self._instruction(source)
        if expected is None:
            return ()
        text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", target)).rstrip("。.")
        expected_keys = tuple(_key(p.source) for p in expected)
        # Check the source's own declared realizations first: plural/singular
        # terms may share a valid Chinese form. Set iteration cannot adjudicate
        # such a form as the wrong object before consulting its explicit grant.
        if self._target_instruction(text, *expected):
            return ()
        for action_key in _ACTIONS:
            action = self._atoms.get(action_key)
            if action is None:
                continue
            for noun_key in _GEOMETRY_NOUNS:
                noun = self._atoms.get(noun_key)
                if noun is None:
                    continue
                if self._target_instruction(text, action, noun):
                    return () if (action_key, noun_key) == expected_keys else ("argos_core_action_object_conflict",)
        return ("argos_core_instruction_target_unconfirmed",)

    @staticmethod
    def _target_instruction(text: str, action: _Record, noun: _Record) -> bool:
        # Full parsing of a small explicit surface grammar, not keyword
        # presence. Negation, ALL, conditions and extra nouns stay unconsumed.
        # 中文语法必须完整消费目标，
        # 只允许明确的义务词与有限把字语序。
        # 额外否定、全称、条件或其它对象，
        # 不会因为核心专业词出现就通过。
        # 空警告仍只证明该窄语法的关系。
        for verb in action.accepted:
            for obj in noun.accepted:
                action_re, object_re = re.escape(_target_key(verb)), re.escape(_target_key(obj))
                patterns = (
                    r"(?:应|须|需|必须)?" + action_re + object_re,
                    r"(?:将|把|对)?" + object_re + r"(?:应|须|需|必须)?(?:进行)?" + action_re,
                    r"(?:应|须|需|必须)(?:将|把|对)" + object_re + r"(?:进行)?" + action_re,
                    r"(?:应|须|需|必须)?进行" + object_re + action_re,
                )
                if any(re.fullmatch(pattern, text) for pattern in patterns):
                    return True
        return False
