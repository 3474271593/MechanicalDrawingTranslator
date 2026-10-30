# mechanical_drawing_translator/mechanical_instructions.py
"""Finite mechanical instructions, with explicit machine composition evidence.

This is a bounded bilingual grammar, not a general Chinese semantic judge. Both
parsers must consume their entire input. An unrecognized Chinese realization is
unconfirmed even when it contains all expected mechanical keywords.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .translation_glossary import Glossary
from .translation_phrase_memory import PhraseEnhancement, PhraseEvidence


@dataclass(frozen=True)
# 原子概念携带可接受中文集合，
# 以及其词条或公开来源证据。
# 它不是整句人工答案，
# 多个原子组合仍属于机器推导。
class Atom:
    key: str
    source: str
    target: str
    targets: tuple[str, ...]
    evidence: tuple[PhraseEvidence, ...]


@dataclass(frozen=True)
# 一个操作独立绑定动作与对象，
# 全称、否定、义务各占一个槽位。
# subject与location承担不同角色，
# 不能靠句中出现同词视作等价。
class Operation:
    action: Atom
    object: Atom
    all_objects: bool = False
    negated: bool = False
    required: bool = False
    subject: Atom | None = None
    location: Atom | None = None


@dataclass(frozen=True)
# 指令最多保存两个操作，
# 可再附一个条件与一个事件。
# 全局条件槽不能描述任意局部范围，
# 因此复杂尾部条件必须先拒绝。
class Instruction:
    operations: tuple[Operation, ...]
    relation: str = "single"
    temporal_relation: str | None = None
    event: Atom | None = None
    condition: str | None = None


# These are atomic concepts, never complete instruction answers. V1-derived
# atoms require their original complete entry to be loaded and retain its ID.
# 派生原子必须找到原V1完整词条，
# 缺少原ID时不创建对应能力。
# 公开原子带公开文献引用，
# 语法连接则标machine证据。
# 三种来源不能互相冒充人工批准。
_DERIVED = (
    ("coat", "COAT", "涂层", ("涂层", "涂覆"), "mgt-055300b473e9d5cf"),
    ("part", "PART", "工件", ("工件", "零件"), "mgt-bda36a74c65c71ad"),
    ("thread", "THREAD", "螺纹", ("螺纹",), "mgt-39200d1e8a8dbbb6"),
    ("form_surface", "FORM SURFACE", "成型面", ("成型面", "成形面"), "mgt-76b92021dc722ce8"),
    ("punch_head", "PUNCH HEAD", "冲子头", ("冲子头", "冲头头部", "冲头头"), "mgt-df8e3ff693a1aff3"),
    ("surface", "SURFACE", "表面", ("表面", "面"), "mgt-055300b473e9d5cf"),
    ("top_surface", "TOP SURFACE", "顶面", ("顶面", "顶部表面", "上表面"), "mgt-f7d356e6191bc855"),
    ("bottom_surface", "BOTTOM SURFACE", "底面", ("底面", "底部表面", "下表面"), "mgt-54b9f40f14d502e6"),
    ("coating", "COATING", "涂层", ("涂层", "涂覆层"), "mgt-51446de343eb0aac"),
)
_PUBLIC = (
    ("burr", "BURR", "毛刺", ("毛刺",), "misumi:mp01/b0015+misumi:exportarticle/429"),
    ("deburr", "DEBURR", "去毛刺", ("去毛刺", "去除毛刺"), "misumi:mp01/b0015+misumi:exportarticle/429"),
    ("heat_treatment", "HEAT TREATMENT", "热处理", ("热处理",), "misumi:gear/faq"),
    ("drilling", "DRILLING", "钻孔", ("钻孔",), "sandvik:parting-off"),
    ("punch", "PUNCH", "冲头", ("冲头", "冲子"), "misumi:product/cid=703"),
    ("die", "DIE", "模具", ("模具", "凹模"), "misumi:product/cid=703"),
)
_STRUCTURAL = (
    ("remove", "REMOVE", "去除", ("去除", "清除")),
    ("cover", "COVER", "覆盖", ("覆盖",)),
    ("slide", "SLIDE FREELY", "自由滑动", ("自由滑动",)),
)
_OBJECT_KEYS = frozenset({"part", "thread", "form_surface", "punch_head", "surface",
                          "top_surface", "bottom_surface", "punch", "die", "burr"})
# 主动、动名词与被动形态分别登记，
# 不对任意英文词做词干猜测。
# 事件槽只接收有限工艺名称，
# 装配必须使用动作义而非组件义。
_ACTION_FORMS = {
    "GRIND": "grind", "GRINDING": "grind", "POLISH": "polish",
    "POLISHING": "polish", "COAT": "coat", "DEBURR": "deburr", "REMOVE": "remove",
}
_PASSIVE_FORMS = {"GROUND": "grind", "POLISHED": "polish", "COATED": "coat"}
_EVENT_KEYS = ("heat_treatment", "drilling", "coating", "assembly")


def _source(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).upper().split()).removesuffix(".").strip()


def _atoms(glossary: Glossary) -> dict[str, Atom]:
    # 当前传入词表决定可用原子全集，
    # 不读取额外整句答案文件。
    # 唯一D6术语才可构造基本动作，
    # 多义记录不凭加载顺序选词。
    phrases = glossary.reviewed_phrases()
    by_id = {p.entry_id: p for p in phrases}
    atoms: dict[str, Atom] = {}
    for key in ("grind", "polish", "assembly"):
        terms = [p for p in phrases if p.kind == "term" and _source(p.source) == key.upper()]
        if len(terms) == 1:
            term = terms[0]
            # Only the process reading applies to an event; 组件/总成 are
            # entity readings of the same D6 entry and cannot be events.
            targets = term.targets if key != "assembly" else ("组装",)
            evidence = PhraseEvidence(term.source, targets[0], term.entry_id,
                                     "d6", "instruction_reviewed_atom")
            atoms[key] = Atom(key, term.source, targets[0], targets, (evidence,))
    # V1派生项还核对origin为v1，
    # 同名ID不能跨来源借用。
    # component-derived/machine明确说明，
    # 原词条审核不等于新句已审核。
    for key, source, target, targets, entry_id in _DERIVED:
        phrase = by_id.get(entry_id)
        if phrase is None or phrase.origin != "v1":
            continue
        evidence = PhraseEvidence(source, target, entry_id, "component-derived/machine",
                                 "instruction_component", "finite-instruction-v1")
        atoms[key] = Atom(key, source, target, targets, (evidence,))
    for key, source, target, targets, ref in _PUBLIC:
        evidence = PhraseEvidence(source, target, None, "public-source/machine",
                                 "instruction_public_atom", ref)
        atoms[key] = Atom(key, source, target, targets, (evidence,))
    for key, source, target, targets in _STRUCTURAL:
        evidence = PhraseEvidence(source, target, None, "machine",
                                 "instruction_grammar", "finite-instruction-v1")
        atoms[key] = Atom(key, source, target, targets, (evidence,))
    return atoms


def ambiguity_warning(source: str, glossary: Glossary) -> tuple[str, ...]:
    """Explain refused ambiguous scopes without promoting local qualifiers."""
    # 歧义检查先于有限语法组合，
    # MAY NOT与否定AND范围不明。
    # 带THEN的前导禁止可能作用整链，
    # 不能只把否定贴到第一个动作。
    # 尾部条件或事件也不能擅自前移。
    text = _source(source)
    warnings = []
    if re.search(r"\bMAY NOT\b", text):
        warnings.append("instruction_ambiguous_may_not")
    if re.search(r"\bDO NOT\b.*\bAND\b", text):
        warnings.append("instruction_ambiguous_negated_and")
    # A leading prohibition can scope over the complete ordered procedure.
    # Attaching it only to the first operation is unsafe. A separate negative
    # final operation (positive THEN DO NOT...) has no following THEN and stays
    # within the finite grammar.
    if re.search(r"\b(?:DO NOT|MUST NOT|SHALL NOT)\b.+\bTHEN\b", text):
        warnings.append("instruction_ambiguous_negated_sequence_scope")
    # The AST has a global condition/event slot. A qualifier following a
    # multi-operation chain may attach only to its last operation; promoting it
    # to the entire chain changes meaning. Refuse before any model fallback.
    if re.search(r"\b(?:AND|THEN)\b.+\b(?:IF SPECIFIED|UNLESS (?:OTHERWISE )?SPECIFIED)$", text):
        warnings.append("instruction_ambiguous_trailing_condition_scope")
    if re.search(r"\b(?:AND|THEN)\b.+\b(?:BEFORE|AFTER)\b", text):
        warnings.append("instruction_ambiguous_trailing_event_scope")
    return tuple(warnings)


def _object(value: str, atoms: dict[str, Atom],
            allowed_keys: frozenset[str] = _OBJECT_KEYS) -> tuple[Atom, bool] | None:
    # THE与ALL只在对象起始消费，
    # ALL记录到该对象的量词属性。
    # 支持的复数只允许词尾S，
    # 不会扩展成通用英文形态解析。
    # allowed_keys可对特定角色进一步限域。
    value = value.removeprefix("THE ")
    all_objects = value.startswith("ALL ")
    if all_objects:
        value = value[4:].removeprefix("THE ")
    for key in allowed_keys:
        atom = atoms.get(key)
        if atom is not None and value in (atom.source, atom.source + "S"):
            return atom, all_objects
    return None


def _operation(text: str, atoms: dict[str, Atom]) -> Operation | None:
    # 先解析有显式主语的特殊关系，
    # 涂层覆盖限定涂层与螺纹角色。
    # 自由滑动限定冲头在模具内，
    # 不把介词IN忽略成一般动宾。
    # 每个特殊语法都必须整句消费。
    match = re.fullmatch(r"(.+?) (?:MUST|SHALL) (NOT )?COVER (.+)", text)
    if match:
        subject = _object(match[1], atoms, frozenset({"coating"}))
        obj = _object(match[3], atoms, frozenset({"thread"}))
        if subject is not None and not subject[1] and obj is not None:
            return Operation(atoms["cover"], obj[0], obj[1], bool(match[2]), True, subject[0])
        return None
    match = re.fullmatch(r"(.+?) (?:MUST|SHALL) SLIDE FREELY IN (.+)", text)
    if match:
        subject = _object(match[1], atoms, frozenset({"punch"}))
        location = _object(match[2], atoms, frozenset({"die"}))
        if subject and location and not subject[1] and not location[1]:
            return Operation(atoms["slide"], subject[0], required=True, location=location[0])
        return None
    # 被动MUST/SHALL仍保留义务，
    # NOT绑定到该项被动动作。
    # 有限对象未识别便放弃解析，
    # 不截去未知宾语只翻加工词。
    match = re.fullmatch(r"(?:THE )?(.+?) (?:MUST|SHALL) (NOT )?BE (GROUND|POLISHED|COATED)", text)
    if match:
        obj = _object(match[1], atoms)
        action = atoms.get(_PASSIVE_FORMS[match[3]])
        if obj is not None and action is not None:
            return Operation(action, obj[0], obj[1], bool(match[2]), True)
        return None
    match = re.fullmatch(r"(DO NOT )?(GRIND|GRINDING|POLISH|POLISHING|COAT|DEBURR|REMOVE) (.+)", text)
    if not match:
        return None
    # 普通命令也须动作与对象均有证据，
    # REMOVE特别限制为去除毛刺。
    # 不能借该一般动词自动删除螺纹，
    # 或把其它零件当作可移除对象。
    obj = _object(match[3], atoms)
    action = atoms.get(_ACTION_FORMS[match[2]])
    if obj is None or action is None:
        return None
    # REMOVE is licensed here only for burr removal, not arbitrary removal of
    # threads, parts or coatings. Other process/object pairs use known objects.
    if action.key == "remove" and obj[0].key != "burr":
        return None
    return Operation(action, obj[0], obj[1], bool(match[1]))


def parse_instruction(source: str, glossary: Glossary) -> Instruction | None:
    """Consume one/two operations and at most one condition and process event."""
    text = _source(source)
    if (not text or not re.fullmatch(r"[A-Z ,]+", text)
            or re.search(r"\bSHARP EDGES?\b", text) or ambiguity_warning(source, glossary)):
        return None
    atoms = _atoms(glossary)
    condition = None
    condition_re = r"IF SPECIFIED|UNLESS (?:OTHERWISE )?SPECIFIED"
    prefix = re.fullmatch(rf"({condition_re}), (.+)", text)
    suffix = re.fullmatch(rf"(.+?)(?:,)? ({condition_re})", text)
    if prefix:
        condition = "if_specified" if prefix[1] == "IF SPECIFIED" else "unless_specified"
        text = prefix[2]
    elif suffix:
        condition = "if_specified" if suffix[2] == "IF SPECIFIED" else "unless_specified"
        text = suffix[1]
    # 时间事件只从既有事件原子选择，
    # BEFORE和AFTER作为关系保存。
    # 未知事件无法用相近工艺顶替，
    # 也不会把剩余文字交模型补齐。
    temporal_relation = None
    event = None
    prefix = re.fullmatch(r"(BEFORE|AFTER) (.+?), (.+)", text)
    suffix = re.fullmatch(r"(.+?) (BEFORE|AFTER) (.+)", text)
    if prefix or suffix:
        event_text = prefix[2] if prefix else suffix[3]
        event = next((atoms[k] for k in _EVENT_KEYS if k in atoms and atoms[k].source == event_text), None)
        if event is None:
            return None
        temporal_relation = (prefix[1] if prefix else suffix[2]).lower()
        text = prefix[3] if prefix else suffix[1]
    # THEN表示有序，AND表示并列，
    # 两者在后续验证中不可互换。
    # 超过两个子句超出当前语法，
    # 任一子句未解析则整条不组合。
    relation = "single"
    if " THEN " in text:
        relation = "then"
        parts = text.split(" THEN ")
        if len(parts) == 2:
            parts[0] = parts[0].removesuffix(",").removesuffix(" AND")
    elif " AND " in text:
        relation = "and"
        parts = text.split(" AND ")
    else:
        parts = [text]
    if len(parts) > 2:
        return None
    operations = tuple(_operation(part, atoms) for part in parts)
    if any(op is None for op in operations):
        return None
    return Instruction(operations, relation, temporal_relation, event, condition)


def _compose_operation(op: Operation) -> str:
    # 中文实现直接从结构槽位展开，
    # 量词和禁止优先于普通动作。
    # required保留必须执行的义务，
    # subject与location采用各自语序。
    # 去毛刺语法需要明确“对”宾语，
    # 不会使用通用词序拼接全部操作。
    obj = ("所有" if op.all_objects else "") + op.object.target
    modal = "不得" if op.negated else "必须" if op.required else ""
    if op.subject:
        return op.subject.target + modal + op.action.target + obj
    if op.location:
        return obj + modal + "在" + op.location.target + "内" + op.action.target
    if op.required:
        return obj + modal + op.action.target
    if op.action.key == "deburr":
        return ("不要" if op.negated else "") + "对" + obj + op.action.target
    return ("不要" if op.negated else "") + op.action.target + obj


def compose_instruction(source: str, glossary: Glossary) -> PhraseEnhancement | None:
    ast = parse_instruction(source, glossary)
    if ast is None:
        return None
    # 连接符取决于源AST的关系，
    # 条件与时间位置由结构统一实现。
    # 新句不是命中某条完整人工译文，
    # 需汇集所有原子来源再加机器语法证据。
    separator = {"single": "", "then": "，然后", "and": "，并"}[ast.relation]
    target = separator.join(_compose_operation(op) for op in ast.operations)
    if ast.event:
        target = ast.event.target + ("前" if ast.temporal_relation == "before" else "后") + "，" + target
    if ast.condition:
        target = ("如有规定，" if ast.condition == "if_specified" else "除非另有规定，") + target
    # 动作、对象、主语、位置逐一取证，
    # 事件原子同样列入来源链。
    # 相同证据去重不删除不同来源，
    # 最终PhraseEnhancement保持机器组合。
    evidence: list[PhraseEvidence] = []
    for op in ast.operations:
        for atom in (op.action, op.object, op.subject, op.location):
            if atom is not None:
                evidence.extend(atom.evidence)
    if ast.event:
        evidence.extend(ast.event.evidence)
    evidence.append(PhraseEvidence(source, target, None, "machine", "instruction_grammar", "finite-instruction-v1"))
    return PhraseEnhancement(target, tuple(dict.fromkeys(evidence)), 0, 0)


@dataclass(frozen=True)
class _TargetOperation:
    action: str
    object: str
    all_objects: bool
    negated: bool
    required: bool
    subject: str | None = None
    location: str | None = None


# 中文候选先按长度逆序排列，
# 避免短词抢占较长专业对象。
# re.escape保证术语按字面匹配，
# 不会把词条内容当作正则规则。
def _alternatives(values: tuple[str, ...]) -> str:
    return "(?:" + "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True)) + ")"


_MODAL = r"(?P<modal>不得|不能|不要|禁止|不许|不可|必须|应当|应|须|需|不)?"
_NEGATIVE = frozenset({"不得", "不能", "不要", "禁止", "不许", "不可", "不"})


def _target_operation(text: str, atoms: dict[str, Atom]) -> _TargetOperation | None:
    # Full matches allow a small number of Chinese orderings, never arbitrary
    # filler between keyword hits. Longer object atoms are independent roles.
    # 目标只解析允许动作与已知对象，
    # 普通中文填词不会被忽略。
    # cover与slide保留特殊角色语序，
    # 不混入一般动宾的宽松模板。
    for action in atoms.values():
        if action.key not in {*_ACTION_FORMS.values(), "cover", "slide"}:
            continue
        action_re = _alternatives(action.targets)
        for key in _OBJECT_KEYS:
            obj = atoms.get(key)
            if obj is None:
                continue
            object_re = _alternatives(obj.targets)
            patterns = []
            if action.key == "cover" and "coating" in atoms:
                patterns = [(_alternatives(atoms["coating"].targets) + _MODAL
                             + action_re + r"(?P<all>所有|全部)?" + object_re, "coating", None)]
            elif action.key == "slide" and "die" in atoms:
                patterns = [(object_re + _MODAL + "在" + _alternatives(atoms["die"].targets)
                             + r"(?:内|中)" + action_re, None, "die")]
            else:
                patterns = [
                    (_MODAL + r"(?:对)?(?P<all>所有|全部)?" + object_re + r"(?:进行)?" + action_re, None, None),
                    (_MODAL + action_re + r"(?P<all>所有|全部)?" + object_re, None, None),
                    (r"(?P<all>所有|全部)?" + object_re + _MODAL + r"(?:进行)?" + action_re, None, None),
                ]
            for pattern, subject, location in patterns:
                match = re.fullmatch(pattern, text)
                if match:
                    # 中文义务与禁止在同一谓词判断，
                    # 负向情态不再同时算正向义务。
                    # 返回角色键而非原句字符串，
                    # 允许受控同义词与合理语序变化。
                    modal = match.groupdict().get("modal")
                    negative = modal in _NEGATIVE
                    return _TargetOperation(action.key, key, bool(match.groupdict().get("all")),
                                            negative, bool(modal) and not negative, subject, location)
    return None


def _parse_target(target: str, atoms: dict[str, Atom]) -> tuple | None:
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", target)).strip("。.")
    # 条件短语必须在目标起始完整消费，
    # 若与除非分别对应不同逻辑。
    # 后继“则/否则”也按原极性限定，
    # 其余副词留下即导致解析失败。
    condition = None
    for prefix, kind in (("除非另有规定", "unless_specified"), ("除非指定", "unless_specified"),
                         ("如有规定", "if_specified"), ("若有规定", "if_specified"), ("若指定", "if_specified")):
        if text.startswith(prefix):
            condition = kind
            text = text[len(prefix):].lstrip(",，")
            # Existing Integrated composition uses 若...则... / 除非...否则....
            # These pairs retain opposite condition polarity; the other adverb
            # remains unconsumed and therefore cannot be certified by this grammar.
            adverb = "则" if kind == "if_specified" else "否则"
            if text.startswith(adverb):
                text = text[len(adverb):].lstrip(",，")
            break
    # 目标事件必须与受控原子同义，
    # 前后关系独立保留用于比对。
    # 只凭出现“装配”不能确定先后，
    # 事件子句后主体仍要全量解析。
    event_key = temporal = None
    for key in _EVENT_KEYS:
        if key not in atoms:
            continue
        match = re.fullmatch(r"(?:在)?" + _alternatives(atoms[key].targets) + r"(前|后)[,，]?(.*)", text)
        if match:
            event_key, temporal = key, "before" if match[1] == "前" else "after"
            text = match[2]
            break
    # Separators are explicit; a THEN chain may use Chinese 先...再....
    relation = "single"
    # 顺序连接与并列连接分别分句，
    # 仅两段THEN可移除第一段“先”。
    # 未知多段链不合并为单一操作，
    # 每段目标必须被操作语法消费。
    if re.search(r"然后|随后|接着|再", text):
        parts = re.split(r"[,，]?(?:然后|随后|接着|再)", text)
        relation = "then"
        if len(parts) == 2:
            parts[0] = parts[0].removeprefix("先")
    elif re.search(r"并且|并|且|和|及", text):
        parts = re.split(r"[,，]?(?:并且|并|且|和|及)", text)
        relation = "and"
    else:
        parts = [text]
    if len(parts) not in (1, 2):
        return None
    operations = tuple(_target_operation(part, atoms) for part in parts)
    if any(op is None for op in operations):
        return None
    return operations, relation, temporal, event_key, condition


# 角色键含对象、极性与全称，
# 主语和位置不能掉出结构比较。
# 义务单独在末尾按计数检查，
# 避免正向义务与普通标签混为一谈。
def _operation_key(op: Operation | _TargetOperation) -> tuple:
    if isinstance(op, Operation):
        return (op.action.key, op.object.key, op.all_objects, op.negated,
                op.subject.key if op.subject else None, op.location.key if op.location else None)
    return op.action, op.object, op.all_objects, op.negated, op.subject, op.location


def lexical_warnings(source: str, target: str, glossary: Glossary) -> tuple[str, ...]:
    """Protect only unambiguous atomic noun senses in narrow source contexts.

    This can check an unsupported sentence but says nothing about its complete
    meaning, omitted modifiers, action binding, conditions or word order.
    """
    text = _source(source)
    target_text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", target))
    keys: list[str] = []
    warnings: list[str] = []
    # DEBURR itself denotes burr removal, even when its object is outside our
    # noun grammar. Permit an intervening object phrase in 去除交叉孔毛刺;
    # this is still only process vocabulary protection, not role validation.
    if (re.search(r"\bDEBURR\b", text)
            and not re.search(r"(?:去除?|清除|清理)[^。.!！?？;；,，]*毛刺", target_text)):
        warnings.append("instruction_lexical_missing:deburr")
    if re.search(r"\bBURRS?\b", text) and re.search(r"\b(?:REMOVE|REMOVAL|DEBURR|NO)\b|\bFREE OF\b", text):
        keys.append("burr")
    # 词义检查仅在明确机械语境启用，
    # 冲头/模具名词不能翻成冲压动作。
    # 力施加到冲头的有限表达也覆盖，
    # 但不生成力值或改写原量纲。
    # 关键词覆盖只证明局部术语存在，
    # 不说明整句条件或绑定正确。
    punch_die_nouns = (re.search(r"\bPUNCH(?:ES)? AND DIE(?:S)?\b", text)
                       or (re.search(r"\b(?:THE|A) PUNCH\b", text)
                           and re.search(r"\bDIE\b", text)
                           and re.search(r"\bSLIDE FREELY\b|\bFITS?\b|\bCLEARANCE\b", text)))
    if punch_die_nouns:
        keys.extend(("punch", "die"))
    # In this explicit force application grammar PUNCH is the mechanical noun
    # receiving the load. Keep the source's number/unit untouched; this guard
    # does not generate force instructions or infer a sense from bare PUNCH.
    if re.fullmatch(r"(?:DO NOT )?APPLY [+-]?(?:\d+\.\d+|\.\d+|\d+)\s*N FORCE TO THE PUNCH", text):
        keys.append("punch")
    if (re.search(r"\bCOATING\b", text) and re.search(r"\bTHREADS?\b", text)
            and (re.search(r"\bCOVER(?:S|ED)?\b", text)
                 or (re.search(r"\bKEEP\b", text) and re.search(r"\bOFF\b", text)))):
        keys.extend(("coating", "thread"))
    if not keys:
        return tuple(warnings)
    # 源输入只接收受控字母与逗号语法，
    # 尺寸、锐边或歧义句不进入组合。
    # 先剥完整条件再识别时间事件，
    # 剩余主体仍须完整解析。
    # IF与UNLESS保存相反条件极性。
    atoms = _atoms(glossary)
    warnings.extend("instruction_lexical_missing:" + key for key in keys
                    if key in atoms and not any(t in target_text for t in atoms[key].targets))
    return tuple(warnings)


def validate_instruction(source: str, target: str, glossary: Glossary) -> tuple[str, ...]:
    """Return conflicts/unconfirmed warnings only for a supported source AST.

    Empty warnings for an unsupported source mean 'outside this guard', not a
    statement of correctness. Consumers must retain their existing safety gates.
    """
    ast = parse_instruction(source, glossary)
    if ast is None:
        return ()
    atoms = _atoms(glossary)
    # 源受支持而目标未解析时保留未确认，
    # 缺动作、缺对象另给具体原因。
    # 不会因关键词都出现就放行未知中文。
    parsed = _parse_target(target, atoms)
    if parsed is None:
        warnings = []
        for op in ast.operations:
            if not any(t in target for t in op.action.targets):
                warnings.append("instruction_action_missing:" + op.action.key)
            for atom in (op.object, op.subject, op.location):
                if atom and not any(t in target for t in atom.targets):
                    warnings.append("instruction_object_missing:" + atom.key)
        return tuple(dict.fromkeys([*warnings, "instruction_target_unconfirmed"]))
    operations, relation, temporal, event_key, condition = parsed
    warnings = []
    source_keys = tuple(_operation_key(op) for op in ast.operations)
    target_keys = tuple(_operation_key(op) for op in operations)
    # AND两边可按多重集比较，
    # THEN仍要求同样的操作顺序。
    # 重复操作次数不能被集合去掉，
    # 事件、条件与关系分别检查。
    if relation == ast.relation == "and":
        equal_operations = sorted(source_keys, key=repr) == sorted(target_keys, key=repr)
    else:
        equal_operations = source_keys == target_keys
    if not equal_operations:
        warnings.append("instruction_action_object_binding_conflict")
    if relation != ast.relation or temporal != ast.temporal_relation or event_key != (ast.event.key if ast.event else None):
        warnings.append("instruction_relation_conflict")
    if condition != ast.condition:
        warnings.append("instruction_condition_conflict")
    # A prohibition itself carries obligation. Positive MUST/SHALL additionally
    # needs an explicit Chinese obligation, not merely a process label.
    # 禁止本身包含负向义务，
    # 正向MUST/SHALL需中文显式情态。
    # 按角色计数核对每个义务，
    # 不能用另一对象的“必须”补足。
    required_source = Counter(_operation_key(op) for op in ast.operations if op.required and not op.negated)
    required_target = Counter(_operation_key(op) for op in operations if op.required and not op.negated)
    if any(count > required_target[key] for key, count in required_source.items()):
        warnings.append("instruction_obligation_missing")
    return tuple(warnings)
