# mechanical_drawing_translator/translation_reference_safety.py
"""Additional finite relation checks; no translation or reference-answer lookup.

An empty result for an unsupported source means outside this guard, never
confirmed correctness. For a fully recognized source, an unconsumed Chinese
realization is explicitly unconfirmed. Existing engineering gates still apply.
"""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .mechanical_instructions import (
    _alternatives, _atoms, _operation_key, _parse_target, parse_instruction,
)
from .translation_glossary import Glossary
from .translation_sequence import compose_instructions


# 该守卫只检查有限受力与THEN关系，
# 不生成译文或查询参考答案。
# 空结果在源不支持时表示无覆盖，
# 不是向服务声明全文专业正确。
_NUMBER = r'[+-]?(?:\d+\.\d+|\.\d+|\d+)'
_NEGATION = r'不得|不要|不能|禁止|不许|不可|严禁|勿|不'


def _source(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).upper().split()).removesuffix('.').strip()


def _force_warnings(source: str, target: str, atoms) -> tuple[str, ...] | None:
    # This specific source grammar establishes a receiver and a literal force.
    # It does not assign a tooling sense to bare PUNCH or arbitrary prose.
    # 受力语法明确数值、N单位与接收者冲头，
    # 孤立PUNCH不推断成工具名词。
    # 源完整匹配才建立角色，
    # 其它载荷形式继续由原工程门禁负责。
    source_match = re.fullmatch(rf'(DO NOT )?APPLY ({_NUMBER})\s*N FORCE TO THE PUNCH', source)
    if source_match is None or 'punch' not in atoms:
        return None
    # 目标允许有限“向/对…施加”语序，
    # 或“力作用于”同一接收者。
    # 否定与受力动作在同一语法绑定，
    # 额外中文子句不会因含冲头而通过。
    receiver = _alternatives(atoms['punch'].targets)
    amount = rf'(?P<amount>{_NUMBER})(?:N|牛顿)(?:的)?力'
    neg = rf'(?P<negative>{_NEGATION})?'
    text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', target)).strip('。.')
    # Full matches bind the force to the receiver. A second unparsed clause or
    # another receiver cannot be justified by an incidental occurrence of 冲头.
    patterns = (
        neg + rf'(?:向|对){receiver}施加' + amount,
        rf'(?:向|对){receiver}' + neg + '施加' + amount,
        amount + neg + rf'作用于{receiver}',
    )
    match = next((m for pattern in patterns if (m := re.fullmatch(pattern, text))), None)
    # 源受支持但目标无法完整解析，
    # 给target_unconfirmed而非无覆盖。
    # 数值保留符号与小数精度字符串，
    # 0.10与0.1不借数值换算视为相同。
    if match is None:
        return ('reference_force_target_unconfirmed',)
    warnings = []
    # Literal spelling deliberately retains sign and decimal precision; the
    # existing engineering validator remains responsible for all other tokens.
    if match['amount'] != source_match[2]:
        warnings.append('reference_force_value_conflict')
    if bool(match['negative']) != bool(source_match[1]):
        warnings.append('reference_force_polarity_conflict')
    return tuple(warnings)


# GRIND TO FIT保存适配目的角色，
# “研磨冲头头部”不是同一要求。
# 动作直接作用对象与目的对象分开，
# 不能仅按相同名词和动词判等价。
def _fit_source_role(source: str, atoms):
    if not {'grind', 'punch_head'}.issubset(atoms):
        return None
    match = re.fullmatch(r'(DO NOT )?GRIND TO FIT (?:THE )?PUNCH HEAD', _source(source))
    if match:
        # Purpose is a separate role: grinding a head is not necessarily
        # grinding something to fit that head.
        return ('fit', 'grind', 'punch_head', bool(match[1]))
    return None


# 先解析有限适配目的表达，
# 再尝试单操作无条件无时间的原AST。
# 复杂操作不会压缩成一个原子角色，
# 避免隐藏局部条件或顺序范围。
def _source_role(source: str, atoms, glossary: Glossary):
    fit = _fit_source_role(source, atoms)
    if fit is not None:
        return fit
    parsed = parse_instruction(source, glossary)
    if (parsed is not None and len(parsed.operations) == 1 and parsed.relation == 'single'
            and parsed.temporal_relation is None and parsed.condition is None):
        return ('atomic', _operation_key(parsed.operations[0]))
    return None


# 中文适配目的也必须被完整解析，
# 其它原子角色复用同一目标AST语法。
# 不同动作、对象或否定保留不同键，
# 不将“为了”当成可丢弃修饰。
def _target_role(target: str, atoms):
    if {'grind', 'punch_head'}.issubset(atoms):
        match = re.fullmatch(
            rf'(?P<negative>{_NEGATION})?'
            + _alternatives(atoms['grind'].targets)
            + r'(?:为了|以|为)?适配'
            + _alternatives(atoms['punch_head'].targets), target)
        if match:
            return ('fit', 'grind', 'punch_head', bool(match['negative']))
    parsed = _parse_target(target, atoms)
    if parsed is not None:
        operations, relation, temporal, event, condition = parsed
        if len(operations) == 1 and relation == 'single' and temporal is None and event is None and condition is None:
            return ('atomic', _operation_key(operations[0]))
    return None


def validate_reference_relations(source: str, target: str, glossary: Glossary) -> tuple[str, ...]:
    """Check a literal punch force or a fully recognized audited THEN chain.

    Supported Chinese force forms are 向/对 receiver 施加 amount and amount
    作用于 receiver, optionally negated. Supported order forms use explicit
    再/然后/随后/接着. These bounded forms may reject valid unlisted wording;
    callers must preserve the English and unconfirmed reason in that case.
    """
    normalized_source = _source(source)
    if (not re.search(r'\bTHEN\b',normalized_source)
            and re.fullmatch(rf'(?:DO NOT )?APPLY {_NUMBER}\s*N FORCE TO THE PUNCH',normalized_source) is None):
        return ()
    # 入口先过滤到明确THEN或受力语法，
    # 减少不相关输出的附加解析开销。
    # 原子词义仍依赖当前合资格词表，
    # 缺少必要词义证据就保持无覆盖。
    atoms = _atoms(glossary)
    force = _force_warnings(normalized_source, target, atoms)
    if force is not None:
        return force
    if not re.search(r'\bTHEN\b', normalized_source):
        return ()
    composition = compose_instructions(source, glossary)
    if composition is None:
        return ()
    # 顺序链要求每个原叶子都有完整角色，
    # 不可解析的一项使本守卫无整链覆盖。
    # 中文按明确再/然后等分段，
    # 段数必须与真实操作数一致。
    expected = tuple(_source_role(e.source_segment, atoms, glossary) for e in composition.evidence)
    if len(expected) < 2 or any(role is None for role in expected):
        return ()
    compact = re.sub(r'\s+', '', unicodedata.normalize('NFKC', target)).strip('。.')
    parts = re.split(r'[,，]?(?:然后|随后|接着|再)', compact)
    if len(parts) != len(expected):
        return ('reference_instruction_target_unconfirmed',)
    parts[0] = parts[0].removeprefix('先')
    actual = tuple(_target_role(part, atoms) for part in parts)
    if any(role is None for role in actual):
        return ('reference_instruction_target_unconfirmed',)
    # 角色序列一致才通过有限关系检查，
    # 相同多重集但不同顺序单列order冲突。
    # 其它差异为role冲突，
    # 未解析目标则保留unconfirmed。
    # 外层工程字面量与来源检查仍有效。
    if actual == expected:
        return ()
    if Counter(actual) == Counter(expected):
        return ('reference_instruction_order_conflict',)
    return ('reference_instruction_role_conflict',)
