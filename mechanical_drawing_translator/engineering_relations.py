# mechanical_drawing_translator/engineering_relations.py
"""Explainable, bounded bilingual engineering relations; no answer generation.

The parser consumes a complete supported source and target, then compares typed
roles rather than word presence or sentence identity. It is deliberately not a
general semantic parser. ``out_of_scope`` is no coverage, ``unconfirmed`` means
the recognized source has no fully parsed Chinese counterpart, and ``matched``
only describes these finite relations, never professional approval. Existing
literal, terminology, policy, provenance and export gates remain necessary.

The small surface lexicon describes machining actions/geometry and coating
references; it grants no new material equivalences or reviewed knowledge IDs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Literal
import unicodedata


# 关系检查只描述有限语法，
# 不生成专业译文或人工批准。
# 动作、对象、否定与全称独立，
# 页码及总页数也分别保留。
# 同词不同角色不能互相抵消。
@dataclass(frozen=True)
class RelationUnit:
    kind: str
    predicate: str
    object: str
    negative: bool = False
    all_objects: bool = False
    sheet: str | None = None
    sheet_total: str | None = None


@dataclass(frozen=True)
# 四种状态必须按覆盖范围解释：
# matched只说明已解析关系相同；
# conflict表示已解析事实冲突；
# unconfirmed表示目标证据不足；
# out_of_scope表示源句未覆盖。
# 空警告不能等同于整句正确。
class RelationAssessment:
    status: Literal['matched', 'conflict', 'unconfirmed', 'out_of_scope']
    source_supported: bool
    target_supported: bool
    source_units: tuple[RelationUnit, ...] = ()
    target_units: tuple[RelationUnit, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Meaning:
    units: tuple[RelationUnit, ...]
    ordered: bool = False


# These are syntax-local engineering readings, not a general glossary and not
# evidence that new complete Chinese sentences were reviewed by a human.
# 词表把形态变体归到同一动作，
# 不在这里开放材料等价或词库。
# 中文同义词只影响表面实现，
# 角色键仍用于后续严格比较。
_SOURCE_ACTIONS = {
    'GRIND': 'grind', 'GRINDING': 'grind',
    'POLISH': 'polish', 'POLISHING': 'polish',
    'COAT': 'coat', 'COATING': 'coat',
}
_TARGET_ACTIONS = {
    'grind': ('研磨', '磨削'), 'polish': ('抛光',),
    'coat': ('涂覆', '涂层'),
}
_SOURCE_OBJECTS = {
    'TOP SURFACE': 'top_surface', 'BOTTOM SURFACE': 'bottom_surface',
    'FORM SURFACE': 'form_surface', 'FORM SURFACES': 'form_surface',
    'SURFACE': 'surface', 'SURFACES': 'surface',
}
_TARGET_OBJECTS = {
    # Preserve the existing mechanical_instructions Atom realizations.
    'top_surface': ('顶面', '顶部表面', '上表面'),
    'bottom_surface': ('底面', '底部表面', '下表面'),
    'form_surface': ('成型面', '成形面'), 'surface': ('表面', '面'),
}
_NEGATIVE = r'不得|不要|不允许|不可|不能|不许|禁止|严禁|勿|不'
_MODAL = r'应|须|需|必须'
_COATED_AREA = r'(?:涂层|涂覆)(?:区域|区)'
_TOPIC = r'(?:涂层|涂覆)(?:规范|规格|技术要求)?'
_REFERENCE_VERB = r'(?:请)?(?:参见|详见|见)'


# 英语压缩空白并统一大小写，
# 仅移除末尾句点以适配短说明。
# 中文去空白与末尾分隔符，
# 其余内容必须由完整语法消费。
# 这些规范化不修改源记录。
def _english(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).upper().split()).rstrip('.').strip()


def _chinese(value: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value)).rstrip('。.；;')


def _source_operation(text: str, *, allow_event: bool = False) -> RelationUnit | None:
    # 完整匹配阻止只凭关键词通过，
    # 动作或对象未登记即无覆盖。
    # 动名词仅能在外层时间事件启用，
    # 避免把标题误作加工命令。
    match = re.fullmatch(r'(DO NOT )?([A-Z]+) (?:THE )?(ALL )?([A-Z ]+)', text.strip())
    if match is None:
        return None
    action, obj = _SOURCE_ACTIONS.get(match[2]), _SOURCE_OBJECTS.get(match[4])
    if action is None or obj is None or (match[2].endswith('ING') and not allow_event):
        return None
    # Gerunds are parsed only as event clauses by the outer temporal grammar;
    # accepting them here does not authorize arbitrary gerund-headed prose.
    return RelationUnit('action_object', action, obj, bool(match[1]), bool(match[3]))


def _source_reference(text: str) -> _Meaning | None:
    # A vertical bar is a known OCR/group separator only inside this complete
    # reference grammar; it is never erased from arbitrary instructions.
    text = re.sub(r'\s*[|;]\s*', ' ', text)
    # 涂覆区标签与引用可分别成事实，
    # 允许独立标签而不强补页码。
    # SHEET OF中的两数承担不同角色，
    # 交换页号与总页数会形成冲突。
    # 引用主题也区分涂覆与涂覆规范。
    units: list[RelationUnit] = []
    if text.startswith('COATED AREA'):
        units.append(RelationUnit('area', 'coated', 'area'))
        text = text[len('COATED AREA'):].lstrip(' ,:')
        if not text:
            return _Meaning(tuple(units))
    if text in ('SPECIFICATIONS', 'COATING SPECIFICATIONS'):
        units.append(RelationUnit('specifications', 'specification',
                                  'coating' if text.startswith('COATING') else 'unspecified'))
        return _Meaning(tuple(units))
    match = re.fullmatch(r'SEE SHEET (\d{1,4})(?: OF (\d{1,4}))? FOR COATING( SPECIFICATIONS)?', text)
    if match is None:
        return None
    units.append(RelationUnit('reference', 'see', 'coating_specifications' if match[3] else 'coating',
                              sheet=match[1], sheet_total=match[2]))
    return _Meaning(tuple(units))


def _source_meaning(text: str) -> _Meaning | None:
    reference = _source_reference(text)
    if reference is not None:
        return reference
    # An explicit temporal connective yields chronological roles, independent
    # of the order in which its clauses were written.
    # 时间句先转成真实先后顺序，
    # 不沿原文子句位置直接比较。
    # BEFORE与AFTER方向在此归一，
    # THEN仍保留有序操作关系。
    # 不能解析的组合退回单句尝试，
    # 不会截取其中一个已知动作。
    candidates = []
    then = re.fullmatch(r'(.+?),? THEN (.+)', text)
    if then:
        candidates.append((then[1], then[2], False))
    initial = re.fullmatch(r'(BEFORE|AFTER) (.+?), (.+)', text)
    if initial:
        candidates.append((*((initial[3], initial[2]) if initial[1] == 'BEFORE'
                            else (initial[2], initial[3])), True))
    infix = re.fullmatch(r'(.+?) (BEFORE|AFTER) (.+)', text)
    if infix:
        candidates.append((*((infix[1], infix[3]) if infix[2] == 'BEFORE'
                            else (infix[3], infix[1])), True))
    for before, after, temporal in candidates:
        first = _source_operation(before, allow_event=temporal)
        second = _source_operation(after, allow_event=temporal)
        if first is not None and second is not None:
            # Retain the application's established ambiguity policy: a leading
            # prohibition may scope over the complete following THEN workflow.
            # Only a final-step prohibition has unambiguous local scope here.
            if not temporal and first.negative:
                continue
            # Negated BEFORE/AFTER may prohibit an event within a time scope,
            # rather than prohibit the event permanently. No scope is guessed.
            if temporal and (first.negative or second.negative):
                continue
            return _Meaning((first, second), ordered=True)
    operation = _source_operation(text)
    return _Meaning((operation,)) if operation is not None else None


def _target_operation(text: str) -> RelationUnit | None:
    # 中文允许有限主动与把字语序，
    # 但否定必须绑定当前谓词。
    # 全称量词绑定当前加工对象，
    # 不会被别处的“全部”补足。
    # fullmatch拒绝额外未消费内容。
    text = text.removeprefix('先')
    for action, verbs in _TARGET_ACTIONS.items():
        verb = '(?:' + '|'.join(map(re.escape, verbs)) + ')'
        for obj, names in _TARGET_OBJECTS.items():
            noun = '(?:' + '|'.join(map(re.escape, names)) + ')'
            # Polarity belongs to this individual predicate. In particular,
            # 不得 cannot migrate to the next operation in a sequence.
            modifier = rf'(?P<modifier>{_NEGATIVE}|{_MODAL})?'
            quantified = rf'(?P<all>所有|全部|全)?{noun}'
            patterns = (
                modifier + verb + quantified,
                modifier + r'(?:将|把|对)?' + quantified + r'(?:进行)?' + verb,
                r'(?:将|把|对)?' + quantified + modifier + r'(?:进行)?' + verb,
                modifier + r'进行' + quantified + verb,
            )
            for pattern in patterns:
                match = re.fullmatch(pattern, text)
                if match:
                    negative = bool(re.fullmatch(_NEGATIVE, match['modifier'] or ''))
                    return RelationUnit('action_object', action, obj, negative, bool(match['all']))
    return None


# 页码只接收所列明确写法，
# 页与总页数保持字符串字面。
# 这里不打开或验证真实PDF页，
# 仅比较译文是否保留引用关系。
def _sheet(text: str) -> tuple[str, str | None] | None:
    # 主题与引用动词允许有限互换，
    # 页码子串必须再经专用解析。
    # 匹配到“见”不代表引用成立，
    # 未解析完整的引用仍是不确定。
    patterns = (
        r'(?:表|页)(\d{1,4})(?:/(\d{1,4}))?',
        r'(?:图纸)?第?(\d{1,4})(?:张图纸|页|张)(?:[,，]?\(?共(\d{1,4})(?:页|张)\)?)?',
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return match[1], match[2]
    return None


def _target_reference(text: str) -> _Meaning | None:
    units: list[RelationUnit] = []
    # Area and reference are independent assertions: a fronted purpose/reference
    # may precede the area label without changing its manufacturing meaning.
    # 区标签可置于引用前或引用后，
    # 此类无序事实可按多重集比较。
    # 仅接受已知分隔位置的区标签，
    # 不会删除任意句中的涂覆文字。
    area_start = re.match(_COATED_AREA + r'[,，;；。|:：]?', text)
    if area_start:
        units.append(RelationUnit('area', 'coated', 'area'))
        text = text[area_start.end():]
    else:
        area_end = re.search(r'[,，;；。|]' + _COATED_AREA + '$', text)
        if area_end:
            units.append(RelationUnit('area', 'coated', 'area'))
            text = text[:area_end.start()]
    if not text and units:
        return _Meaning(tuple(units))
    specification = re.fullmatch(r'(?P<coating>涂层|涂覆)?(?:规范|规格|技术要求)', text)
    if specification:
        units.append(RelationUnit('specifications', 'specification',
                                  'coating' if specification['coating'] else 'unspecified'))
        return _Meaning(tuple(units))
    patterns = (
        rf'(?:有关|关于)?(?P<topic>{_TOPIC})[,，]?{_REFERENCE_VERB}(?P<sheet>.+)',
        rf'{_REFERENCE_VERB}(?P<sheet>.+?)(?:的|对于|以了解|了解|有关|关于|获取|查询)?(?P<topic>{_TOPIC})',
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match is None:
            continue
        page = _sheet(match['sheet'])
        if page is None:
            continue
        topic = ('coating_specifications' if re.search(r'规范|规格|技术要求', match['topic'])
                 else 'coating')
        units.append(RelationUnit('reference', 'see', topic, sheet=page[0], sheet_total=page[1]))
        return _Meaning(tuple(units))
    return None


# 按源侧关系类型选择目标语法，
# 引用不能被加工动作语法代替。
# 时间前后与THEN分别解析，
# 否定时间句仍保持范围拒绝。
# 源和目标必须都具有相同有序性。
def _target_meaning(text: str, *, reference: bool) -> _Meaning | None:
    if reference:
        return _target_reference(text)
    candidates = []
    explicit = re.fullmatch(r'(.+?)[,，]?(?:然后|随后|接着|再)(.+)', text)
    if explicit:
        candidates.append((explicit[1], explicit[2], False))
    initial = re.fullmatch(r'在(.+?)(之前|之后|前|后)[,，]?(?:应|须|需|必须)?(?:先)?(.+)', text)
    if initial:
        candidates.append((*((initial[3], initial[1]) if initial[2] in {'之前', '前'}
                            else (initial[1], initial[3])), True))
    infix = re.fullmatch(r'(.+?)(之前|之后|前|后)[,，]?(?:应|须|需|必须)?(?:先)?(.+)', text)
    if infix:
        candidates.append((*((infix[3], infix[1]) if infix[2] in {'之前', '前'}
                            else (infix[1], infix[3])), True))
    for before, after, temporal in candidates:
        first, second = _target_operation(before), _target_operation(after)
        if first is not None and second is not None:
            if temporal and (first.negative or second.negative):
                continue
            return _Meaning((first, second), ordered=True)
    operation = _target_operation(text)
    return _Meaning((operation,)) if operation is not None else None


# 无序事实按计数比较而非集合，
# 重复次数改变也必须被发现。
# 同一动作组仅顺序相反时，
# 给出专门的顺序冲突原因。
# 动作、对象、否定、全称分别检查，
# 便于归因而非笼统报告误译。
def _conflicts(expected: _Meaning, actual: _Meaning) -> tuple[str, ...]:
    if expected == actual or (not expected.ordered and not actual.ordered
                              and Counter(expected.units) == Counter(actual.units)):
        return ()
    if expected.ordered and actual.ordered and Counter(expected.units) == Counter(actual.units):
        return ('engineering_relation_order_conflict',)
    warnings = []
    if any(unit.kind == 'reference' for unit in expected.units + actual.units):
        if Counter(expected.units) != Counter(actual.units):
            warnings.append('engineering_relation_reference_conflict')
    else:
        if len(expected.units) != len(actual.units):
            return ('engineering_relation_action_object_conflict',)
        for source, target in zip(expected.units, actual.units):
            if (source.kind, source.predicate, source.object) != (target.kind, target.predicate, target.object):
                warnings.append('engineering_relation_action_object_conflict')
            if source.negative != target.negative:
                warnings.append('engineering_relation_negation_scope_conflict')
            if source.all_objects != target.all_objects:
                warnings.append('engineering_relation_quantifier_conflict')
    return tuple(dict.fromkeys(warnings)) or ('engineering_relation_content_conflict',)


def assess(source: str, target: str) -> RelationAssessment:
    """Return typed evidence and explicit coverage, without approving a translation.

    Maximum 2,000 characters bounds this supplemental guard's per-item cost.
    Unsupported sources return no blocking warning: the service must retain its
    existing gates and must never interpret this status as a correctness pass.
    Recognized sources with unparsed targets get a distinct blocking warning,
    not an asserted semantic conflict or an invented professional translation.
    """
    # 两千字符上限约束附加守卫成本，
    # 超范围结果不阻断其它既有检查。
    # 源受支持而中文未完整解析时，
    # 使用target_unconfirmed阻断候选。
    # 不能把未知中文说成已证实冲突，
    # 更不能借此自动补写答案。
    if not isinstance(source, str) or not source.strip() or len(source) > 2000:
        return RelationAssessment('out_of_scope', False, False)
    expected = _source_meaning(_english(source))
    if expected is None:
        return RelationAssessment('out_of_scope', False, False)
    actual = (_target_meaning(_chinese(target), reference=expected.units[0].kind != 'action_object')
              if isinstance(target, str) and target.strip() and len(target) <= 2000 else None)
    if actual is None or actual.ordered != expected.ordered:
        return RelationAssessment('unconfirmed', True, actual is not None,
                                  expected.units, actual.units if actual else (),
                                  ('engineering_relation_target_unconfirmed',))
    # 返回的角色快照是诊断证据，
    # 不替换原始英文或中文文本。
    # 调用者还必须执行字面量守卫，
    # 以及术语、来源与导出门禁。
    warnings = _conflicts(expected, actual)
    return RelationAssessment('conflict' if warnings else 'matched', True, True,
                              expected.units, actual.units, warnings)


def validate(source: str, target: str) -> tuple[str, ...]:
    """Blocking supplement; use :func:`assess` when reporting semantic coverage."""
    return assess(source, target).warnings
