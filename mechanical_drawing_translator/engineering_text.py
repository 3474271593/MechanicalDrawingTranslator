# mechanical_drawing_translator/engineering_text.py
"""Extract and structurally validate immutable engineering information."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# 字面量同时保存类型与原文跨度，
# start/end使用Python字符偏移。
# 这些跨度不是PDF字符坐标，
# 不得据此伪造逐字符图像框。
@dataclass(frozen=True)
class LiteralToken:
    kind: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class LiteralValidation:
    is_valid: bool
    warning_codes: tuple[str, ...] = ()


# 模式可重叠命中同一个数字，
# 如螺纹整体与内部尺寸。
# 先收集全部候选再统一选取，
# 避免模式顺序改变保护范围。
# 工程符号、单位与连接符属于值，
# 不能只保留阿拉伯数字集合。
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("angle", re.compile(
        r"(?:±\s*|[+-]?)(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*°"
    )),
    ("thread", re.compile(r"\bM\d+(?:\.\d+)?\s*[×xX]\s*\d+(?:\.\d+)?\b", re.I)),
    ("diameter", re.compile(r"[Ø⌀]\s*\d+(?:\.\d+)?")),
    ("model", re.compile(
        r"(?<![A-Z0-9])(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)"
        r"[A-Z0-9]+(?:-[A-Z0-9]+)*(?![A-Z0-9])", re.I,
    )),
    ("dimension_pair", re.compile(
        r"(?<![A-Z0-9_.])(?:[Ø⌀]\s*)?(?:±\s*|[+-]?)"
        r"(?:\d+\.\d+|\.\d+|\d+\s*/\s*\d+|\d+)"
        r"(?:\s*[×xX]\s*(?:±\s*|[+-]?)"
        r"(?:\d+\.\d+|\.\d+|\d+\s*/\s*\d+|\d+)){1,}"
        r"(?:\s*(?:mm|cm|µm|um|[\"″]))?(?![A-Z0-9_]|\.\d)", re.I,
    )),
    ("measurement", re.compile(
        r"(?<![A-Z0-9_.])(?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+|\d+\s*/\s*\d+)"
        r"\s*(?:mm|cm|µm|um)(?![A-Z])", re.I,
    )),
    ("inch_measurement", re.compile(
        r"(?<![A-Z0-9_.])(?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+|\d+\s*/\s*\d+)\s*[\"″]",
        re.I,
    )),
    ("fraction", re.compile(
        r"(?<![A-Z0-9_.])(?:±\s*|[+-]?)\d+\s*/\s*\d+(?![A-Z0-9_]|\.\d)",
        re.I,
    )),
    ("range", re.compile(
        r"(?<![A-Z0-9_.])\d+(?:\.\d+)?\s*[-~–—]\s*\d+(?:\.\d+)?"
        r"(?![A-Z0-9_]|\.\d)",
        re.I,
    )),
    ("number", re.compile(
        r"(?<![A-Z0-9_.])(?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+)"
        r"(?![A-Z0-9_]|\.\d|\s*[\"″])",
        re.I,
    )),
)


def extract_immutable_literals(text: str) -> tuple[LiteralToken, ...]:
    """Return longest non-overlapping literal matches in source order."""
    # 最长片段优先占据原文区间，
    # 有任何字符交叠即排除短候选。
    # 等长片段再按起点稳定排序，
    # 最终按源文顺序返回保护列表。
    # 这里不改写或分词原始英文。
    candidates: list[LiteralToken] = []
    for kind, pattern in _PATTERNS:
        candidates.extend(
            LiteralToken(kind, match.group(0), match.start(), match.end())
            for match in pattern.finditer(text)
        )
    selected: list[LiteralToken] = []
    for token in sorted(candidates, key=lambda item: (-(item.end - item.start), item.start)):
        if any(token.start < other.end and other.start < token.end for other in selected):
            continue
        selected.append(token)
    return tuple(sorted(selected, key=lambda item: item.start))


def validate_immutable_literals(source: str, translated: str) -> LiteralValidation:
    """Require engineering values to remain structurally equivalent and ordered.

    Formatting and language equivalents such as ``0.003 MM``/``0.003mm``,
    ``MM``/``毫米`` and ``X``/``×`` are normalized.  Precision, signs,
    degree/tolerance symbols, identifiers, value order and dimension connectors
    remain strict.
    """
    # Complete, typed expressions have semantic order, rather than arbitrary
    # bag-of-identifiers equivalence. Keep these grammars deliberately narrow.
    # 材料“新材WAS旧材”具有方向，
    # 中文“由旧改新”与英文反序。
    # 仅完整受控表达使用此特例，
    # 不能给任意两个型号放宽顺序。
    # 投影序数同样限定完整字段，
    # 第一角与第三角不能互换。
    material = re.fullmatch(r'\s*(?:UPDATED\s+BORDER\.\s*)?(CD\d+)\s+WAS\s+(CD\d+)\s*',source,re.I)
    replacement = re.fullmatch(r'\s*(?:图框已更新[。.]\s*)?材料由\s*(CD\d+)\s*改为\s*(CD\d+)\s*',translated,re.I)
    if material and replacement:
        valid=(material[1].upper(),material[2].upper())==(replacement[2].upper(),replacement[1].upper())
        return LiteralValidation(valid,() if valid else ('engineering_material_change_reversed_or_changed',))
    projection = re.fullmatch(r'\s*(1ST|3RD)\.?\s*ANG\.?\s*PROJ\.?\s*',source,re.I)
    if projection and translated.strip() in {'第一角投影','第三角投影'}:
        valid=translated.strip()==('第一角投影' if projection[1].upper()=='1ST' else '第三角投影')
        return LiteralValidation(valid,() if valid else ('engineering_projection_ordinal_changed',))
    # 源与目标采用同一源上下文归一，
    # 避免目标自行决定放宽规则。
    # 数量相同仍需逐项值与顺序相同，
    # 不能用排序掩盖尺寸顺序调换。
    # 缺失、增加、重排、改变分别归因，
    # 每次只返回首个结构错误类别。
    expected_values = _engineering_signature(source, source)
    actual_values = _engineering_signature(translated, source)
    if expected_values == actual_values:
        return LiteralValidation(True)

    if len(actual_values) < len(expected_values):
        return LiteralValidation(False, ("engineering_token_missing",))
    if len(actual_values) > len(expected_values):
        return LiteralValidation(False, ("engineering_token_added",))
    if sorted(expected_values) == sorted(actual_values):
        return LiteralValidation(False, ("engineering_token_reordered",))
    return LiteralValidation(False, ("engineering_token_changed",))


# 结构模式把数值与量纲成对保存，
# N与KN不可只因数字相同等价。
# 螺纹保留直径与螺距两个槽位，
# 范围保留上下界的原文顺序。
# 序数、列表号与普通数值分型，
# 避免模型把工程身份换成叙述。
_STRUCTURED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("force", re.compile(
        r"(?<![A-Z0-9_.])((?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+))\s*(KN|N)(?![A-Z])", re.I,
    )),
    ("thread", re.compile(r"\bM(\d+(?:\.\d+)?)\s*[×xX*]\s*(\d+(?:\.\d+)?)\b", re.I)),
    ("diameter", re.compile(r"[Ø⌀]\s*([+-]?(?:\d+\.\d+|\.\d+|\d+))")),
    ("identifier", re.compile(
        r"(?<![A-Z0-9])(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)"
        r"[A-Z0-9]+(?:-[A-Z0-9]+)*(?![A-Z0-9])", re.I,
    )),
    ("range", re.compile(
        r"(?<![A-Z0-9_.])([+-]?(?:\d+\.\d+|\.\d+|\d+))\s*[-~–—]\s*"
        r"([+-]?(?:\d+\.\d+|\.\d+|\d+))(?![A-Z0-9_.])", re.I,
    )),
    ("fraction_measurement", re.compile(
        r"(?<![A-Z0-9_.])((?:±\s*|[+-]?)\d+\s*/\s*\d+)\s*(MM|CM|UM)(?![A-Z])", re.I,
    )),
    ("measurement", re.compile(
        r"(?<![A-Z0-9_.])((?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+))\s*(MM|CM|UM)(?![A-Z])",
        re.I,
    )),
    ("inch", re.compile(
        r"(?<![A-Z0-9_.])((?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+|\d+\s*/\s*\d+))\s*[\"″]",
        re.I,
    )),
    ("angle", re.compile(
        r"(?<![A-Z0-9_.])((?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+|\d+\s*/\s*\d+))\s*°",
        re.I,
    )),
    ("ordinal", re.compile(r"(?<![A-Z0-9_.])(\d+)(?:ST|ND|RD|TH|次)(?![A-Z0-9_])", re.I)),
    ("fraction", re.compile(r"(?<![A-Z0-9_.])([+-]?\d+\s*/\s*\d+)(?![A-Z0-9_.])", re.I)),
    ("list_number", re.compile(r"(?<![A-Z0-9_.])(\d+)\.(?!\d)", re.I)),
    ("number", re.compile(
        r"(?<![A-Z0-9_.])((?:±\s*|[+-]?)(?:\d+\.\d+|\.\d+|\d+))(?![A-Z0-9_.])",
        re.I,
    )),
)


def _normalize_engineering_text(text: str) -> str:
    # 统一大小写与Unicode兼容字形，
    # 不把小数转为浮点数进行比较。
    # 因此0.10与0.1不会丢失精度差。
    # 中文量纲转成独立英文边界，
    # 保留紧随单位的正负公差。
    # 微米字符只归一字形与名称，
    # 不会换算为毫米并改写数值。
    value = unicodedata.normalize("NFKC", text).upper()
    # Chinese unit words are token boundaries. Without separators, MM±0.001
    # can cause the signed-number regex to start after ± and silently drop it.
    # This normalizes presentation only; values/precision/signs remain strict.
    value = value.replace("毫米", " MM ").replace("厘米", " CM ")
    value = value.replace("千牛顿", " KN ").replace("牛顿", " N ")
    value = value.replace("微米", " UM ").replace("ΜM", "UM").replace("Μ", "U")
    value = re.sub(r"R\s*最大", "RMAX", value, flags=re.I)
    value = re.sub(r"(?<=\d)\s*MICRO(?![A-Z])", " MICRO", value)
    return value


def _masked_specials(text: str, source_context: str) -> tuple[str, list[tuple[str, ...]]]:
    value = text
    source = _normalize_engineering_text(source_context)
    tokens: list[tuple[str, ...]] = []

    # 特殊表达先生成带类型签名，
    # 再用等长空白占住原来的跨度。
    # 保持索引位置便于后续匹配，
    # 也防止同一个数字被重复计数。
    def mask(match: re.Match[str], token: tuple[str, ...]) -> str:
        tokens.append(token)
        return " " * (match.end() - match.start())

    # SHEET编号仅在源明确表达时开放，
    # 中文页码斜线才可视作同一事实。
    # 普通分数不会自动升级为页码。
    # DET引用同样依赖源侧语境，
    # 不能任意忽略目标中新增的数。
    sheet_source = re.search(r"\bSHEET\s+(\d+)\s+OF\s+(\d+)\b", source, re.I)
    if sheet_source:
        pattern = (r"\bSHEET\s+(\d+)\s+OF\s+(\d+)\b" if value == source else
                   r"(?:表|页|SHEET)?\s*(\d+)\s*/\s*(\d+)")
        match = re.search(pattern, value, re.I)
        if match:
            value = value[:match.start()] + mask(
                match, ("sheet", match.group(1), match.group(2)),
            ) + value[match.end():]

    det_source = re.search(r"\bDET\s*\.\s*(\d+)\b", source, re.I)
    if det_source:
        pattern = r"\bDET\s*\.\s*(\d+)\b" if value == source else rf"(?<!\d){det_source.group(1)}(?!\d)"
        match = re.search(pattern, value, re.I)
        if match:
            value = value[:match.start()] + mask(
                match, ("reference", det_source.group(1)),
            ) + value[match.end():]

    # 粗糙度保留Ra与RMAX的不同身份，
    # 相同数值但指标改变仍不等价。
    # 循环遮盖可收集多个粗糙度值，
    # 每次替换后搜索剩余内容。
    roughness_patterns = (
        ("RA", re.compile(r"(?<![A-Z])RA\s*([+-]?(?:\d+\.\d+|\.\d+|\d+))", re.I)),
        ("RMAX", re.compile(r"(?<![A-Z0-9])([+-]?(?:\d+\.\d+|\.\d+|\d+))\s*RMAX\b", re.I)),
    )
    for marker, pattern in roughness_patterns:
        while True:
            match = pattern.search(value)
            if not match:
                break
            number = match.group(1)
            value = value[:match.start()] + mask(match, ("roughness", marker, number)) + value[match.end():]

    # 仅明确模型几何语境允许3D措辞，
    # 此处将中文“3D模型”视作表现形式。
    # 其它语境的数字3仍须保护，
    # 不能全局抹去所有3D字样。
    if re.search(r"\b(?:MODEL\s+GEOMETRY|SEE\s+MODEL|TO\s+MODEL)\b", source, re.I):
        match = re.search(r"(?<![A-Z0-9])3\s*D\s*(?:图|模型)?", value, re.I)
        if match:
            value = value[:match.start()] + " " * (match.end() - match.start()) + value[match.end():]
    return value, tokens


# 规范签名保留字符串精度与符号，
# 不同类型不因显示相似而合并。
# 英寸、角度、直径均有专门标识，
# 列表编号可等价为普通编号值。
# 只在已识别序数形式时归为序数。
def _canonical_match(kind: str, match: re.Match[str]) -> tuple[str, ...]:
    groups = tuple(re.sub(r"\s+", "", value.upper()) for value in match.groups())
    if kind == "identifier":
        value = re.sub(r"\s+", "", match.group(0).upper())
        ordinal = re.fullmatch(r"(\d+)(?:ST|ND|RD|TH)", value)
        return ("ordinal", ordinal.group(1)) if ordinal else (kind, value)
    if kind == "diameter":
        return (kind, groups[0])
    if kind == "thread":
        return (kind, *groups)
    if kind == "range":
        return (kind, *groups)
    if kind == "fraction_measurement":
        return ("measurement", groups[0], groups[1].replace("Μ", "U"))
    if kind in {"measurement", "force"}:
        return (kind, groups[0], groups[1].replace("Μ", "U"))
    if kind == "inch":
        return (kind, groups[0])
    if kind == "angle":
        return (kind, groups[0])
    if kind == "ordinal":
        return (kind, groups[0])
    return ("number", groups[0]) if kind == "list_number" else (kind, groups[0])


def _engineering_signature(text: str, source_context: str) -> tuple[tuple[str, ...], ...]:
    # 英寸中文等价依赖源含英寸符号，
    # 不凭中文“英寸”扩张源意图。
    # 结构候选仍执行最长不交叠选取，
    # 特殊签名与普通签名共同参与比较。
    normalized = _normalize_engineering_text(text)
    if re.search(r"\d\s*[\"″]", source_context):
        normalized = normalized.replace("英寸", '"')
    normalized, specials = _masked_specials(normalized, source_context)
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for kind, pattern in _STRUCTURED_PATTERNS:
        for match in pattern.finditer(normalized):
            candidates.append((match.start(), match.end(), _canonical_match(kind, match)))
    selected: list[tuple[int, int, tuple[str, ...]]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if any(candidate[0] < other[1] and other[0] < candidate[1] for other in selected):
            continue
        selected.append(candidate)
    ordered = [token for _, _, token in sorted(selected, key=lambda item: item[0])]
    # 乘号数量是多维尺寸结构的一部分，
    # X、×与数值间星号分别计数。
    # 尺寸连接用星号不重复算引用标记，
    # 其余星号仍必须逐个保留。
    # 避免提高覆盖时丢弃图纸关键标记。
    x_connectors = list(re.finditer(r"(?<![A-Z])(?:×|\bX\b)(?![A-Z])", normalized, re.I))
    star_connectors = list(re.finditer(
        r"(?<=[0-9°])\s*\*\s*(?=[+-]?(?:\d|\.))", normalized,
    ))
    connector_count = len(x_connectors) + len(star_connectors)
    if connector_count:
        ordered.append(("dimension_connectors", str(connector_count)))
    connector_star_positions = {
        position for match in star_connectors for position in range(match.start(), match.end())
    }
    marker_count = sum(
        match.start() not in connector_star_positions
        for match in re.finditer(r"\*", normalized)
    )
    if marker_count:
        ordered.append(("reference_markers", str(marker_count)))
    return tuple(specials + ordered)


def validate_engineering_semantics(
    source: str, translated: str, *, extended: bool = True,
    allow_numbering_labels: bool = False,
) -> LiteralValidation:
    """Conservatively preserve high-risk engineering markers and directions."""
    source_upper = unicodedata.normalize("NFKC", source).upper()
    target_upper = unicodedata.normalize("NFKC", translated).upper()
    warnings: list[str] = []

    # A complete numbering label uses No. as 'number', not a prohibition.
    # Longer instructions still require every existing negation check.
    # NO.可能是编号而非否定指令，
    # 但豁免只针对完整字段标签。
    # 目标也须明确含编号类名称，
    # 不能凭源单词NO就跳过否定保护。
    # allow_numbering_labels由调用者控制，
    # 不会取消长句的NO/NOT检查。
    numbering_label = bool(
        is_numbering_label(source_upper)
        and bool(re.search(r'编号|图号|零件号|材料号|物料号|工装号|张次号',translated))
        and (allow_numbering_labels or re.fullmatch(
            r"\s*(?:DIE|PART|DRAWING|TOOL)\s+NO\.\s*", source_upper,
        ))
    )

    # 该层检查关键标记是否仍有对应，
    # 不是通用中英语义等价判断。
    # 上下、远侧、仅限、全部分别保留，
    # extended再启用前后与除非检查。
    # 满足关键词不能证明动作绑定正确，
    # 因此还追加下方的窄绑定检查。
    requirements = (
        (r"(?<![A-Z])RA(?=\s*[+-]?(?:\d|\.))", (r"(?<![A-Z])RA(?=\s*[+-]?(?:\d|\.))",), "engineering_ra_marker_missing"),
        (r"\bHRC\b", (r"\bHRC\b",), "engineering_hrc_marker_missing"),
        (r"\b(?:NO|DO\s+NOT|NOT|WITHOUT|NEVER)\b", (r"不|无|勿|禁止|不得|未|不可|不能|严禁|\bNO\b|\bNOT\b|\bWITHOUT\b",), "engineering_negation_missing"),
        (r"\bTOP\b", (r"上|顶|\bTOP\b",), "engineering_top_direction_missing"),
        (r"\bBOTTOM\b", (r"下|底|\bBOTTOM\b",), "engineering_bottom_direction_missing"),
        (r"\bFAR\s+SIDE\b", (r"反面|背面|远侧|另一面|\bFAR\s+SIDE\b",), "engineering_far_side_missing"),
        (r"\bONLY\b", (r"仅|只|\bONLY\b",), "engineering_only_quantifier_missing"),
        (r"\bALL\b", (r"所有|全部|全|\bALL\b",), "engineering_all_quantifier_missing"),
    )
    if extended:
        requirements += (
            (r"\bAFTER\b", (r"后|之后|完成后|再|随后|\bAFTER\b",), "engineering_after_order_missing"),
            (r"\bBEFORE\b", (r"前|之前|\bBEFORE\b",), "engineering_before_order_missing"),
            (r"\bUNLESS\b", (r"除非|若非|未注|另有说明|\bUNLESS\b",), "engineering_unless_condition_missing"),
        )
    for source_pattern, target_patterns, warning in requirements:
        if warning == 'engineering_negation_missing' and numbering_label:
            continue
        if re.search(source_pattern, source_upper, re.I) and not any(
            re.search(pattern, target_upper, re.I) for pattern in target_patterns
        ):
            warnings.append(warning)
    warnings.extend(_action_binding_warnings(source_upper, target_upper))
    return LiteralValidation(not warnings, tuple(warnings))


def _action_binding_warnings(source: str, target: str) -> tuple[str, ...]:
    """Bind polarity only for fully parsed GRIND/POLISH TOP/BOTTOM commands.

    This bounded grammar does not claim general semantic equivalence. Noun
    statements such as NO GRIND OR PIT MARKS remain under existing gates.
    """
    from collections import Counter
    # 只解析研磨/抛光的顶底命令，
    # 动作对象和否定构成同一三元组。
    # 原句若含未支持子句则记未全覆盖，
    # 不把未知部分暗中判为正确。
    # 没有受支持命令时本守卫无覆盖。
    expected = []
    fully_parsed = True
    for clause in re.split(r"[.;]+", source):
        match = re.fullmatch(r"\s*(DO\s+NOT\s+)?(GRIND|POLISH)\s+(TOP|BOTTOM)\s*", clause)
        if match:
            expected.append((match[2], match[3], bool(match[1])))
        elif clause.strip():
            fully_parsed = False
    if not expected:
        return ()
    # 中文每个子句只能绑定单动作单对象，
    # 多个候选会导致绑定无法核实。
    # 不能让“不”从一条加工句迁移，
    # 也不能把顶面与底面互相交换。
    # 只计被有限语法完整消费的子句。
    actual = []
    for clause in re.split(r"[。.;；,，]+", target):
        actions = [name for name, pattern in (("GRIND", r"研磨|磨削|\bGRIND\b"),
                   ("POLISH", r"抛光|\bPOLISH\b")) for _ in re.finditer(pattern, clause)]
        objects = [name for name, pattern in (("TOP", r"顶部|顶面|上表面|\bTOP\b"),
                   ("BOTTOM", r"底部|底面|下表面|\bBOTTOM\b")) if re.search(pattern, clause)]
        if fully_parsed and clause.strip() and not (actions and objects):
            return ("engineering_action_binding_unverified",)
        if actions and objects:
            if len(actions) != 1 or len(objects) != 1:
                return ("engineering_action_binding_unverified",)
            compact = re.sub(r"\s+", "", clause)
            neg = r"(?:不要|不得|禁止|严禁|勿|不能|不可|不允许|不许|不)?"
            action = r"(?:研磨|磨削|抛光)"
            obj = r"(?:顶部|顶面|上表面|底部|底面|下表面)"
            if not (re.fullmatch(rf"{neg}{action}{obj}|{obj}{neg}{action}", compact)
                    or re.fullmatch(r"\s*(?:DO\s+NOT\s+)?(?:GRIND|POLISH)\s+(?:TOP|BOTTOM)\s*", clause)):
                return ("engineering_action_binding_unverified",)
            negative = bool(re.search(r"不要|不得|不许|不可|禁止|勿|未|不|\bDO\s+NOT\b", clause))
            actual.append((actions[0], objects[0], negative))
    # 多重集保留重复命令的次数，
    # 原文完整时要求目标完全相等。
    # 原文部分覆盖时只比同角色范围，
    # 防止未支持动作干扰窄检查。
    # 返回空元组仍不代表整句通过。
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    if fully_parsed and actual_counts != expected_counts:
        return ("engineering_action_negation_binding_changed",)
    scoped_actual = Counter({item: count for item, count in actual_counts.items()
                            if any(item[:2] == e[:2] for e in expected)})
    if scoped_actual != expected_counts:
        return ("engineering_action_negation_binding_changed",)
    return ()


# 限定字段名可省略NO后的句点，
# 其它开放标签仍要求明确NO.。
# 整串fullmatch限制豁免范围，
# 句内“NO GRIND”等禁止语仍受保护。
def is_numbering_label(source: str) -> bool:
    """Return whether a complete field label uses ``NO.`` as number abbreviation."""
    source_upper = unicodedata.normalize("NFKC", source).upper()
    # Only complete, named fields may omit the abbreviation period. Applying
    # this inside a sentence would suppress real NO/NOT/WITHOUT instructions.
    if re.fullmatch(r"\s*(?:DIE|PART|DRAWING|DWG|MATERIAL|TOOL)\s+NO\.?\s*", source_upper):
        return True
    return bool(re.fullmatch(
        r"\s*(?:[A-Z][A-Z0-9 .,/&'()-]{0,48}\s+|[A-Z]/)?NO\.\s*",
        source_upper,
    ))
