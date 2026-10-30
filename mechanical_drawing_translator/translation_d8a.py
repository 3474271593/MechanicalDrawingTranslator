# mechanical_drawing_translator/translation_d8a.py
"""Conservative page-context translation probes for the D8-A trial."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .translation_targets import TargetEntry


@dataclass(frozen=True)
# 页内语境仅提出候选译法，
# mechanism解释使用的语境规则。
# evidence是附近字段类别等旁证，
# 不能冒充人工逐句审核或真实译文批准。
class D8AResolution:
    translated_text: str
    mechanism: str
    evidence: tuple[str, ...]


# 字段语法消费完整英文标签，
# DRAWN/CHK/APP等仅支持所列缩写。
# 不会把标签后人名或编号截掉，
# 也不进行任意缩写词典替换。
# 新中文仍由服务完整工程校验。
_FIELD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"DR(?:A)?WN\s+BY\s*:?", re.I), "绘图人："),
    (re.compile(r"CHK(?:ECKED)?\s+BY\s*:?", re.I), "审核人："),
    (re.compile(r"APP(?:R|ROVED)?\s+BY\s*:?", re.I), "批准人："),
    (re.compile(r"REL\.?\s+DATE\s*:?", re.I), "发布日期："),
    (re.compile(r"HEAT\s+TREAT(?:MENT)?\s*:?", re.I), "热处理"),
    (re.compile(r"DESIGNED\s+IN\s+METRIC(?:\s+UNITS)?", re.I), "按公制单位设计"),
    (re.compile(r"TOOL\s+I\.?\s*D\.?\s*:?", re.I), "工具编号"),
    (re.compile(r"TITLE\s*:?", re.I), "标题："),
    (re.compile(r"DET\.?\s*:?", re.I), "详图"),
    (re.compile(r"REV[.,]?\s*:?", re.I), "修订"),
)

# 锚点按字段家族分组而非按数量计，
# 多个同类日期不能代替多种图签证据。
# 姓名、材料、修订、编号等互补，
# 至少三类才建立密集图签语境。
_ANCHOR_PATTERNS = (
    re.compile(r"(?:DR(?:A)?WN|CHK(?:ECKED)?|APP(?:R|ROVED)?)\s+BY\s*:?", re.I),
    re.compile(r"(?:TITLE|NAME|PART\s+NAME)\s*:?", re.I),
    re.compile(r"(?:MATERIAL|MAT'L|HEAT\s+TREAT(?:MENT)?|HARDNESS|FINISH)\s*:?", re.I),
    re.compile(r"(?:REV[.,]?|REL\.?\s+DATE)\s*:?", re.I),
    re.compile(r"(?:DOCUMENT|MATERIAL|SERIES|DESIGN\s+FILE|SEC\s*,?\s*SHT\.?)\s+NO\.", re.I),
    re.compile(r"(?:SCALE|UNITS?|TOOL\s+I\.?\s*D\.?)\s*:?", re.I),
)

_NAME_LABEL = re.compile(r"(?:^|\b)(?:TITLE|NAME|PART\s+NAME)\s*:?$", re.I)
_PART_MODIFIERS = {
    "CUT": "切边",
    "CUTTING": "切边",
    "FORM": "成形",
    "TRIM": "修边",
}
_PART_HEADS = {"PUNCH": "冲头", "INSERT": "镶件", "BLOCK": "块"}


def _normalized(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _height(entry: TargetEntry) -> float:
    return max(entry.bounds.max_y - entry.bounds.min_y, 1.0)


def _center(entry: TargetEntry) -> tuple[float, float]:
    bounds = entry.bounds
    return ((bounds.min_x + bounds.max_x) / 2, (bounds.min_y + bounds.max_y) / 2)


# 附近范围按页面典型字高缩放，
# 不硬编码某张PDF的绝对坐标。
# 横纵阈值不同体现图签几何，
# 仅说明空间邻近而非语义关联已证实。
def _nearby(entry: TargetEntry, other: TargetEntry, typical_height: float) -> bool:
    x, y = _center(entry)
    ox, oy = _center(other)
    return abs(y - oy) <= 18 * typical_height and abs(x - ox) <= 65 * typical_height


# 核验字面使用需要调用者显式开启，
# 默认始终使用不可变原OCR英文。
# 源模式变化必须重新准备语境，
# 不能复用另一请求的缓存标签。
def _source(entry: TargetEntry, use_verified_source: bool) -> str:
    return entry.translation_source_text if use_verified_source else entry.source_text


def _anchor_kinds(entry: TargetEntry, use_verified_source: bool = False) -> frozenset[int]:
    text = _normalized(_source(entry, use_verified_source))
    return frozenset(index for index, pattern in enumerate(_ANCHOR_PATTERNS)
                     if pattern.fullmatch(text))


def _title_block_context(entry: TargetEntry, entries: Sequence[TargetEntry],
                         use_verified_source: bool = False) -> tuple[int, ...]:
    # 典型字高取全部条目的中位数，
    # 降低单个大标题对邻域的影响。
    # 空页面退回当前条目的字高，
    # 高度最小一单位避免退化阈值。
    heights = [_height(item) for item in entries]
    typical_height = median(heights) if heights else _height(entry)
    kinds: set[int] = set()
    for item in entries:
        if _nearby(entry, item, typical_height):
            kinds.update(_anchor_kinds(item, use_verified_source))
    return tuple(sorted(kinds))


def _has_adjacent_name_label(entry: TargetEntry, entries: Sequence[TargetEntry],
                             use_verified_source: bool = False) -> bool:
    # 零件名组合还要更近的NAME/TITLE标签，
    # 不只依赖宽泛图签密度。
    # 相同短语在正文中可能是工艺而非名称，
    # 因此不能缺少局部命名字段证据。
    typical_height = median([_height(item) for item in entries]) if entries else _height(entry)
    x, y = _center(entry)
    for item in entries:
        if not _NAME_LABEL.fullmatch(_normalized(_source(item, use_verified_source))):
            continue
        ox, oy = _center(item)
        if abs(y - oy) <= 2.5 * typical_height and abs(x - ox) <= 20 * typical_height:
            return True
    return False


def resolve_contextual_translation(
    entry: TargetEntry, entries: Sequence[TargetEntry],
    *, use_verified_source: bool = False,
) -> D8AResolution | None:
    """Resolve only complete expressions backed by a dense nearby title block."""
    # 调用方负责entries属于同一页，
    # 该函数不会自行打开或分组PDF。
    # 三种锚点不足明确返回None，
    # 保留基础词表与模型路线的原行为。
    anchors = _title_block_context(entry, entries, use_verified_source)
    if len(anchors) < 3:
        return None

    source = _normalized(_source(entry, use_verified_source))
    for pattern, target in _FIELD_PATTERNS:
        if pattern.fullmatch(source):
            return D8AResolution(
                target, "title_block_field",
                (f"nearby_title_block_anchor_kinds={','.join(map(str, anchors))}",),
            )

    # 零件名仅限定一个已知修饰词，
    # 加PUNCH/INSERT/BLOCK三个部件头。
    # 不识别的修饰词不截掉后仅翻译名词，
    # 也不据中文参考推断OCR漏词。
    part = re.fullmatch(r"([A-Z]+)\s+(PUNCH|INSERT|BLOCK)", source)
    if part and _has_adjacent_name_label(entry, entries, use_verified_source):
        modifier = _PART_MODIFIERS.get(part.group(1))
        head = _PART_HEADS[part.group(2)]
        if modifier is not None:
            return D8AResolution(
                modifier + head, "contextual_part_name_composition",
                ("adjacent_name_or_title_label", f"modifier={part.group(1)}", f"head={part.group(2)}"),
            )
    return None


@dataclass(frozen=True)
# 请求内准备源串、中心与字高，
# 锚点及命名标签也只算一次。
# 按输入位置保存而非region ID缓存，
# 同ID重载或换源模式不继承旧语境。
class _PreparedEntry:
    source: str
    x: float
    y: float
    height: float
    anchor_kinds: frozenset[int]
    name_label: bool


def resolve_page_context(
    entries: Sequence[TargetEntry], *, use_verified_source: bool = False,
) -> tuple[D8AResolution | None, ...]:
    """Resolve a supplied page with request-local geometry and text preparation.

    As in the legacy resolver, callers own the document/page scope of ``entries``.
    Prepared values are positional and never retained across requests, so repeated
    region IDs, file reloads, and source-mode switches cannot reuse stale context.
    """
    if not entries:
        return ()

    # 批量入口复用同一页的准备结果，
    # 算法含义保持旧逐项接口一致。
    # 没有跨页或跨请求持久索引，
    # 减少重复规范化而不扩大语境范围。
    prepared = []
    for entry in entries:
        bounds = entry.bounds
        source = _normalized(_source(entry, use_verified_source))
        prepared.append(_PreparedEntry(
            source,
            (bounds.min_x + bounds.max_x) / 2,
            (bounds.min_y + bounds.max_y) / 2,
            max(bounds.max_y - bounds.min_y, 1.0),
            frozenset(index for index, pattern in enumerate(_ANCHOR_PATTERNS)
                      if pattern.fullmatch(source)),
            bool(_NAME_LABEL.fullmatch(source)),
        ))

    typical_height = median([item.height for item in prepared])
    # 后续只遍历有锚点或名称标签的条目，
    # 避免对每项重新匹配所有正则。
    # 每个结果仍对应原输入位置，
    # 无候选时明确append(None)。
    # 优化不改变三类锚点及邻距门槛。
    anchor_entries = tuple(item for item in prepared if item.anchor_kinds)
    name_entries = tuple(item for item in prepared if item.name_label)
    results: list[D8AResolution | None] = []
    for entry in prepared:
        kinds: set[int] = set()
        for item in anchor_entries:
            if (abs(entry.y - item.y) <= 18 * typical_height
                    and abs(entry.x - item.x) <= 65 * typical_height):
                kinds.update(item.anchor_kinds)
        if len(kinds) < 3:
            results.append(None)
            continue

        resolution = None
        for pattern, target in _FIELD_PATTERNS:
            if pattern.fullmatch(entry.source):
                resolution = D8AResolution(
                    target, "title_block_field",
                    (f"nearby_title_block_anchor_kinds={','.join(map(str, sorted(kinds)))}",),
                )
                break

        if resolution is None:
            part = re.fullmatch(r"([A-Z]+)\s+(PUNCH|INSERT|BLOCK)", entry.source)
            if part and any(
                abs(entry.y - item.y) <= 2.5 * typical_height
                and abs(entry.x - item.x) <= 20 * typical_height
                for item in name_entries
            ):
                modifier = _PART_MODIFIERS.get(part.group(1))
                if modifier is not None:
                    resolution = D8AResolution(
                        modifier + _PART_HEADS[part.group(2)],
                        "contextual_part_name_composition",
                        ("adjacent_name_or_title_label", f"modifier={part.group(1)}",
                         f"head={part.group(2)}"),
                    )
        results.append(resolution)
    return tuple(results)
