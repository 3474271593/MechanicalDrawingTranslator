# mechanical_drawing_translator/translation_d8c.py
"""D8-C recovery of split, already-reviewed processing instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .translation_glossary import Glossary
from .translation_targets import TargetEntry


# 只恢复明确以TO FIT结束的加工碎片，
# 不对所有短句尝试任意相邻拼接。
# source_text保持原OCR字面，
# 新完整组只改变关联而不覆盖原区域。
_INCOMPLETE_PROCESS = re.compile(
    r"(?:GRIND|MACHINE|POLISH|CUT|DRILL)\s+TO\s+FIT", re.I,
)


@dataclass(frozen=True)
# 证据记录原动作组与对象组身份，
# 以及拼接英文和已有词表译文。
# glossary_entry_id说明闭合依据，
# 不是从中文参考反推应拼哪两行。
class D8CSemanticEvidence:
    action_group_id: str
    object_group_id: str
    joined_source: str
    reviewed_translation: str
    glossary_entry_id: str
    mechanism: str = "reviewed_phrase_boundary_recovery"


def _height(entry: TargetEntry) -> float:
    return max(1.0, entry.bounds.max_y - entry.bounds.min_y)


def _uses_whole_regions(entry: TargetEntry) -> bool:
    if not entry.spans:
        return True
    # 有spans时必须每区域恰好完整覆盖，
    # 部分共享OCR框不得整体借入。
    # 没有spans沿用完整原区语义，
    # 不伪造缺失字符范围。
    by_id = {region.id: region for region in entry.regions}
    return len(entry.spans) == len(entry.regions) and all(
        span.start == 0 and span.end == len(by_id[span.region_id].source_text)
        for span in entry.spans
    )


def _is_spatial_continuation(action: TargetEntry, candidate: TargetEntry) -> bool:
    # 同文档同页是几何合并前置，
    # 不同OCR来源不互相补句。
    # 旋转超过15度或相对角差过大拒绝，
    # 水平框关系不能推广到竖排阅读顺序。
    if action.document_id != candidate.document_id or action.page_index != candidate.page_index:
        return False
    regions = (*action.regions, *candidate.regions)
    # This geometry models horizontal lines followed from top to bottom. Equal
    # rotated angles do not prove the same reading order, so reject unsupported
    # directions rather than applying horizontal bounds to vertical text.
    if (len({region.source for region in regions}) != 1
            or any(abs(region.rotation_degrees) > 15 for region in regions)
            or max(region.rotation_degrees for region in regions)
               - min(region.rotation_degrees for region in regions) > 8):
        return False
    if (not _uses_whole_regions(action) or not _uses_whole_regions(candidate)
            or set(action.region_ids) & set(candidate.region_ids)):
        return False
    # 候选与动作不能共享原region ID，
    # 宽度、左对齐与垂距均按字高限定。
    # 允许轻微框交叠适配检测边界，
    # 但几何邻近本身仍不是完整句证据。
    height = max(_height(action), _height(candidate))
    vertical_gap = candidate.bounds.min_y - action.bounds.max_y
    left_delta = abs(candidate.bounds.min_x - action.bounds.min_x)
    width_delta = abs(
        (candidate.bounds.max_x - candidate.bounds.min_x)
        - (action.bounds.max_x - action.bounds.min_x)
    )
    return -0.15 * height <= vertical_gap <= 0.8 * height \
        and left_delta <= 1.0 * height and width_delta <= 1.5 * height


def merge_reviewed_instructions(
    entries: Sequence[TargetEntry], glossary: Glossary,
) -> tuple[tuple[TargetEntry, ...], tuple[D8CSemanticEvidence, ...]]:
    """Merge a split instruction only when immutable reviewed knowledge closes it.

    Geometry proposes a continuation; an exact safe glossary hit is the evidence
    that the combined English is a complete known instruction.  Ambiguous or
    unknown continuations remain as their original entries.
    """
    # 输入先冻结为元组保持稳定顺序，
    # consumed防止同一个对象被多次合并。
    # 只遍历未消费的明确残缺加工动作，
    # 未知行不被自动当作宾语。
    values = tuple(entries)
    consumed: set[int] = set()
    replacements: dict[int, TargetEntry] = {}
    evidence: list[D8CSemanticEvidence] = []
    for index, action in enumerate(values):
        if index in consumed or _INCOMPLETE_PROCESS.fullmatch(action.source_text.strip()) is None:
            continue
        matches = []
        for other_index, candidate in enumerate(values):
            if other_index == index or other_index in consumed:
                continue
            if not _is_spatial_continuation(action, candidate):
                continue
            # 几何合格后再以完整拼接英文查词表，
            # 只接受已有exact_rule闭合。
            # 参数规则或模型近义不授予句界恢复，
            # 确保合并不是用新中文猜源关系。
            joined = f"{action.source_text.strip()} {candidate.source_text.strip()}"
            resolution = glossary.resolve(joined)
            if resolution is not None and resolution.route == "exact_rule":
                matches.append((other_index, candidate, joined, resolution))
        # 零个或多个可行对象都保持原条目，
        # 不按距离最近或遍历首项裁决歧义。
        # 新组包含原始两个区域并保留来源，
        # 分类不一致则标review_required。
        if len(matches) != 1:
            continue
        other_index, candidate, joined, resolution = matches[0]
        merged = TargetEntry(
            (*action.regions, *candidate.regions),
            action.category if action.category == candidate.category else "review_required",
            "D8-C：相邻动作与对象恢复为既有审核完整短语",
            paragraph_id=f"d8c-{action.group_id[:12]}",
            boundary_notes=(
                "d8c_reviewed_continuation",
                f"d8c_glossary_entry:{resolution.entry_id}",
            ),
        )
        # 替换写在原动作的位置，
        # 仅消费已采用的另一个对象条目。
        # 边界notes携带完整词条ID，
        # 后续服务附审核短语恢复警告。
        # 最终一组输出一次并保留完整证据。
        replacements[index] = merged
        consumed.add(other_index)
        evidence.append(D8CSemanticEvidence(
            action.group_id, candidate.group_id, joined,
            resolution.translated_text, resolution.entry_id,
        ))
    output = tuple(
        replacements.get(index, item)
        for index, item in enumerate(values)
        if index not in consumed
    )
    return output, tuple(evidence)
