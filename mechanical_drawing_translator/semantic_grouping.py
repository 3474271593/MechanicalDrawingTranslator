# mechanical_drawing_translator/semantic_grouping.py
"""Conservative offline sentence recovery over immutable OCR character spans."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .translation_targets import TargetEntry


SEMANTIC_VERSION = "ocr-3b-v1"


# 区间采用Python字符索引
# start含首字符
# end不含末字符
# 不是UTF-8字节偏移
# 不是字符级图像坐标
# region_id定位原始区域
# 多个句段可以共享同一框
# 不能据此局部擦除原文
@dataclass(frozen=True)
class SourceSpan:
    """Half-open Python character offsets in one original TextRegion."""

    region_id: str
    start: int
    end: int


# These are instruction forms, not drawing-specific sentence lookups.
_START = re.compile(
    r"^(?:\d+[.)]\s+)?(?:NOTE\s*[:)]|ARROWS\s+INDICATE\b|"
    r"DO\s+NOT\b|NO\s+(?:GRIND|PIT|BURRS?|SHARP)\b|"
    r"(?:SEE|PROVIDE|POLISH|GRIND|DRILL|REAM|REMOVE|APPLY|DEBURR|"
    r"CHAMFER|MACHINE|HEAT\s+TREAT|INSPECT|ENSURE|KEEP|USE(?=\s+[A-Z])|INSTALL|"
    r"ASSEMBLE|CHECK|FINISH|MARK|HOLD|ALLOW)\b)", re.IGNORECASE,
)
_CONTINUATION_END = frozenset({
    "TO", "OF", "FOR", "ON", "IN", "AND", "OR", "WITH", "WITHOUT",
    "BEFORE", "AFTER", "UNLESS", "DO", "NOT", "NO", "AS", "BE", "MUST",
    "SHALL", "MAY", "BY", "THE", "THAT", "WHEN", "IF", "UNTIL", "A",
    "AN", "AT", "PER", "THAN", "IS", "ARE", "FOLLOWING", "SEE",
})
_ABBREVIATION = re.compile(
    r"(?:(?:[A-Z]\.){2,}|(?:REF|REV|DIA|TYP|MAX|MIN|NO|DR|APP|CHK|QTY)\.)$",
    re.IGNORECASE,
)
# A subject plus modal is evidence only after explicit sentence punctuation.
# It must not broaden the dash/line-start rules that keep conditions attached.
_SUBJECT_MODAL = re.compile(
    r"^(?:\d+[.)]\s+)?(?!(?:BEFORE|AFTER|UNLESS|IF|WHEN|UNTIL|AND|OR)\b)"
    r"(?:[A-Za-z][A-Za-z'\-]*\s+){1,6}(?:MUST|SHALL|SHOULD)\s+\S", re.I,
)
_LEADING_CONDITION = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?:NOTE\s*[:)]\s*)?"
    r"(?:BEFORE|AFTER|UNLESS|IF|WHEN|UNTIL)\b", re.I,
)


# 仅识别主语加情态动词
# 先去掉编号前缀
# 首个子句也需完整
# 条件开头不能冒充新句
def _subject_modal_start(text: str) -> bool:
    body = re.sub(r"^\d+[.)]\s+", "", text)
    clause = re.split(r"[.。!?;；](?=\s|$)", body, maxsplit=1)[0]
    return bool(_SUBJECT_MODAL.match(text) and _complete_clause(clause))


# 延续基础完整性判断
# THEN结尾仍需后续动作
# 前置条件不能单独成句
# 逗号之后须有明确谓词
# 无标点条件范围保守保持
# 避免条件从指令主体脱落
def _complete_integrated_clause(text: str) -> bool:
    if not _complete_clause(text):
        return False
    if re.search(r"\bTHEN$", text.rstrip(" \t\r\n.。!?;；:"), re.I):
        return False
    if not _LEADING_CONDITION.match(text):
        return True
    # A punctuated condition alone is still not an independent instruction.
    # Require an explicit action/predicate after a comma before introducing a
    # new modal sentence; uncertain unpunctuated condition scopes stay whole.
    body = text.split(",", 1)[1].lstrip() if "," in text else ""
    return bool(_START.match(body) or _subject_modal_start(body))


# 前置条件未结束不得截断
# 至少两个英文词才考虑完整
# 介词/情态词结尾继续附着
# 删尾标点只为检查末词
# 不会修改返回的原文片段
def _complete_clause(text: str) -> bool:
    if (re.match(r"^\s*(?:\d+[.)]\s*)?(?:NOTE\s*[:)]\s*)?"
                 r"(?:BEFORE|AFTER|UNLESS|IF|WHEN|UNTIL)\b", text, re.IGNORECASE)
            and not text.rstrip().endswith((".", "。", ";", "!", "?"))):
        return False  # A leading condition must stay attached to its governing action.
    words = re.findall(r"[A-Za-z]+", text)
    tail = re.search(r"([A-Za-z]+)$", text.rstrip(" \t\r\n.。!?;；:)-–—"))
    return len(words) >= 2 and not (
        tail and tail.group(1).upper() in _CONTINUATION_END
    )


# 短OCR行不因接近就合并
# 左尾或右首需明确语法连接
# 已有句终标点阻止续接
# 否定和条件词保持作用范围
# 只返回分组证据不改字面
def has_explicit_continuation(left: str, right: str) -> bool:
    """Allow a short OCR line to attach only with a clear grammatical link."""
    words = re.findall(r"[A-Za-z]+", left)
    tail = re.search(r"([A-Za-z]+)$", left.rstrip(" \t\r\n:)-–—"))
    return bool(words and not left.rstrip().endswith((".", "。", ";")) and (
        (tail and tail.group(1).upper() in _CONTINUATION_END)
        or re.match(r"^\s*(?:ARE|IS|BE|TO|OR|AND|WITH|WITHOUT|BEFORE|AFTER|UNLESS|WHEN)\b",
                    right, re.IGNORECASE)
    ))


# 冒号字段通常不是正文
# NOTE/NOTES排除于字段判断
# 防止图签邻格误并成句
def is_metadata_field(text: str) -> bool:
    value = text.strip()
    return bool(value.upper() not in {"NOTE:", "NOTES:"} and re.fullmatch(
        r"(?:[A-Za-z.]+\s+){0,2}[A-Za-z.]+:", value,
    ))


# 切点始终引用段落原字面
# 零位置代表段首
# 句终标点后要有独立指令
# 不能仅遇句号就切分
# 编号前缀的句点保留
# 常用缩写末点不算句界
# 紧连字符通常属于复合词
# 受控否定/SEE才有窄例外
# 分隔破折号留在左跨度
# 行首还需前段完整
# integrated额外核查条件范围
# recovered仅放宽前导星号识别
# 不删除图纸标记
# 返回原因用于后续审计
def _boundary_offsets(text: str, line_starts: Sequence[int], *, recovered: bool = False,
                      enable_integrated_quality: bool = False) -> dict[int, str]:
    start_pattern=(re.compile(r"^(?:\*+\s*)?"+_START.pattern[1:],re.IGNORECASE)
                   if recovered else _START)
    cuts: dict[int, str] = {0: "paragraph_start"}
    for match in re.finditer(r"[.。!?;；]+(?=\s|$)", text):
        end = match.end()
        left = text[:end]
        tail = text[end:].lstrip()
        known_start = bool(start_pattern.match(tail))
        integrated_start = enable_integrated_quality and _subject_modal_start(tail)
        if not tail or not (known_start or integrated_start) or not _complete_clause(left):
            continue
        if integrated_start and not known_start:
            prior = max(cuts)
            if not _complete_integrated_clause(text[prior:end]):
                continue
        token = left.split()[-1]
        if re.fullmatch(r"\d+\.", token):
            continue  # Numbered instruction prefix belongs to its following body.
        if match.group() == "." and _ABBREVIATION.search(token):
            continue
        cuts[end] = "sentence_punctuation"
    for match in re.finditer(r"[-–—]\s*(?=[A-Za-z])", text):
        tail = text[match.end():]
        tight_hyphen = (match.start() > 0 and not text[match.start() - 1].isspace()
                        and match.end() == match.start() + 1)
        if tight_hyphen and not re.match(
            r"^(?:NO\s+|DO\s+NOT\s+|SEE\s+|NOTE\s*[:)])", tail, re.IGNORECASE,
        ):
            continue  # RE-GRIND, FINISH-MACHINE and drawing IDs are not separators.
        prior = max(k for k in cuts if k <= match.start())
        if start_pattern.match(tail) and _complete_clause(text[prior:match.start()]):
            # Keep the source separator in the left span; never silently delete it.
            cuts[match.end()] = "separator_before_instruction"
    for offset in line_starts:
        prior = max(k for k in cuts if k < offset) if offset else 0
        complete = (_complete_integrated_clause(text[prior:offset])
                    if enable_integrated_quality else _complete_clause(text[prior:offset]))
        if offset and start_pattern.match(text[offset:]) and complete:
            cuts[offset] = "independent_instruction_line"
    return cuts


# 只划分已有几何段落
# 人工判定条目原样保留
# 区域间插空格构造段落
# 同时记录每区域起止位置
# 原生词边界不等于换行
# integrated用实际向下移动判行
# 近水平且行差足够才采纳
# 边界算法给出段落切点
# 每片反算到原区域半开区间
# 只裁掉片段边缘空白
# 不推测逐字符几何
# 共享框仍保存原Region
# paragraph_id保留父段身份
# 新group_id由跨度重新计算
# 冒号结尾提示上下文复核
# 低置信与碎字增加警告
# OCR原文及原坐标均不改写
# 旧段译文不能直接套给新句
def recover_semantic_entries(
    entries: Sequence[TargetEntry], *, enable_integrated_quality: bool = False,
) -> tuple[TargetEntry, ...]:
    """Partition geometric paragraphs; raw regions and their coordinates are untouched."""
    output: list[TargetEntry] = []
    for entry in entries:
        if entry.reason == "人工判定":
            output.append(entry)
            continue
        paragraph = ""
        locations = []
        for region in entry.regions:
            if paragraph:
                paragraph += " "
            start = len(paragraph)
            paragraph += region.source_text
            locations.append((region, start, len(paragraph)))
        line_starts = [start for _, start, _ in locations]
        if enable_integrated_quality:
            # Native PDF word boundaries are not visual line boundaries. Keep
            # the existing multiline baseline threshold, and require actual
            # downward movement rather than treating every source span as a line.
            line_starts = []
            for (previous, _, _), (current, start, _) in zip(locations, locations[1:]):
                height = min(max(1., r.bounds.max_y - r.bounds.min_y)
                             for r in (previous, current))
                line_step = (current.bounds.min_y + current.bounds.max_y
                             - previous.bounds.min_y - previous.bounds.max_y) / 2
                if (all(abs(r.rotation_degrees) <= 15 for r in (previous, current))
                        and line_step >= .45 * height):
                    line_starts.append(start)
        cuts = _boundary_offsets(
            paragraph, line_starts,
            recovered=any(r.status.startswith("局部复识别") for r in entry.regions),
            enable_integrated_quality=enable_integrated_quality,
        )
        offsets = sorted({*cuts, len(paragraph)})
        for start, end in zip(offsets, offsets[1:]):
            spans = []
            members = []
            for region, region_start, region_end in locations:
                a, b = max(start, region_start), min(end, region_end)
                if a >= b or not paragraph[a:b].strip():
                    continue
                while a < b and paragraph[a].isspace():
                    a += 1
                while b > a and paragraph[b - 1].isspace():
                    b -= 1
                spans.append(SourceSpan(region.id, a - region_start, b - region_start))
                members.append(region)
            if not spans:
                continue
            notes = [cuts.get(start, "sentence_boundary")]
            text = paragraph[start:end].strip()
            if text.endswith(":"):
                notes.append("following_context_requires_review")
            words = re.findall(r"[A-Za-z]+", text)
            if (any(r.confidence is not None and r.confidence < 0.8 for r in members)
                    or (words and all(len(word) == 1 for word in words))):
                notes.append("source_recognition_requires_review")
            output.append(replace(
                entry, regions=tuple(members), spans=tuple(spans),
                paragraph_id=entry.group_id, boundary_notes=tuple(notes),
            ))
    return tuple(output)
