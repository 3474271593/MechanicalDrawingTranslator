# mechanical_drawing_translator/translation_targets.py
"""独立于 OCR 和 Qt 的机械图纸英文翻译目标筛选与人工判定。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from .models import TextBounds, TextRegion
from .semantic_grouping import (
    SourceSpan, has_explicit_continuation, is_metadata_field, recover_semantic_entries,
)


Category = Literal["translatable", "non_translatable", "review_required"]
CATEGORIES: tuple[Category, ...] = (
    "translatable", "non_translatable", "review_required",
)


@dataclass(frozen=True)
class Classification:
    category: Category
    reason: str


# 一个条目可引用多个原始区域
# 原Region对象不被分组改写
# spans记录精确字面片段
# paragraph_id连接父段
# boundary_notes保留句界依据
# semantic_units保留展示聚合关系
# 整段译文不能冒充逐句批准
# 类别和原因是筛选旁路结果
@dataclass(frozen=True)
class TargetEntry:
    """一个可显示目标；多区域目标始终引用原始 OCR 对象。"""

    regions: tuple[TextRegion, ...]
    category: Category
    reason: str
    spans: tuple[SourceSpan, ...] = ()
    paragraph_id: str = ""
    boundary_notes: tuple[str, ...] = ()
    # Display aggregation may retain original semantic inputs without granting
    # their individual translations the authority of a reviewed whole paragraph.
    semantic_units: tuple[TargetEntry, ...] = ()

    # 每个跨度必须找到原区域
    # 起止范围按Python字符数验证
    # 空跨度不合法
    # 不能越过原字面末尾
    # 几何仍是共享区域范围
    def __post_init__(self) -> None:
        if self.spans:
            by_id = {r.id: r for r in self.regions}
            for span in self.spans:
                if (span.region_id not in by_id or not 0 <= span.start < span.end
                        <= len(by_id[span.region_id].source_text)):
                    raise ValueError("语义字符区间与原始 OCR 区域不一致")

    # 顺序对应原始区域引用
    # 组装结果时用来校验身份
    # 不把随机ID当审核实例ID
    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(region.id for region in self.regions)

    # 身份由统一结果模块生成
    # 包含原文与跨度等契约
    # 不能只以包围框代替分组ID
    # 边界变更必须获得新身份
    @property
    def group_id(self) -> str:
        from .translation_results import translation_group_id

        return translation_group_id(self)

    # 有跨度时只取声明字符段
    # 按跨度顺序连接原始字面
    # 无跨度则连接完整原区域
    # 不使用verified文本覆盖存档
    # 用于结果校验和审计追溯
    @property
    def source_text(self) -> str:
        if self.spans:
            by_id = {region.id: region for region in self.regions}
            return " ".join(by_id[s.region_id].source_text[s.start:s.end]
                            for s in self.spans)
        return " ".join(region.source_text.strip() for region in self.regions)

    # 仅送译输入可用图像核验字面
    # 原始source_text仍保持不变
    # 完整跨度才允许替换核验文本
    # 部分跨度不能猜核验字符映射
    # 否则继续使用原OCR切片
    # 保证SourceSpan仍指向原字面
    # 核验文本不会伪装原生来源
    @property
    def translation_source_text(self) -> str:
        """Use image-verified text only as translation input, preserving OCR source."""
        if self.spans:
            by_id = {region.id: region for region in self.regions}
            values = []
            for span in self.spans:
                region = by_id[span.region_id]
                if (span.start == 0 and span.end == len(region.source_text)
                        and region.verified_source_text):
                    values.append(region.verified_source_text)
                else:
                    values.append(region.source_text[span.start:span.end])
            return " ".join(values)
        return " ".join(
            (region.verified_source_text or region.source_text).strip()
            for region in self.regions
        )

    # 部分跨度没有逐字符坐标
    # 明确标shared_region_bounds
    # 完整区域标source_region_bounds
    # 两者都不是字形级擦除证据
    @property
    def coordinate_precision(self) -> str:
        if self.spans:
            by_id = {r.id: r for r in self.regions}
            if any(s.start != 0 or s.end != len(by_id[s.region_id].source_text)
                   for s in self.spans):
                return "shared_region_bounds"
        return "source_region_bounds"

    @property
    def page_index(self) -> int:
        return self.regions[0].page_index

    @property
    def document_id(self) -> str:
        return self.regions[0].document_id

    # 按首次出现顺序汇总来源
    # 相同来源只显示一次
    # 混合来源不会被压成纯原生
    @property
    def source(self) -> str:
        return "/".join(dict.fromkeys(region.source for region in self.regions))

    # 组置信度取已知最小值
    # 原生区域的None不冒充满分
    # 全部未知则仍返回None
    # 不能用平均值掩盖低置信碎片
    @property
    def confidence(self) -> float | None:
        values = [region.confidence for region in self.regions
                  if region.confidence is not None]
        return min(values) if values else None

    # 联合边界用于定位整个组
    # 保留全部原始区域范围
    # 不按字符跨度伪造更小框
    # 旋转方向仍来自原区域证据
    @property
    def bounds(self) -> TextBounds:
        return TextBounds(
            min(region.bounds.min_x for region in self.regions),
            min(region.bounds.min_y for region in self.regions),
            max(region.bounds.max_x for region in self.regions),
            max(region.bounds.max_y for region in self.regions),
        )


# raw_regions保持本次有效输入
# decisions以原区域ID计数
# candidate_entries可能是多区组
# all_entries供全部OCR检查
# counts分母是区域不是语句
# 界面数量不可当翻译准确率
@dataclass(frozen=True)
class SelectionResult:
    raw_regions: tuple[TextRegion, ...]
    decisions: Mapping[str, Classification]
    candidate_entries: tuple[TargetEntry, ...]
    all_entries: tuple[TargetEntry, ...]
    counts: Mapping[Category, int]

    @property
    def category_by_region(self) -> dict[str, Category]:
        return {key: decision.category for key, decision in self.decisions.items()}


# 仅将已知的机械术语和说明用语自动送往翻译；未知词保留给人工判断。
TECHNICAL_WORDS = frozenset({
    "ASSEMBLY", "BELOW", "BLOCK", "CHAMFER", "CLEARANCE", "COATING",
    "COINING", "CUTTING", "DATA", "DEEP", "DETAIL", "DIE", "DIMENSION", "DIMENSIONS",
    "DRAFT", "DRAWING", "FABRICATE", "FINISH", "FINISHING", "FORM",
    "GENERAL", "GEOMETRY", "GRIND", "GROUND", "HARDNESS", "HEAT", "HELD",
    "INSERT", "LENGTH", "MARK", "MARKED", "MARKS", "MATERIAL", "MODEL", "MUST", "NOTE",
    "NOTATION", "PART", "PERMISSIBLE", "PIN", "PIT", "POLISH", "POLISHED",
    "POLISHING", "PROFILE", "PUNCH", "PUNCHES", "REGRIND", "REQUIRED",
    "RETENTION", "REV", "ROLL", "SCALE", "SEE", "SHAPE", "SHARPENING",
    "SHEAR", "SLUG", "SLOT", "SPECIFIED", "STAGE", "STEPS", "STRAIGHT", "SURFACE",
    "SURFACES", "TAPER", "THICK", "THRU", "TOLERANCE", "TOLERANCES",
    "TOOL", "UNIT", "USEABLE", "WIDE", "WIRE", "WITHOUT", "CARBIDE",
})
FIELD_LABELS = frozenset({
    "MATERIAL", "HARDNESS", "STAGE", "REV", "DATE", "SCALE", "UNIT",
    "PART", "NAME", "QTY", "APP", "CHK", "DR", "MARK", "RECORD",
    "REL DATE", "FINISH",
})
UNITS = frozenset({"MM", "CM", "IN", "INCH", "INCHES", "DIA", "REF", "RC", "HRC"})


# NFKC和大写仅供规则比较
# 不写回原始OCR
# 空白折叠不能当原字符跨度
def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).upper().split())


# 长串单字母是OCR碎片信号
# 至少五词才按比例判断
# 返回复核需求而非删除证据
def _is_fragmented_letters(text: str) -> bool:
    words = re.findall(r"\b[A-Z]+\b", _normalized(text))
    return len(words) >= 5 and sum(len(word) == 1 for word in words) >= 3 \
        and sum(len(word) == 1 for word in words) / len(words) >= 0.5


# 当前源文策略先于旧术语规则
# 严格英文标签白名单先放行
# 明确尺寸/代号保持原文
# 语义未知与碎片仍进入后续判断
# 纯数字或符号不送译
# 孤立字母可能是图纸分区
# 姓名缩写与签核字段单独排除
# 单位/数量/比例不是说明句
# 半径螺纹公差按规格识别
# 日期材料牌号保留字面
# 公司名不由机械术语猜译
# 噪声混排进入review_required
# 非英文用语保持疑难归因
# 已知加工术语进入候选
# 未知英文仍可自动尝试
# 分类不代表模型或专家正确
# 整个函数不改原始文本
def classify_text(text: str) -> Classification:
    """按规格、已知语义、未知内容的顺序分类；不改动传入文本。"""
    value = _normalized(text).strip(" ,;:")
    from .source_text_policy import english_translation_eligibility
    policy=english_translation_eligibility(text)
    if policy.reason=='exact_english_field_label':
        return Classification('translatable','严格英文图签标签白名单')
    if not policy.eligible and policy.reason not in {'english_meaning_not_established','fragmented_ocr_letters'}:
        return Classification('non_translatable','原文保留：'+policy.reason)
    if not value or not re.search(r"[A-Z]", value):
        return Classification("non_translatable", "纯数字或工程符号")
    if _is_fragmented_letters(value):
        return Classification("review_required", "疑似 OCR 字母碎片")
    if re.fullmatch(r"[A-Z]|[IVX]{2,}|[A-Z](?:\s*\.\s*[A-Z]){1,3}\.?", value):
        return Classification("non_translatable", "图纸分区或姓名缩写")
    if re.fullmatch(r"[A-Z]\.\s+[A-Z]{2,}", value):
        return Classification("non_translatable", "姓名或签名")
    if value.strip(".") in {"NO", "QTY", "APP", "CHK", "DR", "DATE",
                             "PRODUCT NO", "SER NO", "IDP NO"}:
        return Classification("non_translatable", "图签编号或签核字段")
    if re.fullmatch(r"[A-Z]{2,4}_[A-Z0-9_]+", value):
        return Classification("non_translatable", "文件名或内部代码")
    if value in UNITS or re.fullmatch(r"\d+\s*(?:X|PCS?|SETS?)", value):
        return Classification("non_translatable", "单位或数量标记")
    if re.fullmatch(r"\d+\s+PLACES?", value):
        return Classification("non_translatable", "精度位数标记")
    if re.fullmatch(r"\d[\d.]*\s+\d+\s*[X×]", value):
        return Classification("non_translatable", "尺寸与重复数量标记")
    if re.fullmatch(r"SCALE\s+\d+(?:\s*:\s*\d+)?", value):
        return Classification("non_translatable", "比例数值")
    if re.fullmatch(r"(?:[RC]\s*[O0]?\.?\d[\d.,+\-]*|M\d+(?:[X×*]\d+(?:\.\d+)?)?|Ø\s*\d[\d.,]*)"
                    r"(?:\s*\(\d+\))?", value):
        return Classification("non_translatable", "半径、倒角、直径或螺纹规格")
    if re.fullmatch(r"[+\-]?\d[\d.,+\-]*(?:\s*\([A-Z]{1,4}\))?", value):
        return Classification("non_translatable", "尺寸或公差")
    if re.fullmatch(r"\d[\d.,]*\s*(?:DIA|REF\.?|MM|H7|HRC)(?:\s*\(\d+\))?", value):
        return Classification("non_translatable", "尺寸、单位或配合规格")
    if re.fullmatch(r"\d[\d.,]*\s*(?:MM|CM|IN|INCHES?)\s*\(REF\)", value):
        return Classification("non_translatable", "参考尺寸与单位")
    if re.fullmatch(r"#?\d+(?:-\d+)?\s+(?:THRD|THREAD)(?:\s+THRU)?", value):
        return Classification("non_translatable", "螺纹规格")
    if re.fullmatch(r"\d{1,2}[./-]?[A-Z]{3,}[./-]?\d{2,4}", value) or re.fullmatch(
        r"\d{1,2}[./-][A-Z]{3,}\s+\d{2,4}", value
    ) or re.fullmatch(r"\d{1,2}\.?[A-Z]{1,3}'?\*?\s+\d{2,4}", value):
        return Classification("non_translatable", "日期")
    if re.fullmatch(r"(?:HRA|HRC|HRB)\s*\d+(?:[~+\-]\d+)?", value):
        return Classification("non_translatable", "硬度数值")
    if re.fullmatch(r"[A-Z]{2,}-[A-Z]?\d+\s+RC\s*\d+(?:-\d+)?", value):
        return Classification("non_translatable", "材料牌号与硬度")
    if re.fullmatch(r"[A-Z]{2,}\d{2,}", value) or (
        " " not in value and re.search(r"\d", value)
        and re.fullmatch(r"[A-Z0-9#./_+\-]+", value)
    ):
        return Classification("non_translatable", "型号、图号或产品编号")
    if re.fullmatch(r"[A-Z0-9.]+\s+CO\.*\s*LTD\.?", value):
        return Classification("non_translatable", "公司名称")
    if re.search(r"\b[A-Z]{5,}\s+TOOL\s+\d{5,}", value):
        return Classification("review_required", "术语与疑似 OCR 噪声混排")
    words = re.findall(r"[A-Z]+", value)
    if any(word in {"SIEHE", "SCHNITT", "MASSTAB", "MASSSTAB",
                    "WERKSTUECKKANTEN", "TOLERANZEN"} for word in words):
        return Classification("review_required", "包含非英文图纸用语")
    if any(word in TECHNICAL_WORDS for word in words):
        return Classification("translatable", "包含已知加工术语或说明用语")
    if len(words) >= 2:
        return Classification("review_required", "英文组合语义不确定")
    return Classification("review_required", "孤立英文或缩写语义不确定")


# 至少一单位避免除零
# 仅用于几何相对阈值
# 不改变源区域实际边界
def _height(region: TextRegion) -> float:
    return max(1.0, region.bounds.max_y - region.bounds.min_y)


# 匹配图签字段与公共标签
# 去点和冒号只用于比较
# 防止表格邻格误合正文
def _is_field_label(text: str) -> bool:
    from .source_text_policy import ENGLISH_FIELD_LABELS
    value=' '.join(_normalized(text).replace('.','').strip(':').split())
    return value in FIELD_LABELS or value in ENGLISH_FIELD_LABELS


_NATIVE_PROCESS = re.compile(
    r"^(?:\d+[.)]\s+)?(?:DO\s+NOT\s+)?"
    r"(?:GRIND|POLISH|MACHINE|DRILL|REAM|DEBURR|CHAMFER|REMOVE|COAT)\b", re.I,
)


# PART/MATERIAL可作加工宾语
# 窄例外只允许原生PDF
# 前文必须是受控加工指令
# 句终或冒号中断该上下文
# 签名字段不能借附近动词放行
# 右侧字段前必须有明确连接
# 此判断不豁免几何邻接门禁
def _native_instruction_field_link(left: TextRegion, right: TextRegion, prefix: str) -> bool:
    """Recognize object homographs only within a native machining instruction.

    PART/MATERIAL may be direct objects. Signoff/name fields, punctuation, and
    a sequence of adjacent fields do not gain authority from a nearby verb.
    Geometry and original source identity are checked by the caller.
    """
    if (left.source != "pdf_native" or right.source != "pdf_native"
            or not _NATIVE_PROCESS.match(prefix)
            or re.search(r"[.!?;:](?=\s|$)", prefix)):
        return False
    fields = [r for r in (left, right) if _is_field_label(r.source_text)]
    if any(_normalized(r.source_text) not in {"PART", "MATERIAL"} for r in fields):
        return False
    if _is_field_label(right.source_text) and not re.search(
        r"\b(?:GRIND|POLISH|MACHINE|DRILL|REAM|DEBURR|CHAMFER|REMOVE|COAT|THE|ALL|OF|TO)$",
        prefix, re.I,
    ):
        return False
    return True


# 文档/页/来源必须完全一致
# 大角度或方向差太大拒绝
# 碎字和新编号不跨界合并
# 语义模式不穿过元数据字段
# 行高比例控制到1.4以内
# 同一行要求足够竖向交叠
# 横向间距相对行高限定
# 字段只在窄原生指令例外合并
# 跨行要求足够词数或语法延续
# 已有句点通常阻止下行续接
# 向下行差须真实存在
# 左右缩进与横向覆盖受限
# 局部救回正文可放宽悬挂缩进
# 还必须有明确continuation
# 最后检查两段间通道障碍
# 不能跨尺寸或表格格子拼句
# 这些阈值不修改原坐标
def _can_merge(left: TextRegion, right: TextRegion, all_regions: Sequence[TextRegion],
               *, semantic_mode: bool = False, enable_integrated_quality: bool = False,
               instruction_prefix: str = "") -> bool:
    if (left.document_id != right.document_id or left.page_index != right.page_index
            or left.source != right.source or abs(left.rotation_degrees - right.rotation_degrees) > 8
            or abs(left.rotation_degrees) > 15 or abs(right.rotation_degrees) > 15):
        return False
    if _is_fragmented_letters(left.source_text) or _is_fragmented_letters(right.source_text):
        return False
    if re.match(r"^\s*\d+\.\s", right.source_text):
        return False
    if semantic_mode and (is_metadata_field(left.source_text)
                          or is_metadata_field(right.source_text)):
        return False
    h1, h2 = _height(left), _height(right)
    if max(h1, h2) / min(h1, h2) > 1.4:
        return False
    a, b = left.bounds, right.bounds
    vertical_overlap = min(a.max_y, b.max_y) - max(a.min_y, b.min_y)
    gap_x = b.min_x - a.max_x
    same_line = vertical_overlap >= 0.6 * min(h1, h2) and 0 <= gap_x <= 0.6 * min(h1, h2)
    if _is_field_label(left.source_text) or _is_field_label(right.source_text):
        if not (enable_integrated_quality and same_line
                and _native_instruction_field_link(left, right, instruction_prefix)):
            return False
    words_left = len(re.findall(r"[A-Za-z]+", left.source_text))
    words_right = len(re.findall(r"[A-Za-z]+", right.source_text))
    gap_y = b.min_y - a.max_y
    line_step = (b.min_y + b.max_y - a.min_y - a.max_y) / 2
    # Only restored image-backed bodies may attach a hanging-indent tail;
    # the legacy OCR-3B geometry remains unchanged when OCR-3C is off.
    indent_factor=(1.5 if semantic_mode and left.status.startswith("局部复识别")
                   and has_explicit_continuation(left.source_text,right.source_text) else .8)
    multiline = (
        ((words_left >= 3 and words_right >= 3)
         or (semantic_mode and words_left >= 1 and words_right >= 1
             and has_explicit_continuation(left.source_text, right.source_text)))
        and not left.source_text.rstrip().endswith(".")
        and -0.5 * min(h1, h2) <= gap_y <= 0.8 * min(h1, h2)
        and line_step >= 0.45 * min(h1, h2)
        and abs(a.min_x - b.min_x) <= indent_factor * min(h1, h2)
        and min(a.max_x, b.max_x) - max(a.min_x, b.min_x)
        >= 0.4 * min(a.max_x - a.min_x, b.max_x - b.min_x)
    )
    if not (same_line or multiline):
        return False
    # 文字框占据两段之间的通道时，不越过它拼接表格或尺寸标记。
    for other in all_regions:
        if other is left or other is right or other.page_index != left.page_index:
            continue
        box = other.bounds
        if same_line:
            blocked = (box.min_x < b.min_x and box.max_x > a.max_x
                       and box.min_y < min(a.max_y, b.max_y)
                       and box.max_y > max(a.min_y, b.min_y))
        else:
            blocked = (box.min_y < b.min_y and box.max_y > a.max_y
                       and box.min_x < min(a.max_x, b.max_x)
                       and box.max_x > max(a.min_x, b.min_x))
        if blocked:
            return False
    return True


# IN默认可能是英寸单位
# 只有明确运动谓词才作介词
# 仅针对原生PDF完整上下文
# 这是候选资格窄例外
# _can_merge仍检查距离和障碍
def _native_motion_preposition(region: TextRegion, prefix: str) -> bool:
    """IN is a preposition only after an explicit native motion predicate.

    This is a candidate-eligibility exception, never a geometric permission.
    _can_merge still requires matching page/source/direction, close neighbors
    and an unobstructed channel. Numeric IN units and field labels do not match.
    """
    return bool(region.source == 'pdf_native' and _normalized(region.source_text) == 'IN'
                and re.fullmatch(r'(?:THE )?[A-Z]+(?: [A-Z]+){0,3} (?:MUST|SHALL) SLIDE FREELY',
                                 _normalized(prefix)))


# 先按页/纵坐标/横坐标排序
# ID作稳定同位决胜
# used避免区域重复归组
# 人工override形成分组边界
# 组字面必须满足英文片段策略
# 各候选仍独立检查分类
# 专业IN例外限定前缀
# 相邻合并始终走_can_merge
# 优先最近同一行右邻
# 否则延续紧邻下一行
# 种子本身不可译不向外吸收
# 原区域只作为引用加入元组
# 不会在此合成新OCR文本
# 后续语义层再拆字符跨度
def _groups(regions: tuple[TextRegion, ...], overrides: Mapping[str, Category],
            *, semantic_mode: bool = False,
            enable_integrated_quality: bool = False,
            enable_professional_quality: bool = False) -> list[tuple[TextRegion, ...]]:
    from .source_text_policy import is_translatable_english,is_english_group_fragment
    ordered = sorted(regions, key=lambda r: (r.page_index, r.bounds.min_y,
                                             r.bounds.min_x, r.id))
    used: set[str] = set()
    groups: list[tuple[TextRegion, ...]] = []
    for seed in ordered:
        if seed.id in used:
            continue
        group = [seed]
        used.add(seed.id)
        while seed.id not in overrides:
            if not is_english_group_fragment(' '.join(r.source_text for r in group)):
                break
            current = group[-1]
            if current.id in overrides:
                break
            prefix = " ".join(r.source_text for r in group) if enable_integrated_quality or enable_professional_quality else ""
            candidates = [r for r in ordered if r.id not in used and r.id not in overrides
                          and is_english_group_fragment(r.source_text)
                          and (classify_text(r.source_text).category != "non_translatable"
                               or (enable_professional_quality and _native_motion_preposition(r, prefix)))
                          and _can_merge(current, r, regions, semantic_mode=semantic_mode,
                                         enable_integrated_quality=enable_integrated_quality,
                                         instruction_prefix=prefix)]
            if not candidates:
                break
            # 同一行优先取最近的右邻；否则只延续紧邻的下一行。
            next_region = min(candidates, key=lambda r: (
                r.bounds.min_y > current.bounds.max_y,
                abs(r.bounds.min_y - current.bounds.min_y),
                abs(r.bounds.min_x - current.bounds.max_x),
            ))
            if classify_text(seed.source_text).category == "non_translatable":
                break
            group.append(next_region)
            used.add(next_region.id)
        groups.append(tuple(group))
    return groups


# 输入先冻结为原区域元组
# override类别必须合法
# 语义/综合/专业任一启用走语义模式
# 几何组先分类整体字面
# 每成员保留同组分类依据
# 人工判定在区域级覆盖
# candidate排除明确不可译组
# all_entries仍保留单区域行
# counts按原区域决策计算
# 语义恢复产生新SourceSpan
# 恢复后再次检英文可译策略
# 不因覆盖率指标绕开筛选
# 返回结果独立于OCR和Qt
# 原始对象和几何保持不变
def select_targets(
    regions: Sequence[TextRegion], overrides: Mapping[str, Category] | None = None,
    *, enable_semantic_grouping: bool = False, enable_integrated_quality: bool = False,
    enable_professional_quality: bool = False,
) -> SelectionResult:
    """分类与分组只生成旁路结果，原始 TextRegion 对象和坐标保持不变。"""
    original = tuple(regions)
    semantic_mode = enable_semantic_grouping or enable_integrated_quality or enable_professional_quality
    override_map = dict(overrides or {})
    if any(value not in CATEGORIES for value in override_map.values()):
        raise ValueError("人工判定类别无效")
    decisions: dict[str, Classification] = {}
    candidate_entries: list[TargetEntry] = []
    for group in _groups(original, override_map, semantic_mode=semantic_mode,
                         enable_integrated_quality=enable_integrated_quality,
                         enable_professional_quality=enable_professional_quality):
        combined = classify_text(" ".join(r.source_text for r in group))
        for member in group:
            decisions[member.id] = (
                Classification(override_map[member.id], "人工判定")
                if member.id in override_map else combined
            )
        classification = decisions[group[0].id]
        if classification.category != "non_translatable":
            candidate_entries.append(TargetEntry(group, classification.category,
                                                 classification.reason))
    all_entries = tuple(TargetEntry((r,), decisions[r.id].category,
                                    decisions[r.id].reason) for r in original)
    counts: dict[Category, int] = {category: 0 for category in CATEGORIES}
    for decision in decisions.values():
        counts[decision.category] += 1
    candidates = (recover_semantic_entries(
        candidate_entries, enable_integrated_quality=enable_integrated_quality,
    ) if semantic_mode else tuple(candidate_entries))
    from .source_text_policy import is_translatable_english
    candidates=tuple(e for e in candidates if is_translatable_english(e.source_text))
    return SelectionResult(original, decisions, candidates,
                           all_entries, counts)


class DecisionStore:
    """仅在人工作出判定后，将位置指纹写入用户数据目录。"""

    # 旧人工分类独立存本地
    # 默认目录是LocalAppData
    # 不是正式项目审核来源
    # 主窗口仅显式注入时读取旧判定
    def __init__(self, path: str | Path | None = None):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        self.path = Path(path) if path is not None else (
            base / "MechanicalDrawingTranslator" / "translation_decisions.json"
        )

    # 旧指纹绑定路径/大小/mtime
    # 这是分类缓存键
    # 不能代替项目源SHA校验
    # 内容变更通常使旧判定隔离
    @staticmethod
    def _document_key(path: str | Path) -> str:
        source = Path(path).resolve()
        stat = source.stat()
        data = f"{str(source).casefold()}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    # 随机OCR UUID不进入位置指纹
    # 原文/来源/页/边界共同关联
    # 坐标保留两位作稳定比较
    # 方向保留一位
    # 不能据此授予人工译文来源
    @staticmethod
    def _region_key(region: TextRegion) -> str:
        box = region.bounds
        data = (region.page_index, _normalized(region.source_text), region.source,
                round(box.min_x, 2), round(box.min_y, 2),
                round(box.max_x, 2), round(box.max_y, 2),
                round(region.rotation_degrees, 1))
        return hashlib.sha256(repr(data).encode("utf-8")).hexdigest()

    # 缺文件代表没有旧判定
    # 损坏文件明确报读取错误
    # 根节点必须为JSON对象
    # 不静默伪造人工分类记录
    def _read(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取人工判定文件：{self.path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("人工判定文件格式无效")
        return payload

    # 只返回当前实际区域的判定
    # 位置指纹匹配后还验类别
    # 未知旧类别不进入当前分组
    def overrides_for(self, path: str | Path, regions: Sequence[TextRegion]) -> dict[str, Category]:
        saved = self._read().get(self._document_key(path), {})
        return {region.id: value for region in regions
                if (value := saved.get(self._region_key(region))) in CATEGORIES}

    # None表示移除旧人工分类
    # 有效类别才可写入
    # 先读旧数据保留其他文档
    # 同目录临时文件后原子替换
    # finally清除未提交临时文件
    # 这里不保存译文审核事件
    def set_override(self, path: str | Path, region: TextRegion,
                     category: Category | None) -> None:
        if category is not None and category not in CATEGORIES:
            raise ValueError("人工判定类别无效")
        payload = self._read()
        saved = payload.setdefault(self._document_key(path), {})
        key = self._region_key(region)
        if category is None:
            saved.pop(key, None)
        else:
            saved[key] = category
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".decisions-", suffix=".tmp",
                                             delete=False) as stream:
                temp_path = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
