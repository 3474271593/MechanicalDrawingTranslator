# mechanical_drawing_translator/translation_results.py
"""稳定的翻译结果输入契约，不依赖具体翻译引擎。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, Protocol


# 来源限定测试人工、已审核与机器三类，
# 正式导出默认仍只接收前两类。
# 机器自动预览资格与人工来源分开，
# 不能因为显示了红字就变为可正式导出。
TranslationProvenance = Literal["manual_test", "human_reviewed", "machine"]
STAGE1_PROVENANCE = frozenset({"manual_test", "human_reviewed"})


# 预览需有结果、无不可用标记且含中文，
# 该判断只服务显示资格。
# 它不是专业正确或遮挡安全判定，
# 机器待审文本也可成为preview_only布局。
def preview_translation_available(result) -> bool:
    """Chinese preview eligibility, separate from provenance and expert correctness."""
    return bool(result and 'translation_unavailable' not in result.warnings
                and re.search(r'[\u3400-\u9fff]',result.translated_text))

# 无需翻译的用户决定单独显示原文保留，
# 与未成功模型翻译的回退区分。
# 未有结果显示尚未翻译，
# 避免把空状态说成中文已生成。
def translation_status_text(result) -> str:
    if result is None:return '尚未翻译'
    if 'reviewed94:no_translation_required' in result.warnings:return '原文保留（无需翻译）'
    return '机器译文（自动预览）' if preview_translation_available(result) else '未成功翻译（保留英文）'


class TranslationGroup(Protocol):
    document_id: str
    page_index: int
    region_ids: tuple[str, ...]
    source_text: str


def translation_group_id(entry: TranslationGroup) -> str:
    """为当前 OCR payload 内的 group 生成确定性标识。"""
    # 有字符spans时组身份包含语义版本，
    # 文档、页、每段范围和实际源文。
    # 同一OCR框内不同句子不共享group ID，
    # 防止一条译文借给同框其它文字。
    spans = getattr(entry, "spans", ())
    if spans:
        from .semantic_grouping import SEMANTIC_VERSION

        value = json.dumps((
            SEMANTIC_VERSION, entry.document_id, entry.page_index,
            [(s.region_id, s.start, s.end) for s in spans], entry.source_text,
        ), ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    # 历史无spans模式按文档页和region序列，
    # 保持旧结果关联与缓存可回归。
    # 有spans的新身份不能随意退回旧算法，
    # 否则句界变化后旧译文可能错误复用。
    value = "|".join((entry.document_id, str(entry.page_index), *entry.region_ids))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
# 不可变契约连接翻译、预览与导出，
# source_text必须保留原送审/识别英文。
# warnings是机器风险与回退原因，
# 不能只存中文正文而丢失来源边界。
# region_ids保持原序列用于坐标关联。
class TranslationResult:
    """标注和导出层消费的版本化翻译结果。"""

    schema_version: int
    document_id: str
    page_index: int
    group_id: str
    region_ids: tuple[str, ...]
    source_text: str
    translated_text: str
    provenance: TranslationProvenance
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        # JSON序列化将元组变为普通数组，
        # 不把Python对象直接交外部持久层。
        # 读回时再恢复元组契约，
        # 同一字段名不接受任意额外扩展。
        payload = asdict(self)
        payload["region_ids"] = list(self.region_ids)
        payload["warnings"] = list(self.warnings)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TranslationResult":
        required = {
            "schema_version", "document_id", "page_index", "group_id",
            "region_ids", "source_text", "translated_text", "provenance", "warnings",
        }
        # 缺字段与未知字段分别报错，
        # 防止旧版或拼错key被静默忽略。
        # region_ids和warnings需字符串序列，
        # 不能把一整串误当多个字符项。
        provided = set(payload)
        missing = required - provided
        unknown = provided - required
        if missing:
            raise ValueError(f"translation result missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"translation result unknown fields: {sorted(unknown)}")
        region_ids = payload["region_ids"]
        warnings = payload["warnings"]
        if not isinstance(region_ids, (list, tuple)) or not all(
            isinstance(value, str) for value in region_ids
        ):
            raise ValueError("translation result region_ids must be a string list")
        if not isinstance(warnings, (list, tuple)) or not all(
            isinstance(value, str) for value in warnings
        ):
            raise ValueError("translation result warnings must be a string list")
        # from_dict负责基本类型转换与结构，
        # 并未证明结果对应当前目标。
        # 后续validate_translation_results，
        # 才校验文档、页、组与源串一致性。
        try:
            return cls(
                schema_version=int(payload["schema_version"]),
                document_id=str(payload["document_id"]),
                page_index=int(payload["page_index"]),
                group_id=str(payload["group_id"]),
                region_ids=tuple(region_ids),
                source_text=str(payload["source_text"]),
                translated_text=str(payload["translated_text"]),
                provenance=str(payload["provenance"]),
                warnings=tuple(warnings),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid translation result values: {exc}") from exc


def validate_translation_results(
    entries: Iterable[TranslationGroup],
    results: Iterable[TranslationResult],
    *,
    allowed_provenance: frozenset[str] = STAGE1_PROVENANCE,
) -> dict[str, TranslationResult]:
    """严格校验结果与当前目标 group 的一一对应关系。"""
    # 当前输入目标构成可接受group集合，
    # 外来页或旧任务的group直接拒绝。
    # 每个结果只能出现一次，
    # region_ids内部也不能重复。
    # 没有当前目标的译文不能绑定图纸。
    known = {translation_group_id(entry): entry for entry in entries}
    validated: dict[str, TranslationResult] = {}
    for result in results:
        if result.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if len(set(result.region_ids)) != len(result.region_ids):
            raise ValueError("duplicate region IDs in translation result")
        if result.group_id in validated:
            raise ValueError(f"duplicate group result: {result.group_id}")
        entry = known.get(result.group_id)
        if entry is None:
            raise ValueError(f"unknown group_id: {result.group_id}")
        # 文档、页、重新计算的group身份，
        # 与region顺序必须同时相同。
        # 原始英文也要求完全一致，
        # 阻止OCR改字后误复用旧译文。
        # 译文不可空且来源须在当前允许集内。
        if result.document_id != entry.document_id:
            raise ValueError("document_id does not match target group")
        if result.page_index != entry.page_index:
            raise ValueError("page_index does not match target group")
        if result.group_id != translation_group_id(entry):
            raise ValueError("group_id does not match target group")
        if result.region_ids != entry.region_ids:
            raise ValueError("region_ids do not match target group")
        if result.source_text != entry.source_text:
            raise ValueError("source_text does not match target group")
        if not result.translated_text.strip():
            raise ValueError("translated_text cannot be empty")
        if result.provenance not in allowed_provenance:
            raise ValueError(f"provenance is not accepted in this stage: {result.provenance}")
        validated[result.group_id] = result
    # 返回按group索引的已验证结果，
    # 本函数并不额外断言所有目标都有结果。
    # 调用批次服务需自行保证完整结果槽，
    # 避免把子集合法误说成整批已覆盖。
    return validated
