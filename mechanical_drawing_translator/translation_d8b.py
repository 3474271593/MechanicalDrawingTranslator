# mechanical_drawing_translator/translation_d8b.py
"""Experimental, governed D8-B mechanical-expression resolver."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from .resource_paths import package_resource
from typing import Literal


# 候选参考状态不等于产品准入，
# verified_reference与review_pending分别保留。
# conflicting及unsafe绝不参与组合，
# 不能以英文完整命中解除来源冲突。
ReviewStatus = Literal[
    "verified_reference", "review_pending", "conflicting", "unsafe_or_incorrect",
]


@dataclass(frozen=True)
# 候选同时记录适用与不适用语境，
# 歧义、安全性和审核依据独立保存。
# 英文原字面与规范表达分字段，
# 避免把人工说明当成OCR修正真值。
class KnowledgeCandidate:
    identifier: str
    raw_source: str
    normalized_english: str
    suggested_chinese: str
    category: str
    source_evidence: str
    applicable_context: str
    inapplicable_context: str
    ambiguity: str
    safety: str
    review_status: ReviewStatus
    review_basis: str
    version: str


@dataclass(frozen=True)
class D8BKnowledge:
    knowledge_version: str
    source_name: str
    source_license_status: str
    production_eligible: bool
    candidates: tuple[KnowledgeCandidate, ...]

    @classmethod
    def load(cls, path: Path) -> "D8BKnowledge":
        # 仅从本地明确资源路径加载，
        # schema必须为当前支持版本。
        # 加载保留生产资格和许可证状态字段，
        # 解析器不会替用户补作许可证审核。
        # 候选非空且状态在有限枚举内。
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("unsupported D8-B knowledge schema")
        allowed = {"verified_reference", "review_pending", "conflicting", "unsafe_or_incorrect"}
        candidates = tuple(KnowledgeCandidate(
            identifier=item["id"], raw_source=item["raw_source"],
            normalized_english=item["normalized_english"],
            suggested_chinese=item["suggested_chinese"], category=item["category"],
            source_evidence=item["source_evidence"],
            applicable_context=item["applicable_context"],
            inapplicable_context=item["inapplicable_context"], ambiguity=item["ambiguity"],
            safety=item["safety"], review_status=item["review_status"],
            review_basis=item["review_basis"], version=item["version"],
        ) for item in data["candidates"])
        if not candidates or any(item.review_status not in allowed for item in candidates):
            raise ValueError("invalid D8-B knowledge candidate status")
        return cls(
            data["knowledge_version"], data["source_name"],
            data["source_license_status"], bool(data["production_eligible"]), candidates,
        )


@dataclass(frozen=True)
# 结果明确回传组成候选ID，
# evidence_status默认保持review_pending。
# 即使局部参考已核验，
# 组合后的整句仍由服务最终守卫判断。
class D8BResolution:
    translated_text: str
    mechanism: str
    candidate_ids: tuple[str, ...]
    evidence_status: str = "review_pending"


def default_d8b_knowledge_path() -> Path:
    return package_resource('translation', 'mechanical_knowledge_d8b.json')


# 默认知识对象最多缓存一个实例，
# 缓存资源不包括页面结果或用户译文。
# 显式knowledge参数可替代默认资源，
# 便于受控比较而不改全局词表。
@lru_cache(maxsize=1)
def default_d8b_knowledge() -> D8BKnowledge:
    return D8BKnowledge.load(default_d8b_knowledge_path())


# 已知OCR拼写疑点仅发诊断，
# 不会自动替换为貌似正确单词。
# 有疑点便拒绝本表达解析，
# 保持原英文供图像或人工核实。
def diagnose_input(source: str) -> tuple[str, ...]:
    normalized = source.upper()
    warnings: list[str] = []
    for suspect in ("APROX", "NATERIAL", "DRAMING"):
        if re.search(rf"\b{suspect}\b", normalized):
            warnings.append(f"suspected_ocr_spelling:{suspect}")
    return tuple(warnings)


# UNLESS的specified/stated/noted，
# 各自保留明确中文例外条件。
# 只允许有限otherwise变体，
# 不把IF或其它条件视为等价。
_CONDITIONS = {
    "UNLESS SPECIFIED": ("除非另有规定", "d8b-condition-specified"),
    "UNLESS OTHERWISE SPECIFIED": ("除非另有规定", "d8b-condition-specified"),
    "UNLESS STATED": ("除非另有说明", "d8b-condition-stated"),
    "UNLESS OTHERWISE STATED": ("除非另有说明", "d8b-condition-stated"),
    "UNLESS NOTED": ("除非另有注明", "d8b-condition-noted"),
    "UNLESS OTHERWISE NOTED": ("除非另有注明", "d8b-condition-noted"),
}


def _condition(text: str) -> tuple[str, str] | None:
    return _CONDITIONS.get(re.sub(r"\s+", " ", text.strip().upper()))


def resolve_expression(
    source: str, knowledge: D8BKnowledge | None = None,
) -> D8BResolution | None:
    """Resolve a complete bounded expression; never repairs the source English."""
    if not isinstance(source, str) or not source.strip() or diagnose_input(source):
        return None
    # 按候选ID构建本次可用证据映射，
    # 任何缺失依赖都让对应表达无候选。
    # resolved集中检查冲突与不安全状态，
    # 保证每种表达使用相同准入边界。
    active = knowledge or default_d8b_knowledge()
    by_id = {item.identifier: item for item in active.candidates}

    def resolved(target: str, mechanism: str, identifiers: tuple[str, ...]) -> D8BResolution | None:
        items = tuple(by_id.get(identifier) for identifier in identifiers)
        if any(item is None for item in items):
            return None
        if any(item.review_status in {"conflicting", "unsafe_or_incorrect"} for item in items if item):
            return None
        # 全部组成证据已核验才回传verified，
        # 只要一项待审整句就保持待审。
        # 该状态不自动转为human_reviewed，
        # 实际采用仍需TranslationService复检。
        status = "verified_reference" if all(
            item and item.review_status == "verified_reference" for item in items
        ) else "review_pending"
        return D8BResolution(target, mechanism, identifiers, status)
    stripped = source.strip()
    upper = re.sub(r"\s+", " ", stripped.upper())

    # 独立括号条件保留括号范围，
    # 不是把其中词汇用于任意句中替换。
    # 整串条件命中后才生成中文，
    # 外侧未知文字使其回到其它路径。
    parenthesized = upper.startswith("(") and upper.endswith(")")
    condition_text = upper[1:-1].strip() if parenthesized else upper
    condition = _condition(condition_text)
    if condition is not None:
        target, candidate_id = condition
        if parenthesized:
            target = f"（{target}）"
        return resolved(target, "condition_expression", (candidate_id,))

    match = re.fullmatch(
        r"ALL WELDS\s+(?P<size>\d+(?:\.\d+)?\s*(?:MM|CM|IN))\s+FILLET\s+"
        r"(?P<condition>UNLESS(?: OTHERWISE)? (?:SPECIFIED|STATED|NOTED))",
        upper,
    )
    if match:
        condition = _condition(match.group("condition"))
        if condition is not None:
            # 焊缝模板保留实际原文尺寸大小写，
            # 源中的ALL、FILLET与UNLESS均必需。
            # 不能从中文参考补出缺失焊脚值，
            # 也不换算MM/CM/IN量纲。
            source_size = re.search(r"\d+(?:\.\d+)?\s*(?:mm|cm|in)", stripped, re.I)
            if source_size is None:
                return None
            return resolved(
                f"{condition[0]}，所有焊缝均为 {source_size.group(0)} 角焊缝",
                "weld_instruction", ("d8b-weld-fillet", condition[1]),
            )

    # 加工命令只接收列出的完整表达，
    # 去毛刺与处理锐边分别保留对象。
    # 必须无氧化皮锈蚀保留义务与否定，
    # 不把工艺名词标题当成强制指令。
    if upper == "REMOVE ALL BURRS":
        return resolved("去除所有毛刺", "mandatory_action", ("d8b-action-remove-burrs",))
    if upper in {"REMOVE ALL BURRS AND SHARP EDGES", "REMOVE ALL BURRS & SHARP EDGES"}:
        return resolved(
            "去除所有毛刺并处理所有锐边", "mandatory_action", ("d8b-action-remove-burrs",),
        )
    if upper in {"MUST BE FREE OF SCALE & RUST", "MUST BE FREE OF SCALE AND RUST"}:
        return resolved(
            "必须无氧化皮和锈蚀", "mandatory_condition", ("d8b-clean-scale-rust",),
        )

    # 角公差与圆角值只复制被捕获字面，
    # 末尾单位或度号是否足够安全，
    # 仍交源风险与工程守卫独立判断。
    # 本模块解析成功不代表OCR字符已核实。
    match = re.fullmatch(r"ANGULAR TOLERANCE\s+(?P<value>.+)", stripped, re.I)
    if match and re.fullmatch(r"[±+\-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?°?", match.group("value")):
        return resolved(
            f"角度公差 {match.group('value')}", "numeric_object_binding",
            ("d8b-angular-tolerance",),
        )
    match = re.fullmatch(r"ALL UNSPECIFIED RADII\s*-\s*(R\s*\d+(?:\.\d+)?)", stripped, re.I)
    if match:
        return resolved(
            f"所有未注圆角均为 {match.group(1)}", "numeric_object_binding", ("d8b-radii",),
        )
    match = re.fullmatch(r"TYP\.?\s+(\d+)\s+POSNS?\.?", stripped, re.I)
    if match:
        return resolved(
            f"典型，共 {match.group(1)} 处", "numeric_object_binding", ("d8b-positions",),
        )

    # 热处理的名词、已完成形态与之后关系，
    # 各有独立完整表述而非词干替换。
    # 未知词形不会猜成最接近过程，
    # 统一返回None保持范围边界。
    heat = {
        "HEAT TREAT": "热处理", "HEAT TREATMENT": "热处理",
        "HEAT TREATED": "已热处理", "AFTER HEAT TREATMENT": "热处理后",
    }.get(upper)
    if heat is not None:
        return resolved(heat, "process_morphology", ("d8b-heat-treatment",))
    return None
