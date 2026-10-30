# mechanical_drawing_translator/engineering_source_risk.py
"""Conservative source-side OCR risk checks for engineering symbols."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# 风险结果保存原文与外部旁证，
# corrected字段此阶段始终为空。
# 检测到疑点并不等于确定错字，
# 调用者应回退或继续核验。
@dataclass(frozen=True)
class EngineeringSourceRisk:
    source_text: str
    warning_codes: tuple[str, ...] = ()
    trusted_source_text: str | None = None
    corrected_source_text: str | None = None

    @property
    def is_high_risk(self) -> bool:
        return bool(self.warning_codes)


# 必须出现完整角公差字段名，
# 才把缺失度数视作高风险。
# 单独数字或普通分数不触发，
# 避免把尺寸值一律判为角度。
# 正负号参与被检测的数值模式，
# 不会根据数值大小猜测单位。
_ANGLE_FIELD = re.compile(
    r"\b(?:ANGULAR\s+TOLERANCE|ANGLE\s+TOLERANCE|ANG\.\s*TOL(?:ERANCE)?)\b",
    re.I,
)
_SIGNED_VALUE = re.compile(r"±\s*(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?)")
_SIGNED_FRACTION_WITH_TRAILING_ZERO = re.compile(r"±\s*\d+\s*/\s*\d+0(?!\d)")

_WARNING_TEXT = {
    "engineering_source_angle_unit_missing": "工程符号高风险：角度公差缺少明确度数符号，请对照原图人工确认",
    "engineering_source_degree_zero_confusion": "工程符号高风险：度数符号可能被OCR识别为数字0，不得自动修正",
    "engineering_source_conflicts_with_trusted_evidence": "OCR英文与已登记的原图字面证据冲突，请人工裁决",
    "engineering_source_review_required": "工程符号风险尚未解决，机器译文已安全回退并保持原OCR",
    "d8c_semantic_boundary:reviewed_phrase_recovery": "D8-C已按原区域关系恢复既有审核完整加工说明",
    "d8c_ocr_character_auto_verified": "D8-C已通过原始局部图像核验工程字符；原OCR保持不变",
}


def warning_display_text(code: str) -> str:
    """Return a user-facing explanation while retaining unknown diagnostics."""
    # 已知警告转换为可读解释，
    # 未知代码原样保留便于追踪。
    # 核验后英文只作为详情呈现，
    # 显示文字本身不产生授权。
    if code.startswith("d8c_verified_translation_source:"):
        return "核验后送译英文：" + code.split(":", 1)[1]
    return _WARNING_TEXT.get(code, code)


def assess_source_risk(
    source_text: str, *, trusted_source_text: str | None = None,
) -> EngineeringSourceRisk:
    """Flag source OCR that is unsafe to translate, without repairing it.

    A numeric value without a degree marker is suspicious only inside a complete
    angular-tolerance field.  The narrower trailing-zero warning captures the
    observed degree-symbol/zero confusion while still requiring field context.
    """
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be a non-empty string")
    # NFKC仅用于比较显示变体，
    # 返回对象仍保存传入原文。
    # 空输入属于接口错误并抛出，
    # 不伪造一个低风险有效结果。
    normalized = unicodedata.normalize("NFKC", source_text)
    warnings: list[str] = []
    angle_field = _ANGLE_FIELD.search(normalized) is not None
    signed_value = _SIGNED_VALUE.search(normalized)
    if angle_field and signed_value is not None:
        # 检查正负数值之后紧邻的度数，
        # 不接受句中远处任意一个度号。
        # 尾零疑点仅在缺单位分支追加，
        # 它是进一步图像核验的线索。
        tail = normalized[signed_value.end():]
        if not re.match(r"\s*°", tail):
            warnings.append("engineering_source_angle_unit_missing")
            if _SIGNED_FRACTION_WITH_TRAILING_ZERO.search(normalized):
                warnings.append("engineering_source_degree_zero_confusion")
    # 外部可信英文与OCR单独比较，
    # 只清理两端空白和兼容字符。
    # 存在冲突时保留两份证据，
    # 不自行判断哪一份应覆盖。
    # 警告去重保持首次出现顺序，
    # 便于稳定展示与回归核对。
    if trusted_source_text is not None:
        trusted = unicodedata.normalize("NFKC", trusted_source_text).strip()
        if trusted and trusted != normalized.strip():
            warnings.append("engineering_source_conflicts_with_trusted_evidence")
    return EngineeringSourceRisk(
        source_text=source_text,
        warning_codes=tuple(dict.fromkeys(warnings)),
        trusted_source_text=trusted_source_text,
        corrected_source_text=None,
    )
