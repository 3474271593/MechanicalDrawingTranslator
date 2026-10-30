# mechanical_drawing_translator/export_item_policy.py
"""Identify inert source associations without granting translation/mask rights."""
from .annotation_models import MaskDecision
from .source_text_policy import english_translation_eligibility


# 仅这些明确源类型可以作为无墨迹保留关联，不能泛化到未知英文。
# 数字、工程符号、图号和视图标签保留源关联不等于生成了已审译文。
# 集合是保守白名单，新拒绝原因不会自动获得正式导出例外。
# 源策略负责识别字面类型，本模块再限定例外的无墨迹状态。
# 没有新增用户审核，也没有更改原 OCR 内容或 SourceSpan。
_CONFIRMED_SOURCE_KEEP_REASONS=frozenset({
    'numeric_or_symbol_literal','dimension_unit_or_engineering_marker',
    'drawing_or_part_identifier','label_with_identifier_requires_separate_source_region',
    'zone_or_initials','view_or_detail_reference','standalone_view_reference_label',
    'non_english_source','non_english_drawing_language',
})


# 预览路由变化后旧项目仍需保存原 group 与原区域关联。
# 这种关联可以不画任何内容，正式导出不应要求把它染成红色。
# 但例外不能承载中文、遮挡、多余行或伪造 SAFE 收据。
# 共享谓词由正式导出资格和像素回读共同调用，避免两套标准漂移。
# 返回 True 只说明该项没有输出墨迹，不给 machine 文本审核权限。
# 未知或可能需要翻译的英文即使标 preserved 也不能进入白名单。
def is_inert_preserved_source(item) -> bool:
    """Recompute current source evidence; never trust a saved placement label.

    These records keep old OCR/group links after routing changes, but paint
    neither Chinese nor a mask. Unknown English meaning is not positive keep
    evidence. A SAFE flag/receipt, glyph lines or changed target removes this
    exception even if a forged project calls the record ``preserved``.
    """
    # 原文须非空，译文必须与原文逐字相同，不能偷偷替换工程字面。
    # lines、mask_polygons、mask_evidence 均为空才没有可绘制的隐藏内容。
    # mask_decision 必须是未申请，SAFE 或待分析状态都破坏无墨迹例外。
    # 来源也受限于系统已知类别，不能用自造 provenance 绕过合同。
    # 任一条件不满足就回到正常审核与布局检查，不能部分豁免。
    if (item.placement!='preserved' or not item.source_text.strip()
            or item.translated_text!=item.source_text or item.lines or item.mask_polygons
            or item.mask_decision!=MaskDecision.NOT_REQUESTED or item.mask_evidence
            or item.provenance not in {'machine','manual_test','human_reviewed'}):
        return False
    # 最后重新运行当前源策略，不信任持久项目存的旧分类。
    # 必须明确不 eligible 且原因在确认白名单，两条件缺一不可。
    # 此处不改写 item，调用者可直接拒绝并保留原项目证据。
    policy=english_translation_eligibility(item.source_text)
    return not policy.eligible and policy.reason in _CONFIRMED_SOURCE_KEEP_REASONS
