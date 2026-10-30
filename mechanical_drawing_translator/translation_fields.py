# mechanical_drawing_translator/translation_fields.py
"""Bounded drawing-field vocabulary; new lexical choices require human review."""
import re

# Engineering-curated field meanings, not newly human-approved PDF references.
# Whole expressions only: never substitute these tokens inside arbitrary prose.
# 字段表只用于完整表达精确匹配，
# 新词汇选择仍需专业人工复核。
# DIE、INSERT等普通词不单独替换，
# 避免字段语义错误扩散进加工说明。
FIELDS = {
    'HARDNESS': '硬度', 'DIE NO': '模具编号', 'DIE NO.': '模具编号',
    'TRIM INSERT': '修边镶件', 'REVISED SLUG RETENTION': '修订防跳屑',
    'PART NAME': '零件名称', 'SCALE': '比例', 'UNIT': '单位',
    'REV': '版本', 'REV.': '版本', 'DRAWN BY': '绘图', 'CHECKED BY': '审核',
    "MAT'L": '材料',
}
# 缩写未确立含义则保持英文，
# 不凭图纸常见程度猜中文。
# MM[IN]是双单位记法应原样保留，
# 与普通英文介词IN分开。
UNRESOLVED_ABBREVIATIONS = frozenset({'CSR', 'CV', 'KST', 'UG', 'MTD', 'RS', 'BS'})
NUMBER = r'(?:\d+\.\d+|\.\d+|\d+)'


def resolve_field(source: str) -> tuple[str, str] | None:
    normalized = ' '.join(source.strip().upper().split())
    if normalized in FIELDS:
        return FIELDS[normalized], 'engineering_curated_field'
    if re.fullmatch(r'MM\s*\[\s*IN\s*\]', normalized):
        return source.strip(), 'unit_notation_preserved'
    match = re.fullmatch(r'DETAIL\s+([A-Z])', normalized)
    if match:
        return '详图 ' + match[1], 'drawing_view_letter'
    # 尺寸方括号整对复制到结果，
    # LAND与W.S.H.限定该完整结构。
    # 这里只生成有限词汇候选，
    # 最终仍由服务执行数值量纲守卫。
    match = re.fullmatch(rf'({NUMBER}\s*\[\s*{NUMBER}\s*\])\s+(LAND|W\.S\.H\.)', source.strip(), re.I)
    if match:
        target = '直伸' if match[2].upper() == 'LAND' else 'W.S.H.（线割始孔）'
        return match[1] + ' ' + target, 'dimension_pair_known_term'
    return None


def validate_field_terms(source: str, translated: str) -> tuple[str, ...]:
    # Only unambiguous technical expressions, never generic DIE/INSERT tokens.
    # 字段词义检查仅含明确技术表达，
    # HARDNESS与SLUG RETENTION分别核对。
    # 带字母边界避免型号内部误命中，
    # 没有覆盖其它词不等于它们已正确。
    warnings = []
    for term, target in [('HARDNESS', '硬度'), ('SLUG RETENTION', '防跳屑')]:
        if re.search(r'(?<![A-Z])' + term + r'(?![A-Z])', source.upper()) and target not in translated:
            warnings.append('d7f_term_missing:' + term)
    return tuple(warnings)
