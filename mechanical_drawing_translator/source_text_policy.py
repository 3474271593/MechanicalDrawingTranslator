# mechanical_drawing_translator/source_text_policy.py
"""Conservative English-note eligibility; never rewrites OCR or coordinates.

ASCII letters alone are not evidence of English prose: dimensions, units,
view references and drawing identifiers retain their exact source literal.
This policy decides routing only. Existing engineering validators still own
literal preservation inside qualified mixed English instructions.
"""
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
# 政策结果只决定是否送英文翻译，
# reason解释为何保留或准入。
# 不生成删除数字后的新字符串，
# 因此原OCR、跨度和坐标保持不变。
class SourceTextEligibility:
    eligible: bool
    reason: str


# Exact labels only. The same words followed by a drawing identifier are not
# silently split into fabricated sub-region coordinates or sent as one note.
ENGLISH_FIELD_LABELS=frozenset({'DWG NO','DRAWING NO','MATERIAL','HARDNESS','STAGE',
    'REV','FINISH','PART NAME','NAME','REL DATE','DRAWN BY','CHECKED BY','APPROVED BY',
    'DRWN BY','CHK BY','APPR BY'})
# 纯单位、硬度符号与尺寸标记，
# 即使含ASCII字母也不是英语说明。
# 功能词用于后续句界组合，
# 不能让孤立DO/NOT直接进入模型。
_ENGINEERING_MARKERS=frozenset({'R','C','D','Ø','DIA','REF','MAX','MIN','TYP','EQ','THRU',
    'MM','CM','M','IN','INCH','INCHES','FT','DEG','RAD','HRC','HRA','HRB','RC','X','PCS','PC',
    'PLCS','PLACES','UNC','UNF','NPT','H7'})
_ENGLISH_CUES=frozenset({'DRILL','HOLE','HOLES','FLAT','LAND','VERTICAL','HORIZONTAL',
    'REMOVE','BURRS','BURR','EDGES','EDGE','SHARP','ALLOWED','ALLOW','SLIDE','FREELY','BETWEEN',
    'BEFORE','AFTER','UNLESS','GRINDING','COAT','COATED','APPLY','REAM','DEBURR','INSTALL',
    'CHECK','INSPECT','ENSURE','KEEP','FACE','DEPTH','ANGLE','ANGULAR','RADIUS','DIAMETER',
    'REFERENCE','SECTION','NOTES','TOL','UPPER','LOWER','OUTER','INNER','CENTER','CENTRE',
    'BREAK','ROUND','ROUNDING','PERPENDICULAR','PARALLEL','SMOOTH','DEGREES','SPRING','PLATE',
    'INSERTION','POINT','LOCATION','ROUGHNESS','MAINTAIN'})
_NON_ENGLISH=frozenset({'SIEHE','SCHNITT','MASSTAB','MASSSTAB','WERKSTUECKKANTEN','TOLERANZEN'})
_FUNCTION_WORDS=frozenset({'A','AN','THE','DO','NOT','THEN','ALL','OTHERWISE','ARE','IS','BE','HAS',
    'HAVE','SHALL','SHOULD','MAY','NO','AS','TO','FOR','AND','OR','FROM','AT','IF','WHEN','THAN',
    'EXCEPT','ALSO','WITHIN','EACH','BOTH','EITHER','ONLY','UPON','THAT','IT','THESE','THOSE',
    'BY','ON','OF','WITH','WITHOUT','BEFORE','AFTER','UNLESS','UNTIL','IN'})


def english_translation_eligibility(text: str) -> SourceTextEligibility:
    """Return routing evidence for an entire immutable source span.

    No number/unit removal is returned to callers. Qualified mixed notes keep
    their complete source string so engineering literals remain protected.
    """
    # NFKC和大小写只用于路由判定，
    # 传入原始完整字符串不会被重写。
    # 无英文字母或已知非英语文字，
    # 直接保留原文而不强行英译中。
    value=' '.join(unicodedata.normalize('NFKC',text).upper().split()).strip(' ,;:')
    if not value or not re.search('[A-Z]',value):
        return SourceTextEligibility(False,'numeric_or_symbol_literal')
    if re.search(r'[\u3400-\u9fff\u0400-\u04ff\u3040-\u30ff]',value):
        return SourceTextEligibility(False,'non_english_source')
    label=' '.join(value.replace('.','').split()).strip(':')
    # 完整英语字段标签可直接准入，
    # 字段和后随编号合并时不伪造子区域。
    # PART NO编号列表与图号均原样保留，
    # 不会把数字去掉后只翻译标签。
    if label in ENGLISH_FIELD_LABELS:
        return SourceTextEligibility(True,'exact_english_field_label')
    # A full part-number list is a literal identifier, even when OCR joins its
    # label with punctuation. The anchored grammar cannot swallow an adjacent
    # processing instruction or invent a smaller source span for the label.
    if re.fullmatch(r'PART\s*[- ]?\s*NO[.,:]?\s*\d[A-Z0-9._/-]*(?:\s*,\s*-?\d[A-Z0-9._/-]*)*',value):
        return SourceTextEligibility(False,'label_with_identifier_requires_separate_source_region')
    if re.match(r'^(?:DWG\.?|DRAWING)\s+NO\.?\b',value):
        return SourceTextEligibility(False,'label_with_identifier_requires_separate_source_region')
    # OCR may split DETAIL A into two immutable regions. The orphaned view
    # label is not sufficient evidence of prose, even without its identifier.
    # Full English instructions such as SEE DETAIL A still use the note route.
    # 孤立视图标题可能来自OCR拆框，
    # 没有正文证据不能视作加工说明。
    # SEE DETAIL A等完整说明仍有其它英语线索，
    # 不会因含DETAIL而一律拒绝。
    if value.rstrip('.') in {'VIEW','DETAIL','SECTION','SECT'}:
        return SourceTextEligibility(False,'standalone_view_reference_label')
    if re.fullmatch(r'(?:VIEW|DETAIL|SECTION|SECT\.?)\s+[A-Z0-9]+(?:\s*[-/]\s*[A-Z0-9]+)?',value):
        return SourceTextEligibility(False,'view_or_detail_reference')
    # 有限德语提示只用于保守拒绝，
    # 不是通用自动语言识别。
    # 纯工程标记、分区字母和首字母，
    # 分别保留明确拒绝原因。
    words=re.findall(r'[A-Z]+',value)
    if any(w in _NON_ENGLISH for w in words):
        return SourceTextEligibility(False,'non_english_drawing_language')
    if words and all(w in _ENGINEERING_MARKERS for w in words):
        return SourceTextEligibility(False,'dimension_unit_or_engineering_marker')
    if re.fullmatch(r'[A-Z]|[IVX]{2,}|[A-Z](?:\s*\.\s*[A-Z]){1,3}\.?',value):
        return SourceTextEligibility(False,'zone_or_initials')
    # 零件/图纸代码按完整形态识别，
    # 含数字的连串不按单词音译。
    # 功能词开头的混合指令要区别处理，
    # 避免把真实英语句误判成型号。
    if ((re.fullmatch(r'[A-Z]{1,5}\s*[-_]?\s*\d[\dA-Z ._/-]*',value) and words[0] not in _FUNCTION_WORDS)
        or (' ' not in value and re.search(r'\d',value) and re.fullmatch(r'[A-Z0-9#./_+\-]+',value))
        or re.fullmatch(r'[A-Z]{2,4}_[A-Z0-9_]+',value)):
        return SourceTextEligibility(False,'drawing_or_part_identifier')
    # Reuse the project's established English processing vocabulary as source
    # evidence, without changing any model, translation rules or knowledge pack.
    # 复用项目已有技术词汇作为源证据，
    # WHEN/AFTER/BEFORE粘连只提供路由线索。
    # 保留完整混合尺寸句送后续守卫，
    # 不能靠英文线索直接判整句专业正确。
    from .translation_targets import TECHNICAL_WORDS
    if any(w in TECHNICAL_WORDS or w in _ENGLISH_CUES
           or any(w.startswith(prefix) and w[len(prefix):] in TECHNICAL_WORDS|_ENGLISH_CUES
                  for prefix in ('WHEN','AFTER','BEFORE')) for w in words):
        return SourceTextEligibility(True,'english_note_or_processing_term')
    return SourceTextEligibility(False,'english_meaning_not_established')


def is_translatable_english(text: str) -> bool:
    return english_translation_eligibility(text).eligible


# 允许功能词随既有几何形成完整语义组，
# 保住被原生PDF拆成单词的否定与条件。
# 碎片资格不同于独立送译资格，
# 不能据True虚构单词对应的小图框。
def is_english_group_fragment(text: str) -> bool:
    """Grammar fragments may join via existing geometry; never route alone.

    Keeping DO/NOT/THE/THEN intact is essential to avoid deleting negation or
    conditions when native PDF text arrives as separate word regions. Pure
    units remain subject to the existing explicit native-preposition exception.
    """
    if is_translatable_english(text):return True
    words=re.findall('[A-Z]+',unicodedata.normalize('NFKC',text).upper())
    return bool(words and any(w in _FUNCTION_WORDS for w in words)
                and all(w in _FUNCTION_WORDS or w in _ENGINEERING_MARKERS for w in words))
