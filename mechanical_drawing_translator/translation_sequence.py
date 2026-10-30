# mechanical_drawing_translator/translation_sequence.py
"""Attach explicit temporal relations to two complete reviewed imperatives."""
import re
from .translation_phrase_memory import PhraseEnhancement, PhraseEvidence


# 旧D7G只支持两个完整已审核命令，
# BEFORE/AFTER作为显式连接关系。
# 右侧禁止语不是已完成时间事件，
# 因此不得作时间锚来自动组合。
# 未知动作或非exact词表命中均拒绝。
def compose_sequence(source, glossary):
    # Do not detach numbers, negations or OCR text. Both operands must consume a
    # complete reviewed clause, not merely contain a recognizable noun/verb.
    parts=re.split(r'\s+(BEFORE|AFTER)\s+',source.strip().rstrip('.'),flags=re.I)
    if len(parts)!=3:return None
    left,relation,right=parts
    if re.match(r'DO\s+NOT\b',right,re.I):return None
    actions=r'(?:POLISH|GRIND|COAT|MACHINE|FABRICATE|DO NOT)\s+'
    if not all(re.match(actions,p,re.I) for p in (left,right)):return None
    resolutions=[glossary.resolve(p) for p in (left,right)]
    if any(r is None or r.route!='exact_rule' for r in resolutions):return None
    l,r=resolutions
    # 中文将右子句作为时间参照，
    # 左子句仍为真正需要执行的动作。
    # 不能因为译文换了位置就交换角色，
    # 两边来源entry ID分别留在evidence。
    # 新句只增加机器连接关系，
    # 并没有新增整句人工批准。
    marker='前' if relation.upper()=='BEFORE' else '后'
    translated=f'在{r.translated_text}之{marker}，{l.translated_text}'
    evidence=tuple(PhraseEvidence(p,s.translated_text,s.entry_id,'d7g',
                                 'reviewed_action_object',relation.upper())
                   for p,s in zip((left,right),resolutions))
    return PhraseEnhancement(translated,evidence,
                             sum(s.entry_id.startswith('mgt-') for s in resolutions),0)


# 仅归一开头明确工艺动名词形态，
# 不改句中其它拼写或工程字面。
# 完整叶子必须exact或parameterized，
# 不能凭句里含一个加工词就准入。
# 证据保留实际被消费的完整英文。
def _instruction_leaf(value,glossary):
    value=value.strip()
    inflections={'GRINDING':'GRIND','POLISHING':'POLISH','COATING':'COAT','MACHINING':'MACHINE'}
    first=re.match(r'[A-Za-z]+',value)
    if first and first[0].upper() in inflections:
        value=inflections[first[0].upper()]+value[first.end():]
    if not re.match(r'(?:POLISH|GRIND|COAT|MACHINE|FABRICATE|DO\s+NOT)\b',value,re.I):return None
    resolution=glossary.resolve(value)
    if resolution is None or resolution.route not in {'exact_rule','parameterized_rule'}:return None
    evidence=PhraseEvidence(value,resolution.translated_text,resolution.entry_id,'d7g-integrated',
                           'complete_reviewed_instruction_operand')
    return resolution.translated_text,(evidence,)


def instruction_scope_ambiguous(source,glossary):
    """A prohibition over an unmarked AND chain has no unique action scope."""
    text=source.strip().rstrip('.')
    # AND否定链要求明确局部范围，
    # 前面已见负向而后面又转正向时拒绝。
    # 不会任意把DO NOT作用到整条链，
    # 也不能只保留第一项的禁止。
    parts=re.split(r'\s+AND\s+',text,flags=re.I)
    if 2<=len(parts)<=8 and all(_instruction_leaf(p,glossary) for p in parts):
        negative_seen=False
        for part in parts:
            negative=bool(re.match(r'DO\s+NOT\b',part.strip(),re.I))
            if negative_seen and not negative:return True
            negative_seen|=negative
    prefix=re.fullmatch(r'(AFTER|BEFORE)\s+(.+?),\s*(.+)',text,re.I)
    suffix=None if prefix else re.fullmatch(r'(.+?)\s+(BEFORE|AFTER)\s+(.+)',text,re.I)
    reference=prefix[2] if prefix else suffix[3] if suffix else ''
    # A negative directive is not a completed event that can anchor BEFORE/AFTER.
    return bool(re.match(r'DO\s+NOT\b',reference,re.I) and _instruction_leaf(reference,glossary))


def compose_instructions(source, glossary, professional_knowledge=None):
    """Compose new whole instructions from complete audited action/object clauses.

    This extends the existing D7-G resolver, not an unrestricted translator.
    Every leaf must resolve in the existing glossary. Explicit connectors carry
    scope; unconsumed words, quantities, OR, reference marks and ambiguous nested
    conditions are rejected. Inflection changes only a leading processing verb.
    """
    # Integrated组合最长两千字符，
    # 引用星号、井号和换行保持拒绝边界。
    # 这些标记可能承担工程范围，
    # 不为得到中文而先清理掉它们。
    # 所有源歧义先于语法分解检查。
    if not isinstance(source,str) or len(source)>2000 or re.search(r'[*#\n\r]',source):return None
    text=source.strip().rstrip('.').strip()
    if instruction_scope_ambiguous(text,glossary):return None

    # 叶子中不得再嵌未定范围连接词，
    # 时间、条件、AND/OR须由外层控制。
    # 避免一个子句悄悄吞掉下一项限定，
    # 每个操作都需完整已审核译法。
    def simple(value):
        # Grammar operands cannot themselves contain unscoped operators. This
        # avoids attaching IF/UNLESS or BEFORE/AFTER to the wrong action.
        if re.search(r'\b(?:BEFORE|AFTER|UNLESS|IF|THEN|AND|OR)\b',value,re.I):return None
        return _instruction_leaf(value,glossary)

    # 外层按完整语法逐族尝试，
    # 前置时间、UNLESS、IF、后置时间，
    # 最后才是THEN或AND链。
    # 受控优先级不允许剩余词被忽略，
    # 任何分支都要两个或全部叶子成功。
    result=None
    # Explicit temporal prefixes: the first operand is the temporal reference,
    # the second is the governing action. Chinese keeps that same relationship.
    match=re.fullmatch(r'(AFTER|BEFORE)\s+(.+?),\s*(.+)',text,re.I)
    if match:
        right,left=simple(match[2]),simple(match[3])
        if right and left:
            marker='后' if match[1].upper()=='AFTER' else '前'
            result=(f'在{right[0]}之{marker}，{left[0]}',(*right[1],*left[1]))
    else:
        match=re.fullmatch(r'(.+?)\s+UNLESS\s+(?:OTHERWISE\s+)?SPECIFIED',text,re.I)
        if match:
            body=simple(match[1])
            if body:result=('除非另有规定，否则'+body[0],body[1])
        else:
            match=re.fullmatch(r'IF\s+SPECIFIED,\s*(.+)',text,re.I)
            if match:
                body=simple(match[1])
                if body:result=('若有规定，则'+body[0],body[1])
            else:
                temporal=re.fullmatch(r'(.+?)\s+(BEFORE|AFTER)\s+(.+)',text,re.I)
                if temporal:
                    left,right=simple(temporal[1]),simple(temporal[3])
                    if left and right:
                        marker='前' if temporal[2].upper()=='BEFORE' else '后'
                        result=(f'在{right[0]}之{marker}，{left[0]}',(*left[1],*right[1]))
                else:
                    # THEN最多八个顺序操作，
                    # AND最多八个并列操作。
                    # 所有叶子都存在才生成中文，
                    # 无法翻译的一项不会从组合消失。
                    # 输出先/再与并分别表示不同关系。
                    ordered=bool(re.search(r',?\s+THEN\s+',text,re.I))
                    parts=re.split(r',?\s+THEN\s+' if ordered else r'\s+AND\s+',text,flags=re.I)
                    if 2<=len(parts)<=8:
                        operands=[simple(p) for p in parts]
                        if all(operands):
                            targets=[p[0] for p in operands]
                            result=(('先'+'，再'.join(targets)) if ordered else '，并'.join(targets),
                                    tuple(e for p in operands for e in p[1]))
    if result is None:return None
    target,evidence=result
    return PhraseEnhancement(target,evidence,sum((e.entry_id or '').startswith('mgt-') for e in evidence),0)


def validate_instruction_relations(source, target, glossary):
    """Reject demonstrated relation errors within the controlled clause grammar.

    This is not a general Chinese semantic judge. Only complete audited operands
    and explicit, recognized Chinese forms participate. Unrecognized equivalents
    retain the existing gates rather than being declared wrong from keyword absence.
    """
    # 验证只针对能够完整组合的源语法，
    # 不支持的源返回空警告表示无覆盖。
    # 中文未出现受控等价形式，
    # 不能仅据词缺失说专业译文必错。
    # 原工程守卫仍需继续执行。
    if instruction_scope_ambiguous(source,glossary):return ('engineering_instruction_scope_ambiguous',)
    composition=compose_instructions(source,glossary)
    if composition is None:return ()
    compact=re.sub(r'\s+','',target)
    negation=r'(?:不得不|不能不|不允许|不要|不得|禁止|不可|不能|不许|勿|不)'
    cores=[];positions=[]
    # 否定绑定每个完整加工叶子，
    # “不得不/不能不”属于双重否定疑点。
    # 不能把目标中另一处禁止词移过来，
    # 同一动作正文的重复命中也不猜位置。
    for operand in composition.evidence:
        expected_negative=bool(re.match(r'DO\s+NOT\b',operand.source_segment,re.I))
        core=re.sub(r'^'+negation,'',re.sub(r'\s+','',operand.translated_segment))
        cores.append(core)
        matches=list(re.finditer(r'('+negation+r')?'+re.escape(core),compact))
        for match in matches:
            prefix=match[1] or ''
            if prefix in {'不得不','不能不'}:return ('engineering_instruction_polarity_unverified',)
            if bool(prefix)!=expected_negative:return ('engineering_instruction_polarity_changed',)
        positions.append(matches[0].start() if len(matches)==1 else None)
    upper=source.strip().rstrip('.').upper()
    if re.search(r'\bUNLESS\s+(?:OTHERWISE\s+)?SPECIFIED$',upper) and re.match(r'(?:若有规定|如果有规定)',compact):
        return ('engineering_instruction_condition_changed',)
    if upper.startswith('IF SPECIFIED,') and compact.startswith('除非'):
        return ('engineering_instruction_condition_changed',)
    # 条件反转与时间反转单独归因，
    # IF与UNLESS不能互相替代。
    # 只有明确先/再且操作位置唯一，
    # 才按出现顺序核对THEN时间链。
    # 没有可证实冲突不等于已证明正确。
    temporal=re.match(r'^(AFTER|BEFORE)\s+',upper)
    suffix=None if temporal else re.search(r'\s+(BEFORE|AFTER)\s+',upper)
    order=None
    if temporal or suffix:
        relation=(temporal or suffix)[1];reference=0 if temporal else 1
        order=([0,1] if relation=='AFTER' else [1,0]) if temporal else ([1,0] if relation=='AFTER' else [0,1])
        if reference<len(cores):
            marker=re.search(r'在(?:'+negation+r')?'+re.escape(cores[reference])+r'之?([前后])',compact)
            if marker and marker[1]!=('后' if relation=='AFTER' else '前'):
                return ('engineering_instruction_order_changed',)
    elif re.search(r'\bTHEN\b',upper):order=list(range(len(positions)))
    if order is not None and '先' in compact and '再' in compact and all(p is not None for p in positions) and len(set(positions))==len(positions):
        actual=sorted(range(len(positions)),key=positions.__getitem__)
        if actual!=order:return ('engineering_instruction_order_changed',)
    return ()
