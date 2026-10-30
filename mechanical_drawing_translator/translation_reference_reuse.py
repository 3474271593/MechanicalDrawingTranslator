# mechanical_drawing_translator/translation_reference_reuse.py
"""Reuse explicit, scoped phrase structure without importing reference answers.

Only existing glossary definitions and existing phrase resolutions are inputs.
New paragraphs stay machine compositions; no new whole-sentence approval occurs.
"""
from __future__ import annotations
from functools import lru_cache
import re
import unicodedata
from .translation_phrase_memory import PhraseEnhancement, PhraseEvidence

def _key(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC',value).upper().split())

def _literal_key(value: str) -> str:
    # Only optional terminal sentence punctuation is ignorable. The old memory
    # word tokenizer intentionally ignores operators; it cannot prove complete
    # source consumption for new structural composition.
    return _key(value).rstrip(' .;:')

# Reference只读取当前词表和记忆解析，
# 不导入参考PDF或评测中文答案。
# 索引只接收单一目标完整表达，
# 新段落仍是机器组合而非新批准。
class ReferencePhraseReuse:
    def __init__(self, glossary, memory):
        self.memory=memory
        aliases={};grades={}
        self.literal_sources=set()
        for phrase in glossary.reviewed_phrases():
            if len(phrase.targets)!=1:continue
            source=_key(phrase.source)
            self.literal_sources.add(_literal_key(phrase.source))
            # A genuine written acronym definition, not any parenthetical text.
            # 缩写必须有真实书写的英文全称，
            # 缩写字母逐一等于各词首字母。
            # 任意括号内容不自动构成缩写定义，
            # 也不能从常见行业知识补写未给全称。
            # NOTES编号与APPLY头已有完整译法，
            # TO范围也必须已有独立完整译法。
            # 两边同时成功才允许有限连接，
            # 不新建任意注释编号参数模板。
            match=re.fullmatch(r'([A-Z](?:\.[A-Z]){1,7}\.?)\s*\(([A-Z]+(?: [A-Z]+){1,7})\)',source)
            if match and match[1].replace('.','')==''.join(w[0] for w in match[2].split()):
                aliases.setdefault(_key(match[1]),[]).append(phrase)
            # Move only the very same explicit grade found in an existing
            # complete entry; never infer a translation for another grade.
            # N级记号仅移动同一个明确等级，
            # 例如括号级号前移到名词前。
            # 不会推导其它N数值的中文，
            # 源数值变化仍须原工程守卫保护。
            match=re.fullmatch(r'(.+?)\s*\((N\d{1,2})\)',source)
            if match:grades.setdefault(match[2]+' '+match[1],[]).append(phrase)
        self.aliases=aliases;self.grades=grades

    # 缩写与等级变体都要求唯一中文，
    # 普通叶子还须在完整字面源集合中。
    # 记忆score=1且不能是near_clause，
    # 词token相同不等于内部符号全部消费。
    def _leaf(self, source):
        key=_key(source)
        candidates=self.aliases.get(key,()) or self.grades.get(key,())
        if candidates:
            if len({p.targets[0] for p in candidates})!=1:return None
            p=candidates[0]
            return PhraseEvidence(source,p.targets[0],p.entry_id,p.origin,'reference_definition_variant')
        if _literal_key(source) not in self.literal_sources:return None
        r=self.memory.resolve_clause(source)
        if r is None or r.score!=1.0 or r.match_kind=='near_clause':return None
        return PhraseEvidence(source,r.translated_text,r.entry_id,r.origin,r.match_kind,r.knowledge_id)

    @staticmethod
    def _result(parts, separator):
        return PhraseEnhancement(separator.join(p.translated_segment.rstrip('。；') for p in parts),
            tuple(parts),sum(p.origin=='v1' for p in parts),0,
            sum(p.origin=='d7c' for p in parts),sum(p.origin=='d7d' for p in parts),sum(p.origin=='d7e' for p in parts))

    def resolve(self, source):
        if not isinstance(source,str) or not source.strip() or len(source)>2000:return None
        # 前导破折号只在完整APPLY说明脱离，
        # 原标记作为source证据原样保留。
        # 主体递归仍必须有全覆盖复用，
        # 不去掉标记后让任意英语交模型猜译。
        marker=re.fullmatch(r'([-–—])\s*(NOTES?\s+\d+(?:\s*&\s*\d+)?\s+APPLY\s+TO\s+[^\r\n]+)',source.strip(),re.I)
        if marker:
            body=self.resolve(marker[2])
            if body is not None:
                evidence=PhraseEvidence(marker[1],marker[1],None,'source','layout_marker_preserved')
                return PhraseEnhancement(marker[1]+' '+body.translated_text,(evidence,*body.evidence),
                    body.v1_segment_count,0,body.d7c_segment_count,body.d7d_segment_count,body.d7e_segment_count)
        key=_key(source)
        if key in self.aliases or key in self.grades:
            leaf=self._leaf(source)
            return self._result((leaf,),'') if leaf else None
        # This is a grammatical open head plus its complete prepositional scope.
        # Both sides must already be independently resolved, with original note
        # numbers; no new target words or global number template are invented.
        match=re.fullmatch(r'(NOTES?\s+\d+(?:\s*&\s*\d+)?\s+APPLY)\s+(TO\s+[^\r\n]+)',source.strip(),re.I)
        if match:
            left,right=self._leaf(match[1]),self._leaf(match[2])
            if left and right:return self._result((left,right),'')
        # 换行组合仅接收二到三十二行，
        # 空行可忽略，所有带字符行必须消费。
        # 不能跳过未知行只连接已知中文，
        # 原文条件或否定仍有完整来源边界。
        lines=tuple(line.strip() for line in source.splitlines() if line.strip())
        if not 2<=len(lines)<=32:return None

        @lru_cache(None)
        # 每个连续行组先查完整既有叶子，
        # 不是按单词逐项拼中文。
        # 每路径最多八叶子，
        # 候选过多时提前返回已有路径。
        # 最后仍要求译段序列唯一。
        # 歧义时返回None保持原英语安全回退。
        def paths(start):
            if start==len(lines):return ((),)
            found=[]
            # Join wrapped lines only if the entire joined source has a full
            # existing resolution. Every character-bearing line is consumed.
            for end in range(start+1,len(lines)+1):
                leaf=self._leaf(' '.join(lines[start:end]))
                if leaf is None:continue
                for tail in paths(end):
                    if len(tail)<8:
                        path=(leaf,*tail)
                        if path not in found:found.append(path)
                        if len(found)>16:return tuple(found)
            return tuple(found)
        options=[p for p in paths(0) if 2<=len(p)<=8]
        if not options:return None
        translations={tuple(p.translated_segment for p in path) for path in options}
        if len(translations)!=1:return None
        # 最终用分号保留多个独立断言，
        # 证据逐段带原entry与来源。
        # 服务仍核验完整源目标的数值和关系，
        # 此处的全覆盖不等于专业终审。
        return self._result(options[0],'；')
