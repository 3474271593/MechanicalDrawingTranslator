# mechanical_drawing_translator/user_review_overlay.py
"""Read user-attested decisions without admitting them to the global lexicon."""
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
# 审核上下文限定历史实例范围，
# PDF全路径不是可复用基本文件名。
# page为一基页码而非内部page_index，
# bounds必须对应原审核对象。
class ReviewContext:
    # PDF identity is the complete historical path, not just a reusable filename.
    pdf: str
    page: int
    bounds: dict[str, float]
    instance_id: str = ''


# 只规范路径分隔符与大小写，
# 不把不同目录的同名PDF合并。
# 此历史overlay不做内容哈希身份，
# 不能把它误述为Reviewed94的内容绑定。
def _pdf(value):
    return str(value).replace('\\', '/').casefold()


class ReviewOverlay:
    def __init__(self, data):
        self.records = data['records']
        self.attestation = data['user_approval_attestation']
        # 索引按完整PDF身份和页号分桶，
        # 同页保留全部原审核行而非覆盖。
        # attestation保持用户声明原值，
        # 构造器本身不生成任何新批准。
        self._by_page = {}
        for r in self.records:
            self._by_page.setdefault((_pdf(r['pdf']), int(r['page'])), []).append(r)

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text('utf-8'))
        if data.get('schema') != 'MechanicalDrawingTranslator.user_approved_review_overlay.v2':
            raise ValueError('unsupported review overlay schema')
        # load入口校验schema、实例唯一性，
        # 状态只允许approved/pending/rejected/conflict。
        # 批准还须真实目标与指定来源声明，
        # 不凭一个approved单词制造审核证据。
        rows = data['records']
        if len({r['instance_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate review instance')
        for r in rows:
            if r['decision'] not in {'approved', 'pending', 'rejected', 'conflict'}:
                raise ValueError('invalid review status')
            if r['decision'] == 'approved' and (not r['accepted_targets_raw'] or
                    r['approval_provenance'] != 'user_attested_review_using_ai_reference'):
                raise ValueError('missing attested approval')
            # 四个边界键必须完整且无多余项，
            # 坐标须为有限可比较数值。
            # 这里只校验原审核序列化结构，
            # 服务还会核对当前真实条目范围。
            bounds = json.loads(r['coordinates'])
            if set(bounds) != {'min_x', 'min_y', 'max_x', 'max_y'} or not all(
                    math.isfinite(float(v)) for v in bounds.values()):
                raise ValueError('invalid review coordinates')
        return cls(data)

    def match(self, context: ReviewContext, original_source: str):
        matches = []
        # 实例ID若显式提供必须完全相同，
        # 源英文始终要求逐字一致。
        # 框误差仅限0.001的序列化舍入，
        # 不是相邻空间搜索或模糊OCR匹配。
        # 匹配返回多项时由候选层拒绝歧义。
        for r in self._by_page.get((_pdf(context.pdf), context.page), ()):
            if context.instance_id and context.instance_id != r['instance_id']:
                continue
            if original_source != r['source_text']:
                continue
            bounds = json.loads(r['coordinates'])
            # Only serialization rounding is tolerated; this is not spatial search.
            if set(bounds) != set(context.bounds) or any(
                    abs(float(bounds[k]) - float(context.bounds[k])) > .001 for k in bounds):
                continue
            matches.append(r)
        return tuple(matches)

    @staticmethod
    def translation_candidate(records):
        if not records:
            return None, 'no_instance_match'
        if len(records) != 1:
            return None, 'ambiguous_instance'
        # 必须唯一approved实例才可取中文，
        # safe_use_flags说明仅可作证据时不采用。
        # sentence_decision也须为approved，
        # 只批准术语不能借为整句批准。
        r = records[0]
        if r['decision'] != 'approved':
            return None, 'source_status_' + r['decision']
        if r['safe_use_flags']:
            return None, 'context_or_ocr_note_only'
        if r['sentence_decision'] != 'approved':
            return None, 'term_approval_only'
        # 送审术语必须覆盖完整源表达，
        # 允许有限末标点与大小写整理。
        # 目标含分隔、括注或换行视作未结构化，
        # 避免把多译法或说明直接贴到图纸。
        # 拒绝仍保留明确reason供服务展示。
        if r['term_source'].strip('., ').upper() != r['source_text'].strip('., ').upper():
            return None, 'term_within_larger_source'
        target = r['accepted_targets_raw']
        if any(c in target for c in '|／/（()）\n'):
            return None, 'unstructured_target'
        return target, 'complete_instance_target'
