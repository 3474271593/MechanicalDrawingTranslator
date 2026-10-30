# mechanical_drawing_translator/knowledge_v2_runtime.py
"""Portable compiled V2 data, with a caller-trusted release manifest fingerprint."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from .resource_paths import package_resource
from time import perf_counter

from .knowledge_v2 import (
    KnowledgeMatch, blocked_v1_refs, digest, expected_grant, knowledge_id,
    normalize, safety_warnings, validate_context,
)
from .translation_glossary import Glossary, GlossaryResolution
from .user_review_overlay import ReviewContext, ReviewOverlay

# 运行时包是可携带的受控派生资源，
# 离线读取不回访原开发目录。
# 清单指纹由调用者独立信任，
# 包内自报摘要不能替代发布pin。
RUNTIME_SCHEMA = 'MechanicalDrawingTranslator.knowledge_runtime.1'
RUNTIME_VERSION = 'MDT-KNOWLEDGE-2.0-runtime.1'
# A separate release pin is generated only by the explicit build CLI.
from .knowledge_v2_release import MANIFEST_SHA256


def default_runtime_root() -> Path:
    return package_resource('translation', 'knowledge_v2')


# off保持legacy缓存身份，
# shadow与candidate包含版本和指纹。
# 同页在不同知识配置下的结果，
# 不能因内容相似共用旧缓存。
def runtime_identity(mode: str, fingerprint: str = '') -> str:
    if mode not in {'off', 'shadow', 'candidate'}:
        raise ValueError('invalid knowledge mode')
    return 'legacy' if mode == 'off' else f'{RUNTIME_VERSION}:{mode}:{fingerprint}'


@dataclass(frozen=True)
# 使用记录区分候选与最终决策，
# 即使命中也可以只作shadow对照。
# knowledge_id与source_refs保留证据，
# reason描述采用、拒绝或策略挂起。
class KnowledgeUsage:
    mode: str
    fingerprint: str
    knowledge_id: str | None
    source_refs: tuple[str, ...]
    scope: str
    decision: str
    candidate: str | None
    reason: str


class RuntimeGlossary:
    """D6 remains first; migrated exacts have only the compiled V2 entry point."""
    def __init__(self, base: Glossary, runtime: 'RuntimeKnowledge'):
        self.base = base
        self.runtime = runtime
        self.version = base.version + '+' + runtime.fingerprint
        # 基础D6在精确解析中优先返回，
        # 迁移V1只由runtime.exact接入。
        # 组合词表用于算法与术语验证，
        # 不把辅助项变成新的精确整句。
        self._algorithms = Glossary.combine((base, runtime.algorithm_glossary))

    def resolve(self, source):
        result = self.base.resolve(source)
        if result is not None:
            return result
        candidate = self.runtime.exact(source)
        if candidate is None:
            return None
        ref = next(ref for ref in candidate.source_refs if ref.startswith('v1_raw:'))
        return GlossaryResolution(candidate.target, 'exact_rule', ref.split(':', 1)[1])

    def reviewed_phrases(self):
        return self._algorithms.reviewed_phrases()

    def validate_terms(self, *args, **kwargs):
        return self._algorithms.validate_terms(*args, **kwargs)


class RuntimeKnowledge:
    def __init__(self, payload: dict, fingerprint: str):
        self.fingerprint = fingerprint
        self.version = RUNTIME_VERSION
        self._records = {r['ref']: r for r in payload['source_records']}
        if len(self._records) != len(payload['source_records']):
            raise ValueError('duplicate runtime source identity')
        # 主记录仅来自V1与用户实例审核，
        # 源引用与知识ID必须分别唯一。
        # 建立索引前重算冲突挂起集合，
        # 不相信文件中已有的自动资格。
        primary = [r for r in self._records.values() if r['family'] in {'v1', 'review'}]
        self._knowledge = {k['id']: k for k in payload['knowledge']}
        if len(self._knowledge) != len(payload['knowledge']):
            raise ValueError('duplicate runtime knowledge identity')
        blocked = blocked_v1_refs(list(self._records.values()))
        seen = set()
        exacts = {}
        self._ref_to_knowledge = {}
        # 逐条核验ID与整串归一身份，
        # grant集合应与source_refs完全一致。
        # 每个原记录只能被消费一次，
        # 重复映射不能靠最终数量掩盖。
        # 目标、原状态与反向知识ID同行检查。
        for k in self._knowledge.values():
            if k['id'] != knowledge_id(k['source'], k['target']) or k['normalized_source'] != normalize(k['source']):
                raise ValueError('invalid runtime knowledge identity')
            if set(k['source_refs']) != {g['source_ref'] for g in k['grants']} or len(k['grants']) != len(k['source_refs']):
                raise ValueError('runtime grant/source mapping mismatch')
            for grant in k['grants']:
                ref = grant['source_ref']
                r = self._records[ref]
                raw = r['raw']
                if ref in seen or r['family'] not in {'v1', 'review'} or digest(raw) != r['raw_sha256']:
                    raise ValueError('invalid/duplicate runtime raw record')
                seen.add(ref)
                source, target = ((raw['source'], raw['target']) if r['family'] == 'v1'
                                  else (raw['source_text'], raw['accepted_targets_raw']))
                status = raw['decision'] if r['family'] == 'review' else raw['source_reference']
                if (normalize(source), target, status, r['knowledge_ids']) != (
                        k['normalized_source'], k['target'], r['original_status'], [k['id']]):
                    raise ValueError('runtime pair/status differs from source')
                # 原始记录重新推导授权与原因，
                # 运行时不能扩大离线构建器的范围。
                # 批准实例仍需保留provenance，
                # 只含approved字符串不足以信任。
                eligibility, reason = expected_grant(r, blocked)
                if (grant['eligibility'], grant['reason'], grant['original_status']) != (eligibility, reason, status):
                    raise ValueError('runtime authorization differs from source')
                if r['family'] == 'review':
                    validate_context(ReviewContext(raw['pdf'], int(raw['page']), json.loads(raw['coordinates']), raw['instance_id']))
                    if raw['decision'] == 'approved' and raw['approval_provenance'] != 'user_attested_review_using_ai_reference':
                        raise ValueError('runtime approval evidence missing')
                self._ref_to_knowledge[ref] = k
                if eligibility == 'auto_exact':
                    key = normalize(source)
                    if key in exacts and exacts[key]['target'] != target:
                        raise ValueError('ambiguous runtime exact meaning')
                    exacts[key] = k
        # 全部主来源必须有唯一知识映射，
        # 任何记录丢失都拒绝加载。
        # 审核实例ID与人工声明同时存在，
        # 避免空审核集被包装成已批准。
        if seen != {r['ref'] for r in primary}:
            raise ValueError('incomplete runtime source mapping')
        reviews = [r['raw'] for r in primary if r['family'] == 'review']
        if not payload['approval_attestation'] or len({r['instance_id'] for r in reviews}) != len(reviews):
            raise ValueError('runtime approval identity/attestation invalid')
        self.review_overlay = ReviewOverlay({'records': reviews,
            'user_approval_attestation': payload['approval_attestation']})
        # 策略挂起优先于自动精确词条，
        # 同一英文即使有auto授权也不执行。
        # 验证词表保留原V1安全ID，
        # 算法词表再排除禁止执行的表达。
        # 验证证据与执行准入是两个视图。
        self._policy = {normalize(row['source']): row for row in payload['runtime_policy']}
        self._exacts = {key: k for key,k in exacts.items() if key not in self._policy}
        safe = [r['raw'] for r in primary if r['family'] == 'v1' and r['historical_safety'] == 'safe']
        # Validation keeps original IDs needed by D7 evidence checks; execution
        # excludes the two unsafe literal expressions, while D7 overrides retain
        # their original, separately reviewed algorithm scope.
        def glossary(rows):
            return Glossary.from_payload({'schema_version': 1, 'glossary_version': RUNTIME_VERSION, 'entries': rows})
        self.validation_glossary = glossary(safe)
        deny = set(self._policy) | {normalize('GENERAL TOLERANCES (UNLESS SPECIFIED)')}
        self.algorithm_glossary = glossary([r for r in safe if normalize(r['source']) not in deny])
        # 派生计数与包内统计必须一致，
        # 这里只证明完整性而非质量达标。
        # 加载耗时与匹配耗时分别记录，
        # 不能把索引热命中说成模型推理。
        self.counts = {'primary_records': len(primary), 'knowledge': len(self._knowledge),
                      'source_auto_exact': len(exacts), 'runtime_auto_exact': len(self._exacts),
                      'review_records': len(reviews)}
        if self.counts != payload['counts']:
            raise ValueError('runtime source inventory/counts incomplete')
        self.load_seconds = 0.0
        self.match_seconds = 0.0
        self.match_calls = 0

    @classmethod
    def load(cls, root: Path, expected_digest: str) -> 'RuntimeKnowledge':
        started = perf_counter()
        try:
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise ValueError('a trusted runtime manifest SHA256 is required')
            # 先核对独立pin对应的manifest字节，
            # 再按固定runtime.json读取payload。
            # 清单键、版本、文件名均有限定，
            # 避免从包中解析任意资源位置。
            # 大小与摘要同时匹配后才解码。
            raw = (root / 'manifest.json').read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_digest:
                raise ValueError('runtime manifest fingerprint mismatch')
            manifest = json.loads(raw)
            if set(manifest) != {'schema', 'version', 'file', 'bytes', 'sha256', 'upstream_sha256'} or manifest['file'] != 'runtime.json':
                raise ValueError('invalid runtime manifest/file identity')
            if manifest['schema'] != RUNTIME_SCHEMA or manifest['version'] != RUNTIME_VERSION:
                raise ValueError('unsupported runtime schema/version')
            data = (root / 'runtime.json').read_bytes()
            if len(data) != manifest['bytes'] or hashlib.sha256(data).hexdigest() != manifest['sha256']:
                raise ValueError('runtime payload hash/size mismatch')
            payload = json.loads(data)
            if payload['schema'] != RUNTIME_SCHEMA or payload['version'] != RUNTIME_VERSION or payload['upstream_sha256'] != manifest['upstream_sha256']:
                raise ValueError('runtime payload identity mismatch')
            # payload版本及上游摘要继续核对，
            # 构造期间再验证源授权不变量。
            # 失败不会返回部分可用runtime，
            # 上层可明确报告资源配置错误。
            result = cls(payload, expected_digest)
            result.load_seconds = perf_counter() - started
            return result
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f'cannot load complete V2 runtime package: {exc}') from exc

    # 精确命中只读已经筛选的_exacts，
    # 返回参与自动资格的真实引用。
    # finally把未命中也计入调用统计，
    # 使诊断分母包含全部查找请求。
    def exact(self, source: str) -> KnowledgeMatch | None:
        started = perf_counter()
        try:
            k = self._exacts.get(normalize(source))
            if k is None:
                return None
            refs = tuple(g['source_ref'] for g in k['grants'] if g['eligibility'] == 'auto_exact')
            return KnowledgeMatch(k['id'], k['target'], refs, 'v1_exact')
        finally:
            self.match_seconds += perf_counter() - started
            self.match_calls += 1

    def policy_hold(self, source: str) -> str | None:
        # Same complete clauses remain unsafe when wrapped by the already
        # supported leading reference stars or sentence separators. This is a
        # rejection scope, never a fuzzy translation/matching alias.
        # 分句策略保留小数点内部数字，
        # 只剥离已支持的前导引用星号。
        # 这是扩大拒绝范围而非新增翻译别名，
        # 不能据此模糊匹配其它英文句。
        for clause in re.split(r'(?<!\d)[.;—]+(?!\d)', source):
            key = normalize(re.sub(r'^\s*\*{1,4}\s+', '', clause)).strip()
            if key in self._policy:
                return self._policy[key]['reason']
            if key == 'GENERAL TOLERANCES (UNLESS SPECIFIED)':
                return 'engineering_unless_condition_missing'
        return None

    def knowledge_for_ref(self, ref: str) -> dict | None:
        return self._ref_to_knowledge.get(ref)

    # 实例准入要求恰好一条匹配审核，
    # 其grant仍为historical_instance。
    # 此接口不会给approved实例，
    # 自动赋予PDF正式导出的资格。
    def review_eligible(self, records) -> bool:
        return len(records) == 1 and any(g['eligibility'] == 'historical_instance' and
            g['source_ref'] == 'review:' + records[0]['instance_id'] for g in
            self._ref_to_knowledge['review:' + records[0]['instance_id']]['grants'])

    def glossary(self, base: Glossary) -> RuntimeGlossary:
        return RuntimeGlossary(base, self)
