# mechanical_drawing_translator/knowledge_v2.py
"""Read-only, source-scoped V2 candidates. Deliberately unused by production."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from .engineering_source_risk import assess_source_risk
from .engineering_text import validate_engineering_semantics, validate_immutable_literals
from .user_review_overlay import ReviewContext, ReviewOverlay

# 此文件提供源证据与候选校验，
# 生产读取采用编译后的runtime。
# 不能把离线候选库构建器，
# 误认成默认已启用翻译链路。
# V2授权保留V1与实例两种范围，
# 辅助来源不能自行晋升为词条。
SCHEMA = 'MechanicalDrawingTranslator.translation_knowledge.2.0'
VERSION = 'MDT-KNOWLEDGE-2.0'
AUTO = frozenset({'auto_exact', 'historical_instance'})


# 规范JSON固定键序、缩进与末换行，
# 使同一结构产生稳定字节摘要。
# 中文原字面保留而非ASCII转义，
# 哈希用于证据身份而非人工签名。
def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


# 英文仅做整串兼容归一，
# 不存在模糊拼写修复或子串命中。
# 知识ID同时绑定英文与准确中文，
# 目标空白、标点也可能改变身份。
# ID截断用于稳定识别，
# 完整文件真实性仍由清单校验。
def normalize(text: str) -> str:
    """Same full-string normalization as the current Glossary; no fuzzy repair."""
    return ' '.join(unicodedata.normalize('NFKC', text).upper().split())


def knowledge_id(source: str, target: str) -> str:
    # Chinese whitespace/punctuation can carry engineering meaning: keep it exact.
    return 'MDTK2-' + digest([normalize(source), target])[:24]


def read_source_manifest(manifest_path: Path) -> tuple[list[dict], dict]:
    """A separate manifest and input bytes are the local trust boundary."""
    # 清单是独立提供的信任根，
    # 不能仅相信候选包自报来源。
    # 各来源按source_id稳定读取，
    # 同一role重复会立刻拒绝。
    # 先比输入字节SHA256再解码，
    # 防止替换原件后继承旧授权。
    manifest = json.loads(manifest_path.read_text('utf-8'))
    if manifest['version'] != VERSION:
        raise ValueError('manifest version mismatch')
    sources, documents = [], {}
    for spec in sorted(manifest['sources'], key=lambda s: s['source_id']):
        if spec['role'] in documents:
            raise ValueError('duplicate source role')
        raw = (manifest_path.parent / spec['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError(f'input hash mismatch: {spec["source_id"]}')
        documents[spec['role']] = ({'text': raw.decode('utf-8')} if spec.get('format') == 'text'
                                   else json.loads(raw.decode('utf-8')))
        sources.append({k: spec[k] for k in ('source_id', 'role', 'sha256', 'original_path')})
    return sources, documents


def source_catalog(documents: dict) -> dict:
    """Enumerate every authoritative raw record, independently of bundle claims."""
    # 从原始文档重新枚举来源全集，
    # 而非按候选包声称的记录计数。
    # 引用包含role、section与原ID，
    # 避免跨资源相同编号相互覆盖。
    # 清理日志、历史映射只保存旁证，
    # 不能因被打包就获得自动翻译资格。
    result = {}
    def add(role, raw, identifier, family, section=''):
        ref = f'{role}:{section + ":" if section else ""}{identifier}'
        if ref in result:
            raise ValueError('duplicate original source ref')
        result[ref] = (role, raw, str(identifier), family, section)
    for role, family, key, id_key in (
        ('v1_raw','v1','entries','id'), ('review','review','records','instance_id'),
        ('legacy_raw','legacy_aux',None,'record_id'),
        ('cleanup_valid_candidates','legacy_aux',None,'record_id'),
        ('cleanup_pairing_concerns','legacy_aux',None,'record_id'),
        ('cleanup_invalid_records','legacy_aux',None,'record_id'),
        ('cleanup_legacy_mapping','legacy_aux',None,'old_record_id'),
        ('cleanup_conflicts_reclassified','legacy_aux',None,'conflict_group')):
        rows = documents[role][key] if key else documents[role]
        for raw in rows:
            add(role, raw, raw[id_key], family)
    # 不同历史资源保持各自章节范围，
    # 未提供ID时使用章内稳定序号。
    # 审核日志单列辅助来源族，
    # 不会覆盖原始approved/pending。
    sections = {'mechanical_glossary': ('entries',),
        'mechanical_domain_knowledge_d7c': ('reuse_overrides','reviewed_phrases','templates'),
        'mechanical_generalization_d7d': ('term_aliases','templates'),
        'mechanical_professional_d7e': ('terms',), 'mechanical_knowledge_d8b': ('candidates',)}
    for role, names in sections.items():
        for section in names:
            for i, raw in enumerate(documents[role][section]):
                add(role, raw, raw.get('id', str(i + 1)), 'runtime_aux', section)
    for role in ('legacy_valid_review_journal','legacy_concerns_review_journal'):
        for identifier, raw in documents.get(role, {}).get('reviews', {}).items():
            add(role, raw, identifier, 'review_journal_aux')
    return result


def validate_context(context: ReviewContext) -> None:
    # 实例授权必须绑定文档与正页码，
    # 坐标键集合必须完整且没有扩展项。
    # 拒绝布尔值、无穷大与NaN坐标，
    # 因为它们无法表达可信矩形。
    # 左右或上下颠倒也视为结构错误，
    # 不替用户静默调整授权边界。
    bounds = context.bounds
    if not isinstance(context.pdf, str) or not context.pdf.strip() or type(context.page) is not int or context.page < 1:
        raise ValueError('invalid context document/page')
    keys = {'min_x','min_y','max_x','max_y'}
    if not isinstance(bounds, dict) or set(bounds) != keys or any(
            type(v) not in (int,float) or not math.isfinite(v) for v in bounds.values()):
        raise ValueError('invalid/nonfinite context coordinates')
    if bounds['min_x'] > bounds['max_x'] or bounds['min_y'] > bounds['max_y']:
        raise ValueError('inverted context coordinates')
    if not isinstance(context.instance_id, str):
        raise ValueError('invalid instance identity')


# 知识准入仍运行工程三重守卫：
# 字面量结构、方向条件、源风险。
# 编号字段特例仅在语义守卫开放，
# 不会跳过数值与源OCR风险。
# 同警告按首次出现顺序去重。
def safety_warnings(source: str, target: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((
        *validate_immutable_literals(source, target).warning_codes,
        *validate_engineering_semantics(source, target, allow_numbering_labels=True).warning_codes,
        *assess_source_risk(source).warning_codes,
    )))


def blocked_v1_refs(records: list[dict]) -> set[str]:
    """Keep both V1 and later reuse-override meanings; do not adjudicate globally."""
    # V1与D7覆盖译法冲突时先挂起，
    # 不以资源加载顺序作最终裁决。
    # 保留两种来源的原始意思，
    # 待有明确范围时分别使用。
    v1 = {r['raw']['id']: r for r in records if r['family'] == 'v1'}
    blocked = set()
    for r in records:
        if r.get('section') == 'reuse_overrides':
            parent = v1.get(r['raw'].get('source_entry_id'))
            if parent and parent['raw']['target'] != r['raw'].get('target'):
                blocked.add(parent['ref'])
    return blocked


def expected_grant(record: dict, blocked: set[str]) -> tuple[str, str]:
    # 授权由原始来源重新计算，
    # 不接受包内任意填写的eligibility。
    # V1隔离项优先于精确匹配，
    # 原始安全也还要经当前工程检查。
    # 实例批准不能扩大成全局短语，
    # 带说明的证据可能只作evidence。
    raw = record['raw']
    if record['family'] == 'v1':
        if record['historical_safety'] != 'safe':
            return 'isolated', 'original_v1_isolation'
        if record['ref'] in blocked:
            return 'conflict_hold', 'v1_vs_d7_reuse_override_requires_scope_review'
        warnings = safety_warnings(raw['source'], raw['target'])
        return ('risk_hold', ','.join(warnings)) if warnings else ('auto_exact', 'inherit_v1_exact_scope')
    if record['family'] == 'review':
        target, reason = ReviewOverlay.translation_candidate((raw,))
        if raw['decision'] != 'approved':
            return raw['decision'], reason
        if target is None:
            return 'evidence_only', reason
        warnings = safety_warnings(raw['source_text'], target)
        return ('risk_hold', ','.join(warnings)) if warnings else ('historical_instance', reason)
    return 'auxiliary_only', 'not_a_primary_v2_admission_source'


def validate_bundle(bundle: dict, *, manifest_path: Path | None = None) -> None:
    """Reject dangling provenance, forged status/targets and broadened grants."""
    # 验证按来源、授权、反向映射逐层进行，
    # 任一断链都拒绝整个候选包。
    # 不能只检查知识总数与目标文字，
    # 否则可能丢失原始隔离或待审记录。
    # 异常归一为ValueError便于上层处理，
    # 不会降级为一个部分可信资源。
    try:
        if manifest_path is None:
            raise ValueError('trusted source manifest is required')
        trusted_sources, documents = read_source_manifest(manifest_path)
        if bundle['sources'] != trusted_sources:
            raise ValueError('source manifest differs from trusted source inventory')
        if bundle['approval_attestation'] != documents['review']['user_approval_attestation'] or bundle['review_basis'] != documents['review']['review_basis']:
            raise ValueError('approval attestation/review basis differs from original')
        if bundle['schema'] != SCHEMA or bundle['version'] != VERSION:
            raise ValueError('unsupported V2 schema/version')
        records = bundle['source_records']
        refs = {r['ref']: r for r in records}
        if len(refs) != len(records):
            raise ValueError('duplicate source ref')
        # 候选包必须覆盖原始记录全集，
        # 遗漏和重复都属于不可接受变更。
        # 人工声明与review_basis保持原值，
        # 不补写原件里没有的审核人证据。
        catalog = source_catalog(documents)
        if refs.keys() != catalog.keys():
            raise ValueError('incomplete original source record inventory')
        sources = {s['source_id']: s for s in bundle['sources']}
        if len(sources) != len(bundle['sources']):
            raise ValueError('duplicate source identity')
        # 每条记录的原始内容、身份、家族，
        # 以及章节均与独立目录逐项核对。
        # V1安全状态由safe原件重新推导，
        # 不能由打包结果自行解除隔离。
        # raw摘要与来源文件摘要分层校验。
        for r in records:
            role, original, identifier, family, section = catalog[r['ref']]
            if (r['source_id'], r['raw'], r['record_id'], r['family'], r['section']) != (role, original, identifier, family, section):
                raise ValueError('source record differs from trusted original')
            if family == 'v1' and r['historical_safety'] != ('safe' if identifier in
                    {s['id'] for s in documents['v1_safe']['entries']} else 'isolated'):
                raise ValueError('historical V1 safety status changed')
            if r['source_id'] not in sources or digest(r['raw']) != r['raw_sha256']:
                raise ValueError('invalid source/raw hash')
            if r['source_sha256'] != sources[r['source_id']]['sha256']:
                raise ValueError('source hash does not match manifest')
            raw = r['raw']
            expected_status = (raw['decision'] if r['family'] in {'review', 'review_journal_aux'} else
                               raw.get('review_status', raw.get('source_reference', 'not_individually_attested')))
            if r['original_status'] != expected_status:
                raise ValueError('original approval status was changed')
            if r['family'] == 'review':
                validate_context(ReviewContext(raw['pdf'], int(raw['page']), json.loads(raw['coordinates']), raw['instance_id']))
                ReviewOverlay({'records': [raw], 'user_approval_attestation': bundle['approval_attestation']})
                if raw['decision'] not in {'approved', 'pending', 'rejected', 'conflict'}:
                    raise ValueError('invalid original decision')
                if raw['decision'] == 'approved' and raw['approval_provenance'] != 'user_attested_review_using_ai_reference':
                    raise ValueError('missing approval provenance')
        blocked = blocked_v1_refs(records)
        # 知识ID不得重复或与英中对不符，
        # 每个grant必须恰好关联一个来源。
        # 只允许V1与原审核实例提供准入，
        # 辅助词表本身不授予整句授权。
        # 源串归一相同但译文不同仍分开，
        # 避免把冲突中文合并成同一答案。
        kids = set()
        mapped = defaultdict(list)
        for k in bundle['knowledge']:
            if k['id'] in kids or k['id'] != knowledge_id(k['source'], k['target']):
                raise ValueError('duplicate/unstable knowledge id or target changed')
            kids.add(k['id'])
            if k['normalized_source'] != normalize(k['source']):
                raise ValueError('invalid normalization')
            if not k['source_refs'] or len(set(k['source_refs'])) != len(k['source_refs']):
                raise ValueError('empty/duplicate provenance')
            if set(k['source_refs']) != {g['source_ref'] for g in k['grants']} or len(k['grants']) != len(k['source_refs']):
                raise ValueError('grant/source mapping mismatch')
            for g in k['grants']:
                r = refs[g['source_ref']]
                raw = r['raw']
                if r['family'] not in {'v1', 'review'}:
                    raise ValueError('auxiliary source cannot admit translation')
                source = raw['source'] if r['family'] == 'v1' else raw['source_text']
                target = raw['target'] if r['family'] == 'v1' else raw['accepted_targets_raw']
                if normalize(source) != k['normalized_source'] or target != k['target']:
                    raise ValueError('knowledge differs from original pair')
                expected, reason = expected_grant(r, blocked)
                if (g['eligibility'], g['reason']) != (expected, reason):
                    raise ValueError('unsafe/broadened eligibility')
                if g['original_status'] != r['original_status']:
                    raise ValueError('grant approval status changed')
                if g['match_type'] != ('normalized_full_exact' if r['family'] == 'v1' else 'original_instance_exact'):
                    raise ValueError('match scope changed')
                # 历史实例范围完整保留PDF、页码，
                # 原框、实例ID和原送审英文。
                # 这些上下文不得被迁移时丢弃，
                # 否则局部授权会错误扩大成全局。
                context = ({'pdf': raw['pdf'], 'page': int(raw['page']),
                            'bounds': json.loads(raw['coordinates']), 'instance_id': raw['instance_id'],
                            'original_source': raw['source_text']} if r['family'] == 'review' else None)
                if g['context'] != context:
                    raise ValueError('instance context changed')
                mapped[r['ref']].append(k['id'])
            if set(k['conflict_ids']) - {c['id'] for c in bundle['conflicts']}:
                raise ValueError('dangling conflict')
        for r in records:
            if sorted(mapped[r['ref']]) != sorted(r['knowledge_ids']):
                raise ValueError('incomplete reverse provenance map')
            if r['family'] in {'v1', 'review'} and len(mapped[r['ref']]) != 1:
                raise ValueError('primary record was lost or duplicated')
        # 冲突项仍必须指向真实原始记录，
        # 不能以空来源制造已处理状态。
        # 已知V1/D7冲突要求显式未决，
        # 并保持blocks_auto阻止自动准入。
        for c in bundle['conflicts']:
            if not c['source_refs'] or set(c['source_refs']) - refs.keys():
                raise ValueError('conflict lost sources')
        for r in records:
            if r['ref'] in blocked and not any(c['status'] == 'unresolved' and
                    r['ref'] in c['source_refs'] and c['blocks_auto'] for c in bundle['conflicts']):
                raise ValueError('known runtime conflict silently removed')
    except (KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid V2 structure: {exc}') from exc


@dataclass(frozen=True)
# 匹配结果默认仍标machine，
# 它表达候选来源而非导出许可。
# scope区分整串V1与历史实例，
# source_refs供详情与审计追溯。
class KnowledgeMatch:
    knowledge_id: str
    target: str
    source_refs: tuple[str, ...]
    scope: str
    provenance: str = 'machine'
    warnings: tuple[str, ...] = ('v2_shadow_candidate_requires_existing_product_gates',)


# 此快照类用于候选包完整验证与解析，
# 并非product默认实例化的入口。
# 本模块的摘要、授权与上下文校验函数，
# 则由knowledge_v2_runtime正式复用。
# “快照类未默认使用”不能解释为，
# 整个模块都与生产运行无关。
class KnowledgeV2:
    """Snapshot loader. No setters, model invocation, PDF I/O, or service wiring."""
    def __init__(self, bundle: dict, *, manifest_path: Path | None = None):
        validate_bundle(bundle, manifest_path=manifest_path)
        # 通过验证后深复制为内部快照，
        # 调用者改原字典不应改变当前索引。
        # 英文按归一整串建立多值索引，
        # 保留同源不同义的潜在冲突。
        data = deepcopy(bundle)
        self._by_source = defaultdict(list)
        self._reviews = {}
        for k in data['knowledge']:
            self._by_source[k['normalized_source']].append(k)
        for r in data['source_records']:
            if r['family'] == 'review':
                self._reviews[r['ref']] = r['raw']
        self._overlay = ReviewOverlay({'records': list(self._reviews.values()),
                                      'user_approval_attestation': data['approval_attestation']})

    @classmethod
    def load(cls, path: Path, *, manifest_path: Path | None = None) -> 'KnowledgeV2':
        return cls(json.loads(path.read_text('utf-8')), manifest_path=manifest_path or path.parent / 'inputs/manifest.json')

    def resolve(self, source: str, context: ReviewContext | None = None) -> KnowledgeMatch | None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError('source must be a non-empty string')
        if context is not None:
            validate_context(context)
        items = self._by_source.get(normalize(source), ())
        # 全局精确候选先检查唯一中文含义，
        # 多义时不选第一个答案。
        # 实际调用原文还需重新跑安全守卫，
        # 归一命中不取消工程字面检查。
        auto = [(k, g) for k in items for g in k['grants'] if g['eligibility'] == 'auto_exact']
        if auto:
            # Never silently choose the first of competing meanings.
            if len({k['target'] for k, _ in auto}) != 1:
                return None
            k = auto[0][0]
            if safety_warnings(source, k['target']):
                return None
            return KnowledgeMatch(k['id'], k['target'], tuple(g['source_ref'] for _, g in auto), 'v1_exact')
        if context is None:
            return None
        # 没有实例上下文不能使用局部批准，
        # overlay再核对原页、原框与字面。
        # 唯一候选还须找到historical grant，
        # 审核状态相同也不能跨实例借用。
        matched = self._overlay.match(context, source)
        target, _ = self._overlay.translation_candidate(matched)
        if target is None or safety_warnings(source, target):
            return None
        allowed = [(k, g) for k in items for g in k['grants']
                   if g['eligibility'] == 'historical_instance' and
                   self._reviews.get(g['source_ref'], {}).get('instance_id') == matched[0]['instance_id']]
        if len(allowed) != 1:
            return None
        k, g = allowed[0]
        return KnowledgeMatch(k['id'], target, (g['source_ref'],), 'historical_instance')
