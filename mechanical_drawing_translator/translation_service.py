# mechanical_drawing_translator/translation_service.py
"""Glossary-first, safety-validated local translation orchestration."""

from __future__ import annotations

import re
import threading
from contextvars import ContextVar
from functools import wraps
from dataclasses import dataclass, replace
from pathlib import Path
from .resource_paths import package_resource
from typing import Literal, Sequence
from time import perf_counter

from .engineering_text import is_numbering_label, validate_engineering_semantics, validate_immutable_literals
from .translation_domain_knowledge import (
    DomainKnowledge,
    default_d7c_knowledge_path,
    default_v1_isolation_path,
)
from .translation_backend import (
    CTranslate2Backend,
    Generation,
    TranslationBackend,
    TranslationInferenceError,
)
from .translation_glossary import Glossary, default_glossary_path, default_v1_glossary_path
from .translation_generalization import (
    GeneralizationKnowledge,
    default_d7d_knowledge_path,
)
from .translation_professional import (
    ProfessionalKnowledge,
    default_d7e_knowledge_path,
)
from .translation_phrase_memory import PhraseEnhancement, PhraseEvidence, PhraseMemory
from .translation_results import TranslationResult, validate_translation_results
from .translation_targets import SelectionResult, TargetEntry
from .translation_fields import resolve_field, validate_field_terms, UNRESOLVED_ABBREVIATIONS
from .translation_sequence import compose_sequence, compose_instructions, validate_instruction_relations, instruction_scope_ambiguous
from .translation_d8a import resolve_contextual_translation, resolve_page_context
from .translation_d8b import D8BKnowledge, default_d8b_knowledge_path, resolve_expression
from .engineering_source_risk import assess_source_risk
from .user_review_overlay import ReviewContext, ReviewOverlay
from .reviewed94 import load_packaged as load_reviewed94


# 服务承担整条离线送译编排，
# 规则、记忆、组合与模型路线分开记录。
# fallback保持原英文并给不可用标记，
# source_preserved是明确保留源文，
# 二者不能混成模型成功翻译。
# 任何路线均不自动获得人工来源。
TranslationRoute = Literal[
    "exact_rule", "parameterized_rule", "hybrid_memory", "model", "fallback", "source_preserved",
]


@dataclass(frozen=True)
# 轻量详情保存译文、警告与路线，
# evidence解释使用了哪些语言片段。
# knowledge_trace记录V2候选的决策，
# 不会把审计候选直接视作最终结果。
# EntryTranslation再绑定坐标契约结果，
# 方便UI展示详情而不改写原OCR。
class TextTranslation:
    translated_text: str
    warnings: tuple[str, ...]
    route: TranslationRoute
    evidence: tuple[PhraseEvidence, ...] = ()
    knowledge_trace: tuple = ()


@dataclass(frozen=True)
class EntryTranslation:
    """One contract result paired with its auditable translation route."""

    result: TranslationResult
    detail: TextTranslation


@dataclass(frozen=True)
# 配置只描述资源位置与显式能力，
# 所有试验增强在字段层默认关闭。
# 模型输入上限按实际token执行，
# 不是按英语单词或字符数截断。
# 知识off/shadow/candidate各有边界，
# 后续构造器检查父能力依赖。
class TranslationConfig:
    model_root: Path
    glossary_path: Path
    max_input_tokens: int = 512
    supplemental_glossary_paths: tuple[Path, ...] = ()
    enable_supplemental_glossaries: bool = False
    enable_phrase_enhancement: bool = False
    enable_d7c_enhancement: bool = False
    enable_d7d_enhancement: bool = False
    enable_d7e_enhancement: bool = False
    enable_marked_rules: bool = False
    enable_d7f: bool = False
    enable_d7g: bool = False
    enable_d8a: bool = False
    enable_d8b: bool = False
    enable_engineering_symbol_risk: bool = False
    enable_d8c: bool = False
    enable_d8c_ocr_verification: bool = False
    review_overlay_path: Path | None = None
    knowledge_mode: str = 'off'
    knowledge_root: Path | None = None
    knowledge_manifest_sha256: str | None = None
    # 质量候选开关不是质量验收结论，
    # 启用后仍须各层工程守卫。
    # Reviewed94只加载独立增量授权，
    # 不会重写旧V1或全部审核状态。
    # Integrated、Professional与Reference，
    # 依照明确父配置逐层增添能力。
    # Convergence与ArgosCore也不替换模型，
    # 而是收敛可全覆盖的已知源表达。
    enable_v2_quality_optimization: bool = False
    enable_reviewed94: bool = False
    enable_integrated_quality: bool = False
    enable_professional_quality: bool = False
    enable_reference_quality: bool = False
    enable_convergence_quality: bool = False
    enable_argos_core: bool = False
    enable_final_optimization: bool = False
    enable_product_core: bool = False
    enable_semantic_edit: bool = False

    @classmethod
    def packaged(
        cls, *, enable_v1: bool = True, enable_phrase_enhancement: bool = False,
        enable_d7c_enhancement: bool = False,
        enable_d7d_enhancement: bool = False,
        enable_d7e_enhancement: bool = False,
        enable_marked_rules: bool = False,
        enable_d7f: bool = False,
        enable_d7g: bool = False,
        enable_d8a: bool = False,
        enable_d8b: bool = False,
        enable_engineering_symbol_risk: bool = False,
        enable_d8c: bool = False,
        enable_d8c_ocr_verification: bool = False,
        review_overlay_path: Path | None = None,
        knowledge_mode: str = 'off', knowledge_root: Path | None = None,
        knowledge_manifest_sha256: str | None = None,
        enable_v2_quality_optimization: bool = False,
        enable_reviewed94: bool = False,
        enable_integrated_quality: bool = False,
        enable_professional_quality: bool = False,
        enable_reference_quality: bool = False,
        enable_convergence_quality: bool = False,
        enable_argos_core: bool = False,
        enable_final_optimization: bool = False,
        enable_product_core: bool = False,
        enable_semantic_edit: bool = False,
    ) -> "TranslationConfig":
        # 默认路径相对包文件构建，
        # 不依赖开发机工作目录或用户桌面。
        # 仅组合资源路径与开关参数，
        # 不会此时下载模型或执行推理。
        # V1补充默认路径仍指safe资源，
        # 隔离资源不作为普通词库追加。
        resource_root = package_resource()
        return cls(
            resource_root / "models" / "translation" / "argos-en-zh-1.9",
            default_glossary_path(),
            supplemental_glossary_paths=(default_v1_glossary_path(),),
            enable_supplemental_glossaries=enable_v1,
            enable_phrase_enhancement=enable_phrase_enhancement,
            enable_d7c_enhancement=enable_d7c_enhancement,
            enable_d7d_enhancement=enable_d7d_enhancement,
            enable_d7e_enhancement=enable_d7e_enhancement,
            enable_marked_rules=enable_marked_rules,
            enable_d7f=enable_d7f,
            enable_d7g=enable_d7g,
            enable_d8a=enable_d8a,
            enable_d8b=enable_d8b,
            enable_engineering_symbol_risk=enable_engineering_symbol_risk,
            enable_d8c=enable_d8c,
            enable_d8c_ocr_verification=enable_d8c_ocr_verification,
            review_overlay_path=review_overlay_path,
            knowledge_mode=knowledge_mode, knowledge_root=knowledge_root,
            knowledge_manifest_sha256=knowledge_manifest_sha256,
            enable_v2_quality_optimization=enable_v2_quality_optimization,
            enable_reviewed94=enable_reviewed94,
            enable_integrated_quality=enable_integrated_quality,
            enable_professional_quality=enable_professional_quality,
            enable_reference_quality=enable_reference_quality,
            enable_convergence_quality=enable_convergence_quality,
            enable_argos_core=enable_argos_core,
            enable_final_optimization=enable_final_optimization,
            enable_product_core=enable_product_core,
            enable_semantic_edit=enable_semantic_edit,
        )


# 可用性按显式unavailable警告判断，
# 中文存在不等同于专业正确。
# source_preserved也可能返回原文，
# 调用者应结合route理解含义。
# 不能用是否含汉字代替此契约。
def translation_is_available(result: TranslationResult | TextTranslation) -> bool:
    return not any(
        warning == "translation_unavailable" or warning.startswith("translation_unavailable:")
        for warning in result.warnings
    )


# 回退值始终以本次源英文为正文，
# 原因按首次出现顺序去重。
# 保留统一translation_unavailable，
# 让预览与导出能够稳定识别失败。
# 回退不补写缺失的OCR内容，
# 也不借其它句子的中文顶替。
def _fallback(source: str, *causes: str) -> TextTranslation:
    unique = tuple(dict.fromkeys(causes))
    return TextTranslation(source, ("translation_unavailable", *unique), "fallback")


# 去空白只用于检测异常重复，
# 实际返回译文不被此函数改写。
# 足够长且短片段连续重复五次，
# 才判作模型退化或记忆组合异常。
# 这是输出健康检查而非专业语义评分。
def _has_excessive_repetition(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return len(compact) >= 20 and re.search(r"(.{2,12})\1{4,}", compact) is not None


# 缓存只覆盖一次顶层调用，
# 上下文变量隔离不同请求执行。
# 递归子请求复用当前纯校验结果，
# 结束后finally恢复上一级上下文。
# 不跨页面保存句子翻译结果，
# 也不缓存模型推理或用户批准。
# 因此不会用旧页检查污染新页。
def _validation_request(method):
    """Deduplicate pure gates within one request; never retain page results."""
    @wraps(method)
    def run(self, *args, **kwargs):
        if not (self.enable_v2_quality_optimization or self.reviewed94) or self._validation_cache.get() is not None:
            return method(self, *args, **kwargs)
        token = self._validation_cache.set({})
        try:
            return method(self, *args, **kwargs)
        finally:
            self._validation_cache.reset(token)
    return run


class TranslationService:
    def __init__(
        self,
        backend: TranslationBackend,
        glossary: Glossary,
        *,
        max_input_tokens: int = 512,
        phrase_memory: PhraseMemory | None = None,
        validate_hybrid_terms: bool = False,
        generalization_knowledge: GeneralizationKnowledge | None = None,
        enforce_extended_semantics: bool = False,
        professional_knowledge: ProfessionalKnowledge | None = None,
        enable_marked_rules: bool = False,
        enable_d7f: bool = False,
        enable_d7g: bool = False,
        enable_d8a: bool = False,
        enable_d8b: bool = False,
        d8b_knowledge: D8BKnowledge | None = None,
        enable_engineering_symbol_risk: bool = False,
        enable_d8c: bool = False,
        enable_d8c_ocr_verification: bool = False,
        review_overlay: ReviewOverlay | None = None,
        knowledge_mode: str = 'off', knowledge_runtime=None,
        enable_v2_quality_optimization: bool = False,
        reviewed94=None,
        enable_integrated_quality: bool = False,
        enable_professional_quality: bool = False,
        enable_reference_quality: bool = False,
        enable_convergence_quality: bool = False,
        enable_argos_core: bool = False,
        enable_final_optimization: bool = False,
        enable_product_core: bool = False,
        enable_semantic_edit: bool = False,
    ) -> None:
        if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int) \
                or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        self.backend = backend
        self.glossary = glossary
        self.max_input_tokens = max_input_tokens
        self.phrase_memory = phrase_memory
        # 部分高级能力强制启用完整术语检查，
        # 防止组合路径跳过词义保护。
        # 扩展语义条件随Integrated/Professional，
        # 包含前后顺序与除非等高风险信息。
        # 配置增强不能关闭既有字面保护。
        self.validate_hybrid_terms = validate_hybrid_terms or enable_d7f or enable_d7g or enable_integrated_quality or enable_professional_quality
        self.generalization_knowledge = generalization_knowledge
        self.enforce_extended_semantics = enforce_extended_semantics or enable_integrated_quality or enable_professional_quality
        self.professional_knowledge = professional_knowledge
        self.enable_marked_rules = enable_marked_rules
        self.enable_d7f = enable_d7f
        self.enable_d7g = enable_d7g
        self.enable_d8a = enable_d8a
        self.enable_d8b = enable_d8b
        self.d8b_knowledge = d8b_knowledge
        self.enable_engineering_symbol_risk = enable_engineering_symbol_risk
        self.enable_d8c = enable_d8c
        self.enable_d8c_ocr_verification = enable_d8c_ocr_verification
        self.review_overlay = review_overlay
        # 非off模式必须有完整可信runtime，
        # 不能只凭模式名构造空知识服务。
        # quality仅允许在candidate下执行，
        # shadow只观察候选而不切换译文。
        # 不合法配置在推理前尽早报错。
        if knowledge_mode not in {'off', 'shadow', 'candidate'} or (knowledge_mode != 'off' and knowledge_runtime is None):
            raise ValueError('V2 mode requires a complete validated runtime package')
        self.knowledge_mode = knowledge_mode
        self.knowledge_runtime = knowledge_runtime
        if enable_v2_quality_optimization and knowledge_mode != 'candidate':
            raise ValueError('quality optimization requires candidate mode')
        self.enable_v2_quality_optimization = enable_v2_quality_optimization
        self.reviewed94 = reviewed94
        self.enable_integrated_quality=enable_integrated_quality
        if enable_professional_quality and knowledge_mode != 'candidate':
            raise ValueError('professional quality requires candidate mode')
        self.enable_professional_quality=enable_professional_quality
        # Reference依赖Professional候选，
        # Convergence再依赖Reference。
        # ArgosCore与Final依次要求父能力，
        # 保证候选运行身份能够解释真实链路。
        # ProductCore也要求ArgosCore，
        # SemanticEdit只在Final父配置运行。
        # 这些约束不是默认启用承诺。
        if enable_reference_quality and (knowledge_mode!='candidate' or not enable_professional_quality):
            raise ValueError('reference quality requires professional candidate mode')
        self.enable_reference_quality=enable_reference_quality
        if enable_convergence_quality and not enable_reference_quality:
            raise ValueError("convergence quality requires reference candidate parent")
        self.enable_convergence_quality=enable_convergence_quality
        if enable_argos_core and not enable_convergence_quality:
            raise ValueError('Argos core requires convergence candidate parent')
        self.enable_argos_core=enable_argos_core
        self.argos_core=None
        if enable_argos_core:
            from .translation_argos_core import ArgosCoreComposer
            self.argos_core=ArgosCoreComposer(glossary,phrase_memory)
        if enable_final_optimization and not enable_argos_core:
            raise ValueError('final optimization requires Argos core candidate parent')
        self.enable_product_core=enable_product_core
        if enable_product_core and not enable_argos_core:
            raise ValueError("product core requires Argos core")
        self.enable_final_optimization=enable_final_optimization
        self.qualified_proposals=None
        if enable_final_optimization or enable_product_core:
            from .translation_qualified_proposals import QualifiedProposalEngine
            self.qualified_proposals=QualifiedProposalEngine(glossary,phrase_memory)
        if enable_semantic_edit and not enable_final_optimization:
            raise ValueError('semantic edit requires Final Optimization parent')
        self.enable_semantic_edit = enable_semantic_edit
        self.semantic_editor = None
        if enable_semantic_edit:
            from .translation_aligned_edit import AlignedTermEditor
            self.semantic_editor = AlignedTermEditor(glossary)
        # 每个服务拥有独立校验上下文，
        # runtime_identity包含知识版本与pin。
        # 各阶段版本标记继续追加身份，
        # 缓存不得跨不同候选配置共用。
        # 审核增量还加入自身fingerprint，
        # 避免同名候选文件变动后沿用旧结果。
        self._validation_cache = ContextVar('translation_validation_cache', default=None)
        self.runtime_identity = ('legacy' if knowledge_mode == 'off' else
            f'{knowledge_runtime.version}:{knowledge_mode}:{knowledge_runtime.fingerprint}')
        if enable_v2_quality_optimization:
            self.runtime_identity += ':quality.1'
        if reviewed94 is not None:
            self.runtime_identity += ':reviewed94.1:' + reviewed94.fingerprint
        if enable_integrated_quality:
            self.runtime_identity += ':integrated.1'
        if enable_professional_quality:
            self.runtime_identity += ':professional.1'
        if enable_reference_quality:
            self.runtime_identity += ':reference.1'
        if enable_convergence_quality:
            self.runtime_identity += ':convergence.2'
        if enable_argos_core:
            self.runtime_identity += ':argoscore.1'
        if enable_final_optimization:
            self.runtime_identity += ':final.1'
        if enable_product_core:
            self.runtime_identity += ':product.1'
        if enable_semantic_edit:
            self.runtime_identity += ':semantic-edit.1'
        # D7D别名只扩大已有术语可接受译法，
        # 不在此阶段新增整句人工授权。
        # Professional粗糙度上限依赖quality开关，
        # 不能因加载知识对象便隐式生效。
        # 模型加载另受互斥锁保护，
        # 构造服务本身不支付CT2推理成本。
        self._extra_term_targets = ({
            phrase.entry_id: generalization_knowledge.term_targets(phrase.entry_id)
            for phrase in glossary.reviewed_phrases() if phrase.kind == 'term'
        } if generalization_knowledge is not None else None)
        if professional_knowledge is not None:
            professional_knowledge.enable_roughness_limits = enable_v2_quality_optimization
        self.knowledge_audit_seconds = 0.0
        self._loaded = False
        self._load_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: TranslationConfig) -> "TranslationService":
        if config.knowledge_mode not in {'off', 'shadow', 'candidate'}:
            raise ValueError('invalid knowledge mode')
        if config.enable_reviewed94 and config.knowledge_mode != 'candidate':
            raise ValueError('reviewed94 requires V2 candidate mode')
        if config.enable_integrated_quality and config.knowledge_mode != 'candidate':
            raise ValueError('integrated quality requires V2 candidate mode')
        # 工厂把配置落实到实际服务对象，
        # 先验证模式与增量候选前置条件。
        # 自定义V2目录必须显式提供可信pin，
        # 不能自动信任目录里的自报摘要。
        # off不加载runtime，shadow也不换词表。
        runtime = None
        if config.knowledge_mode != 'off':
            from .knowledge_v2_runtime import RuntimeKnowledge, default_runtime_root, MANIFEST_SHA256
            if config.knowledge_root is not None and not config.knowledge_manifest_sha256:
                raise ValueError('custom V2 package requires an explicit trusted manifest hash')
            runtime = RuntimeKnowledge.load(config.knowledge_root or default_runtime_root(),
                config.knowledge_manifest_sha256 or MANIFEST_SHA256)
        # 基础D6先加载并保持其优先级，
        # candidate使用RuntimeGlossary包装。
        # 旧模式仍按补充开关追加safe V1，
        # 避免开启shadow时改变旧候选集合。
        # 资源组合冲突由词表加载器拒绝。
        glossary = Glossary.load(config.glossary_path)
        if config.knowledge_mode == 'candidate':
            glossary = runtime.glossary(glossary)
        elif config.enable_supplemental_glossaries and config.supplemental_glossary_paths:
            glossary = Glossary.load_many((
                config.glossary_path, *config.supplemental_glossary_paths,
            ))
        # D7C加载时明确使用安全验证视图，
        # candidate仍保留原V1证据ID。
        # 执行视图中的挂起表达不因此重新放行，
        # 隔离清单也必须同时通过验证。
        # D7D/D7E只有对应开关才加载。
        domain_knowledge = None
        if config.enable_d7c_enhancement:
            domain_knowledge = DomainKnowledge.load(
                default_d7c_knowledge_path(), safe_glossary=(
                    runtime.validation_glossary if config.knowledge_mode == 'candidate' else glossary),
                isolation_path=default_v1_isolation_path(),
            )
        generalization_knowledge = None
        if config.enable_d7d_enhancement:
            generalization_knowledge = GeneralizationKnowledge.load(
                default_d7d_knowledge_path(), glossary=glossary,
            )
        professional_knowledge = None
        if config.enable_d7e_enhancement:
            professional_knowledge = ProfessionalKnowledge.load(
                default_d7e_knowledge_path(), glossary=glossary,
            )
        # 短语记忆按实际增强开关构建，
        # 仅声明高级开关并不会凭空产生规则。
        # 同一实例聚合D7C/D7D/D7E能力，
        # 未识别片段才通过受控回调送模型。
        # 传入服务的是已验证对象而非答案路径。
        phrase_memory = (
            PhraseMemory.from_glossary(
                glossary, domain_knowledge=domain_knowledge,
                generalization_knowledge=generalization_knowledge,
                professional_knowledge=professional_knowledge,
            )
            if (config.enable_phrase_enhancement or config.enable_d7c_enhancement
                or config.enable_d7d_enhancement or config.enable_d7e_enhancement) else None
        )
        return cls(
            CTranslate2Backend(config.model_root),
            glossary,
            max_input_tokens=config.max_input_tokens,
            phrase_memory=phrase_memory,
            validate_hybrid_terms=(
                config.enable_d7c_enhancement or config.enable_d7d_enhancement
                or config.enable_d7e_enhancement
            ),
            generalization_knowledge=generalization_knowledge,
            enforce_extended_semantics=(
                config.enable_d7d_enhancement or config.enable_d7e_enhancement
            ),
            professional_knowledge=professional_knowledge,
            enable_marked_rules=config.enable_marked_rules,
            enable_d7f=config.enable_d7f,
            enable_d7g=config.enable_d7g,
            enable_d8a=config.enable_d8a,
            enable_d8b=config.enable_d8b,
            d8b_knowledge=(
                D8BKnowledge.load(default_d8b_knowledge_path())
                if config.enable_d8b else None
            ),
            enable_engineering_symbol_risk=config.enable_engineering_symbol_risk,
            enable_d8c=config.enable_d8c,
            enable_d8c_ocr_verification=config.enable_d8c_ocr_verification,
            review_overlay=(runtime.review_overlay if config.knowledge_mode == 'candidate' else
                            ReviewOverlay.load(config.review_overlay_path) if config.review_overlay_path is not None else None),
            knowledge_mode=config.knowledge_mode, knowledge_runtime=runtime,
            enable_v2_quality_optimization=config.enable_v2_quality_optimization,
            reviewed94=load_reviewed94() if config.enable_reviewed94 else None,
            enable_integrated_quality=config.enable_integrated_quality,
            enable_professional_quality=config.enable_professional_quality,
            enable_reference_quality=config.enable_reference_quality,
            enable_convergence_quality=config.enable_convergence_quality,
            enable_argos_core=config.enable_argos_core,
            enable_final_optimization=config.enable_final_optimization,
            enable_product_core=config.enable_product_core,
            enable_semantic_edit=config.enable_semantic_edit,
        )

    # 双重检查配合互斥锁避免重复加载，
    # 耗时模型初始化由调用线程承担。
    # backend.load成功后才设置_loaded，
    # 失败可保留明确异常而非假装就绪。
    # 本层没有联网重试或自动换模型。
    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self.backend.load()
            self._loaded = True

    @_validation_request
    # 单串入口先执行Product源文政策，
    # 不合资格时明确source_preserved。
    # Reviewed94完整精确授权优先选择，
    # 其它输入进入同一批处理核心。
    # D8B、源风险和V2审计在核心后执行，
    # 防止后置候选绕过工程边界。
    def translate_text(self, source_text: str) -> TextTranslation:
        if self.enable_product_core:
            from .source_text_policy import english_translation_eligibility
            policy=english_translation_eligibility(source_text)
            if not policy.eligible:
                return TextTranslation(source_text,('source_text_preserved','source_policy:'+policy.reason),'source_preserved')
        if self.reviewed94 is not None:
            candidate=self.reviewed94.exact(source_text)
            if candidate is not None:return self._reviewed_translation(candidate,source_text)
        baseline = self._translate_texts((source_text,))[0]
        translated = self._resolve_d8b(source_text, baseline) if self.enable_d8b else baseline
        return self._audit_knowledge(source_text, self._apply_source_risk(source_text, translated))

    def translate_entries(self, entries: Sequence[TargetEntry]) -> tuple[TranslationResult, ...]:
        return tuple(item.result for item in self.translate_entries_with_details(entries))

    @_validation_request
    def translate_entries_with_details(
        self, entries: Sequence[TargetEntry], *,
        review_contexts: Sequence[ReviewContext | None] | None = None,
    ) -> tuple[EntryTranslation, ...]:
        # 条目与审核上下文必须一一对应，
        # 数量不一致立即拒绝而非错配。
        # Product先把应保留文本分离，
        # 仅合资格子集继续真实送译。
        # 回填按原索引恢复输入顺序，
        # 不会因过滤改变group与region关联。
        if review_contexts is not None and len(review_contexts) != len(entries):
            raise ValueError('review context count must match entries')
        if self.enable_product_core:
            from .source_text_policy import english_translation_eligibility
            policies=tuple(english_translation_eligibility(e.source_text) for e in entries)
            if any(not p.eligible for p in policies):
                indices=[i for i,p in enumerate(policies) if p.eligible]
                routed=self.translate_entries_with_details(tuple(entries[i] for i in indices),
                    review_contexts=tuple(review_contexts[i] for i in indices) if review_contexts is not None else None) if indices else ()
                outputs=dict(zip(indices,routed))
                for i,(entry,policy) in enumerate(zip(entries,policies)):
                    if policy.eligible:continue
                    warnings=('source_text_preserved','source_policy:'+policy.reason)
                    outputs[i]=EntryTranslation(TranslationResult(1,entry.document_id,entry.page_index,
                        entry.group_id,entry.region_ids,entry.source_text,entry.source_text,'machine',warnings),
                        TextTranslation(entry.source_text,warnings,'source_preserved'))
                return tuple(outputs[i] for i in range(len(entries)))
        # 原始source_text始终用于结果契约，
        # translation_source只决定实际送译。
        # D8C核验开关关闭时二者相同，
        # 开关开启也不能覆盖原OCR字段。
        # 后续详情需要同时解释两份字面。
        source_texts = tuple(entry.source_text for entry in entries)
        translation_sources = tuple(
            entry.translation_source_text if self.enable_d8c_ocr_verification
            else entry.source_text
            for entry in entries
        )
        # D8A在同页条目中寻找局部语境，
        # Reviewed94启用时采用请求内预计算。
        # 其它情况保持原逐项解析路径，
        # 不能将不同页面的上下文混用。
        # 使用核验字面同样受D8C开关限制。
        contextual = (resolve_page_context(entries,use_verified_source=self.enable_d8c_ocr_verification)
            if self.enable_d8a and self.reviewed94 is not None else tuple(
            resolve_contextual_translation(
                entry, entries, use_verified_source=self.enable_d8c_ocr_verification,
            ) if self.enable_d8a else None
            for entry in entries
        ))
        # With the addendum disabled, preserve the legacy candidate set and
        # ordering. Enabled corrections are authorized before model inference.
        # 完整增量授权在模型之前选择，
        # 已安全授权的条目不先推理再丢弃。
        # 未授权条目的索引单独送旧核心，
        # 避免开启增量改变剩余输入顺序。
        # 批准只限资源声明的精确或实例范围。
        reviewed = (self.reviewed94.resolve_entries(entries, review_contexts or (None,)*len(entries),
                    use_verified_source=self.enable_d8c_ocr_verification)
                    if self.reviewed94 is not None else (None,)*len(entries))
        # Validate known reviewed candidates first; the unchanged legacy route is
        # used for every other entry. No model output is generated then discarded
        # for a complete, safely authorized correction.
        remaining=[i for i,c in enumerate(reviewed) if c is None]
        stable=dict(zip(remaining,self._translate_texts(tuple(translation_sources[i] for i in remaining)))) if remaining else {}
        stable_translated=tuple(self._reviewed_translation(c,entry.source_text) if c is not None else stable[i]
                                for i,(entry,c) in enumerate(zip(entries,reviewed)))
        # 各后处理阶段按固定优先级串联，
        # 增量已选择结果不再被D8B替换。
        # D8A候选需完整守卫才替代基线，
        # 再附原条目的核验证据。
        # 实例overlay最后仍受原框原文约束，
        # V2审计记录最终采用或拒绝。
        baseline_translated = tuple(
            self._resolve_d8b(source, stable) if reviewed[i] is None else stable
            for i,(source, stable) in enumerate(zip(translation_sources, stable_translated))
        ) if self.enable_d8b else stable_translated
        translated = tuple(
            self._select_contextual_translation(
                translation_sources[index], item, baseline_translated[index],
                source_texts[index],
            ) if reviewed[index] is None else baseline_translated[index]
            for index, item in enumerate(contextual)
        )
        translated = tuple(
            self._apply_entry_evidence(entry, translation_source, result) if reviewed[i] is None else result
            for i,(entry, translation_source, result) in enumerate(zip(entries, translation_sources, translated))
        )
        translated = tuple(
            self._apply_review_overlay(entry, source, result, context) if reviewed[i] is None else result
            for i,(entry, source, result, context) in enumerate(zip(
                entries, translation_sources, translated,
                review_contexts if review_contexts is not None else (None,) * len(entries),
            ))
        )
        translated = tuple(self._audit_knowledge(source, result, entry=entry, context=context) if reviewed[i] is None else result
            for i,(entry, source, result, context) in enumerate(zip(entries, translation_sources, translated,
                review_contexts if review_contexts is not None else (None,) * len(entries))))
        # 结果构造复制原文档、页码和group，
        # region_ids与原始英文保持可追溯。
        # 所有自动路线统一machine来源，
        # 人工知识命中也不会改成human_reviewed。
        # 最终契约验证数量、身份与来源，
        # 后台与UI据此绑定正确条目。
        results = tuple(
            TranslationResult(
                schema_version=1,
                document_id=entry.document_id,
                page_index=entry.page_index,
                group_id=entry.group_id,
                region_ids=entry.region_ids,
                source_text=entry.source_text,
                translated_text=text_result.translated_text,
                provenance="machine",
                warnings=text_result.warnings,
            )
            for entry, text_result in zip(entries, translated)
        )
        validate_translation_results(
            entries, results, allowed_provenance=frozenset({"machine"}),
        )
        return tuple(
            EntryTranslation(result, detail)
            for result, detail in zip(results, translated)
        )

    def _reviewed_translation(self,candidate,original_source):
        """Expert lexical authority is exact; engineering gates remain independent.

        The addendum may supersede older lexical wording only for this complete
        accepted expression. It does not disable numerical/semantic/professional
        limit checks, confer human provenance, or grant PDF export eligibility.
        """
        from .knowledge_v2_runtime import KnowledgeUsage
        source,target=candidate.source,candidate.target
        # 用户明确无需翻译时保留原始英文，
        # 附单独preserved决策与审核行号。
        # 该结果不是模型改写成功，
        # 不会删除原文中的工程标记。
        # 增量来源ID保持独立MDTK2-R94命名。
        if candidate.action=='preserve':
            evidence=PhraseEvidence(original_source,original_source,candidate.evidence_id,'reviewed94',candidate.scope,
                'MDTK2-R94-'+candidate.evidence_id)
            trace=KnowledgeUsage('candidate',self.reviewed94.fingerprint,evidence.knowledge_id,
                tuple('reviewed94-row:'+str(r) for r in candidate.rows),candidate.scope,'preserved',original_source,'user_no_translation_required')
            return TextTranslation(original_source,('reviewed94:no_translation_required',
                'reviewed94:preserved:'+candidate.evidence_id),'exact_rule',(evidence,),(trace,))
        # 专家完整表达可替代旧词汇偏好，
        # 仅过滤已列明的词汇缺失类警告。
        # 数值、方向、条件及其它专业约束，
        # 一律继续阻止危险译文。
        # 源符号高风险也不能借审核跳过，
        # 拒绝时回退原始OCR而非修正候选。
        warnings=self._final_validation_warnings(source,target,include_terms=True)
        lexical=('required_term_missing:','professional_term_missing:','d7f_term_missing:')
        rejected=tuple(w for w in warnings if not w.startswith(lexical))
        if self.enable_engineering_symbol_risk:
            rejected=(*rejected,*assess_source_risk(source).warning_codes)
        if rejected:return _fallback(original_source,*rejected,'reviewed94:rejected:'+candidate.evidence_id)
        evidence=PhraseEvidence(original_source,target,candidate.evidence_id,'reviewed94',candidate.scope,
            'MDTK2-R94-'+candidate.evidence_id)
        # 采用增量仍标机器待审核，
        # rows记录真实授权来自哪些表格行。
        # 若送译英文与原OCR不同，
        # 附accepted_source说明实际采用字面。
        # 这不是修改TextRegion原文或跨度，
        # 更不授予正式PDF导出资格。
        notes=('machine_translation_unreviewed:机器译文未经人工审核',
               'reviewed94:used:'+candidate.evidence_id,
               'reviewed94:rows:'+','.join(map(str,candidate.rows)))
        if source!=original_source:notes+=('reviewed94:accepted_source:'+source,)
        trace=KnowledgeUsage('candidate',self.reviewed94.fingerprint,evidence.knowledge_id,
            tuple('reviewed94-row:'+str(r) for r in candidate.rows),candidate.scope,'adopted',target,'expert_expression_and_engineering_gates')
        return TextTranslation(target,notes,'exact_rule',(evidence,),(trace,))

    def _apply_review_overlay(self, entry, source, baseline, context):
        if self.review_overlay is None or context is None:
            return baseline
        if self.knowledge_mode == 'candidate':
            from .knowledge_v2 import validate_context
            validate_context(context)
        # Bind caller identity to the actual entry; supplied page/bounds cannot
        # substitute for the entry's original coordinates or OCR string.
        # 审核上下文必须对应当前真实条目，
        # 外部传入页码与框不能替代原坐标。
        # 一基页码与内部零基页索引显式换算，
        # 上下文框要求完全一致。
        # 随后仍以entry.source_text匹配原审核。
        bounds = {k: getattr(entry.bounds, k) for k in ('min_x', 'min_y', 'max_x', 'max_y')}
        if context.page != entry.page_index + 1 or context.bounds != bounds:
            return baseline
        records = self.review_overlay.match(context, entry.source_text)
        # 历史续行合并只可借原区域展示证据，
        # 不能逐碎片重复使用整组中文。
        # 无显式实例且无spans时才尝试兼容，
        # 即使命中也标merged_evidence_only。
        # 原实例批准范围不会被合并扩张。
        merged_evidence_only = False
        if (not records and not context.instance_id and not entry.spans
                and 'd8c_reviewed_continuation' in entry.boundary_notes):
            records = tuple(r for region in entry.regions for r in self.review_overlay.match(
                ReviewContext(context.pdf, context.page, {k: getattr(region.bounds, k) for k in bounds}),
                region.source_text,
            ))
            merged_evidence_only = bool(records)
        if not records:
            return TextTranslation(
                baseline.translated_text,
                (*baseline.warnings, 'user_review_overlay:no_instance_match'),
                baseline.route, baseline.evidence,
            )
        # 核验英文不同于原批准英文时拒绝借用，
        # candidate还须runtime授予历史实例资格。
        # 合并成分始终只作context evidence，
        # 不自动将多个批准拼成新批准句。
        # 理由保留在基线warnings便于复核。
        target, reason = self.review_overlay.translation_candidate(records)
        if target is not None and self.knowledge_mode == 'candidate' and source != entry.source_text:
            target, reason = None, 'verified_source_differs_from_original_approval'
        if target is not None and self.knowledge_mode == 'candidate' and not self.knowledge_runtime.review_eligible(records):
            target, reason = None, 'v2_scope_not_eligible'
        if merged_evidence_only:
            target, reason = None, 'merged_constituent_context_only'
        evidence = tuple(PhraseEvidence(
            r['source_text'], r['accepted_targets_raw'], r['instance_id'],
            'user_approved_review_overlay.v2', 'instance_context_evidence',
            (self.knowledge_runtime.knowledge_for_ref('review:' + r['instance_id'])['id']
             if self.knowledge_mode == 'candidate' else None),
        ) for r in records)
        # Existing reviewed glossary remains authoritative, including conflicts.
        # 既有词表精确结果优先于实例覆盖，
        # 冲突不会凭后加载overlay静默改判。
        # 源风险与完整最终守卫继续独立检查，
        # 任一失败只留下证据与拒绝原因。
        # 合格候选采用hybrid_memory路线，
        # 仍需machine_translation_unreviewed。
        if target is not None and self.glossary.resolve(source) is not None:
            reason = 'existing_glossary_priority'
            target = None
        if target is not None and self.enable_engineering_symbol_risk and assess_source_risk(source).is_high_risk:
            reason = 'source_risk_evidence_only'
            target = None
        if target is not None:
            rejected = self._final_validation_warnings(source, target, include_terms=True)
            if rejected:
                reason = 'candidate_rejected:' + ','.join(rejected)
                target = None
        identifiers = ','.join(r['instance_id'] for r in records)
        if target is None:
            return TextTranslation(baseline.translated_text,
                (*baseline.warnings, f'user_review_overlay:{identifiers}:{reason}'),
                baseline.route, (*baseline.evidence, *evidence))
        return TextTranslation(target,
            ('machine_translation_unreviewed:机器译文未经人工审核',
             f'user_review_overlay:{identifiers}:used'),
            'hybrid_memory', evidence)

    # 语境选择只能提出替换候选，
    # 不能绕过完整源目标最终守卫。
    # 拒绝时保留原基线正文与路线，
    # 追加d8a_candidate_rejected原因。
    # 采用时明确context机制与专业待审，
    # 不把页内关键词视作专家整句批准。
    def _select_contextual_translation(self, source, item, baseline, original_source):
        """Apply existing final gates to a replacement and retain rejection reasons."""
        if item is None:
            return baseline
        rejected = self._final_validation_warnings(source, item.translated_text, include_terms=True)
        if rejected:
            return TextTranslation(
                baseline.translated_text,
                (*baseline.warnings, *(f"d8a_candidate_rejected:{code}" for code in rejected)),
                baseline.route, baseline.evidence,
            )
        return TextTranslation(
                item.translated_text,
                ("machine_translation_unreviewed:机器译文未经人工审核",
                 "d8a_professional_review_pending",
                 f"d8a_context:{item.mechanism}"),
                "hybrid_memory",
                (PhraseEvidence(
                    original_source, item.translated_text, "d8a-context-v1", "d8a",
                    item.mechanism,
                ),),
        )

    def _apply_entry_evidence(
        self, entry: TargetEntry, translation_source: str, translated: TextTranslation,
    ) -> TextTranslation:
        # 先以实际送译英文检查源风险，
        # 再把句界恢复与字符核验写入详情。
        # verified_source_text不是新OCR真值，
        # 源区域、坐标与原字面仍由entry保存。
        # 证据说明从原字面到送译字面的变化。
        resolved = self._apply_source_risk(translation_source, translated)
        warnings = list(resolved.warnings)
        evidence = list(resolved.evidence)
        if self.enable_d8c and "d8c_reviewed_continuation" in entry.boundary_notes:
            warnings.append("d8c_semantic_boundary:reviewed_phrase_recovery")
        if self.enable_d8c_ocr_verification and translation_source != entry.source_text:
            warnings.extend((
                "d8c_ocr_character_auto_verified",
                f"d8c_verified_translation_source:{translation_source}",
            ))
            evidence.append(PhraseEvidence(
                entry.source_text, translation_source, "d8c-local-image-glyph",
                "d8c-image-verification", "verified_translation_input",
            ))
        return TextTranslation(
            resolved.translated_text, tuple(dict.fromkeys(warnings)),
            resolved.route, tuple(evidence),
        )

    # D8B有限表达仅生成候选译文，
    # 完整守卫失败则保持原基线。
    # 多个候选ID可共同解释拼接来源，
    # 但输出仍属于实验机器译文。
    # 无命中不会强行使用近似表达。
    def _resolve_d8b(self, source: str, baseline: TextTranslation) -> TextTranslation:
        resolution = resolve_expression(source, self.d8b_knowledge)
        if resolution is None:
            return baseline
        warnings = self._final_validation_warnings(
            source, resolution.translated_text, include_terms=True,
        )
        if warnings:
            return baseline
        evidence_id = "+".join(resolution.candidate_ids)
        return TextTranslation(
            resolution.translated_text,
            ("machine_translation_unreviewed:机器译文未经人工审核",
             "d8b_professional_review_pending",
             f"d8b_mechanism:{resolution.mechanism}"),
            "hybrid_memory",
            (PhraseEvidence(
                source, resolution.translated_text, evidence_id,
                "d8b-experimental-v1", resolution.mechanism,
            ),),
        )

    # Integrated混合东亚文字按原样回退，
    # 英文模型不能猜哪段已是翻译。
    # 工程符号风险开关独立控制，
    # 未解决角公差等疑点统一安全回退。
    # 来源风险与模型译文好坏分开归因。
    def _apply_source_risk(
        self, source: str, translated: TextTranslation,
    ) -> TextTranslation:
        if self.enable_integrated_quality and re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]',source):
            return _fallback(source,'mixed_language_constraints_preserved')
        if not self.enable_engineering_symbol_risk:
            return translated
        risk = assess_source_risk(source)
        if not risk.is_high_risk:
            return translated
        return _fallback(
            source, *risk.warning_codes, "engineering_source_review_required",
        )

    def translate_selection(self, selection: SelectionResult) -> tuple[TranslationResult, ...]:
        return self.translate_entries(selection.candidate_entries)

    def _translate_texts(self, source_texts: Sequence[str]) -> tuple[TextTranslation, ...]:
        # 核心先分配与输入等长的结果槽，
        # 每种规则都写回自己的原始索引。
        # 只把未解析输入收入model_inputs，
        # model_indices保证批次回填顺序。
        # 末尾不得保留None，
        # 缺项视为推理契约错误而非静默丢句。
        resolved: list[TextTranslation | None] = [None] * len(source_texts)
        model_indices: list[int] = []
        model_inputs: list[str] = []
        for index, source in enumerate(source_texts):
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source_text must be a non-empty string")
            if self.enable_integrated_quality and re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]',source):
                # Keep the detected mixed-language evidence, including Chinese
                # restrictions, intact. An English-only model cannot safely
                # infer which language fragment is an existing translation.
                resolved[index]=_fallback(source,'mixed_language_constraints_preserved')
                continue
            # 执行前先查V2策略挂起，
            # 拒绝范围优先于词表与模型路线。
            # 不能因相同句子能被后续组合，
            # 就绕开已知危险字面限制。
            # 该分支的回退原因带runtime_hold。
            # 后置D8或语境候选选择结束后，
            # 再次检查runtime策略挂起。
            # 危险字面不能从兼容算法重新进入，
            # 即使它曾通过某条局部检查。
            # 运行身份与实际使用知识ID附详情，
            # trace保持候选和最终决策的区别。
            if self.knowledge_mode == 'candidate':
                hold = self.knowledge_runtime.policy_hold(source)
                if hold:
                    resolved[index] = _fallback(source, 'v2_runtime_hold:' + hold)
                    continue
            # 完整精确或参数规则优先于推理，
            # candidate使用完整最终守卫。
            # 旧模式保留原先字面与基本语义检查，
            # 避免本轮候选改变历史诊断口径。
            # 规则命中仍可能因工程信息不符回退。
            rule = self.glossary.resolve(source)
            if rule is not None:
                if self.knowledge_mode == 'candidate':
                    rejected = self._final_validation_warnings(source, rule.translated_text, include_terms=True)
                    evidence = self._v2_rule_evidence(source, rule)
                    resolved[index] = (_fallback(source, *rejected) if rejected else
                        TextTranslation(rule.translated_text, (), rule.route, evidence))
                    # quality只在被拒精确规则后尝试组合，
                    # 并且必须存在已构建phrase_memory。
                    # 新组合重新接受完整守卫，
                    # 不能用旧规则的批准状态遮盖错误。
                    # 成功后继续保留机器与组合待审警告。
                    if rejected and self.enable_v2_quality_optimization and self.phrase_memory:
                        enhancement = self.phrase_memory.enhance(source, self._translate_phrase_segment)
                        if enhancement is not None:
                            retry = self._validate_phrase_enhancement(source, enhancement)
                            if translation_is_available(retry):
                                resolved[index] = replace(retry, warnings=(*retry.warnings,
                                    'machine_translation_unreviewed:机器译文未经人工审核',
                                    'v2_quality_composition_review_pending'))
                    continue
                literal_check = validate_immutable_literals(source, rule.translated_text)
                semantic_check = validate_engineering_semantics(
                    source, rule.translated_text, extended=False,
                )
                if literal_check.is_valid and semantic_check.is_valid:
                    resolved[index] = TextTranslation(
                        rule.translated_text, (), rule.route,  # type: ignore[arg-type]
                    )
                else:
                    resolved[index] = _fallback(
                        source, *literal_check.warning_codes, *semantic_check.warning_codes,
                    )
                continue
            # 合资格全覆盖组合先检查源恢复禁区，
            # 歧义或未知语言不因词叶命中被放宽。
            # 提案只能使用当前已知源片段，
            # 成功还需完整字面、关系和术语守卫。
            # 该路线不调用模型补齐未覆盖内容。
            if self.qualified_proposals is not None and not self._final_source_recovery_blocked(source):
                enhancement=self.qualified_proposals.propose(source)
                if enhancement is not None:
                    candidate=self._validate_phrase_enhancement(source,enhancement)
                    if translation_is_available(candidate):
                        resolved[index]=replace(candidate,warnings=tuple(dict.fromkeys((*candidate.warnings,
                            'machine_translation_unreviewed:机器译文未经人工审核',
                            'qualified_full_coverage_review_pending'))))
                        continue
            # ArgosCore名称不表示在此跑Argos，
            # 它只尝试有限全覆盖复用语法。
            # 找到组合后直接验证并确定该项结果，
            # 失败仍以安全回退保存原因。
            # 不继续用另一条路径掩盖已发现冲突。
            if self.argos_core is not None:
                enhancement=self.argos_core.resolve(source)
                if enhancement is not None:
                    candidate=self._validate_phrase_enhancement(source,enhancement)
                    resolved[index]=replace(candidate,warnings=tuple(dict.fromkeys((*candidate.warnings,
                        'machine_translation_unreviewed:机器译文未经人工审核',
                        'argos_core_composition_review_pending'))))
                    continue
            # Convergence索引按服务惰性构建一次，
            # 只恢复唯一完整已知表达的空白形态。
            # 维度图例与明确材料定义保持窄范围，
            # 不读取开发评测答案。
            # 特定非英语或不透明工程字段，
            # 在模型之前明确保留原文。
            if getattr(self,'enable_convergence_quality',False):
                from .translation_convergence import ConvergenceReuse,unsupported_input_warning
                reuse=getattr(self,'_convergence_reuse',None)
                if reuse is None:reuse=self._convergence_reuse=ConvergenceReuse(self.glossary)
                enhancement=reuse.resolve(source)
                if enhancement is not None:
                    candidate=self._validate_phrase_enhancement(source,enhancement)
                    resolved[index]=replace(candidate,warnings=tuple(dict.fromkeys((*candidate.warnings,
                        'machine_translation_unreviewed:机器译文未经人工审核','convergence_structure_review_pending'))))
                    continue
                warning=unsupported_input_warning(source)
                if warning:
                    resolved[index]=_fallback(source,warning)
                    continue
            # 前导星号仅作为独立引用标记处理，
            # 保留星号数量并翻译已知完整主体。
            # 普通乘法、尾星号或未知主体，
            # 继续保留原路径而不去掉标记。
            # 专业歧义在任何自由模型回退前拒绝。
            marked = self._resolve_marked_rule(source) if self.enable_marked_rules else None
            if marked is not None:
                resolved[index] = marked
                continue
            if self.enable_professional_quality:
                from .mechanical_instructions import ambiguity_warning
                ambiguous = ambiguity_warning(source, self.glossary)
                if ambiguous:
                    resolved[index] = _fallback(source, *ambiguous)
                    continue
            # Integrated先拒绝已知范围歧义，
            # 再用有限条件与加工子句组合。
            # 生成中文仍需完整最终校验，
            # 通过后标integrated待专业复核。
            # 这里不是完整自然语言语义解析器。
            if self.enable_integrated_quality:
                if instruction_scope_ambiguous(source,self.glossary):
                    resolved[index]=_fallback(source,'engineering_instruction_scope_ambiguous')
                    continue
                composition=compose_instructions(source,self.glossary,self.professional_knowledge)
                if composition is not None:
                    candidate=self._validate_phrase_enhancement(source,composition)
                    resolved[index]=(replace(candidate,warnings=(*candidate.warnings,
                        'machine_translation_unreviewed:机器译文未经人工审核',
                        'integrated_composition_review_pending')) if translation_is_available(candidate) else candidate)
                    continue
            if self.enable_professional_quality:
                # Professional使用有角色槽的有限AST，
                # 加工动作、对象、否定与义务成组保存。
                # 解析不了的原句不截去未知部分，
                # 后续其它受控路径仍可尝试。
                # 可组合不等于专家认可整个新句。
                from .mechanical_instructions import compose_instruction
                instruction = compose_instruction(source, self.glossary)
                if instruction is not None:
                    candidate = self._validate_phrase_enhancement(source, instruction)
                    resolved[index] = (replace(candidate, warnings=tuple(dict.fromkeys((
                        *candidate.warnings, 'machine_translation_unreviewed:机器译文未经人工审核',
                        'professional_semantic_composition_review_pending'))))
                        if translation_is_available(candidate) else candidate)
                    continue
            # Reference复用显式定义或结构关系，
            # 运行只读取现有合资格词表与记忆。
            # 索引按当前服务缓存，
            # 不从评测参考文件导入答案。
            # 合成输出始终附reference待审标记。
            if getattr(self,'enable_reference_quality',False) and self.phrase_memory is not None:
                from .translation_reference_reuse import ReferencePhraseReuse
                reuse=getattr(self,'_reference_reuse',None)
                if reuse is None:
                    reuse=self._reference_reuse=ReferencePhraseReuse(self.glossary,self.phrase_memory)
                enhancement=reuse.resolve(source)
                if enhancement is not None:
                    candidate=self._validate_phrase_enhancement(source,enhancement)
                    resolved[index]=(replace(candidate,warnings=tuple(dict.fromkeys((
                        *candidate.warnings,'machine_translation_unreviewed:机器译文未经人工审核',
                        'reference_structure_composition_review_pending'))))
                        if translation_is_available(candidate) else candidate)
                    continue
            # D7F只处理完整已登记图纸字段，
            # 缩写无法确定意义时先保留原文。
            # 数字加已知字段仍要字面量校验，
            # 不把词汇表扩展到任意句中替换。
            # 新增中文字段保持lexical_review_pending。
            if self.enable_d7f:
                field = resolve_field(source)
                if field is not None:
                    translated, kind = field
                    warnings = self._final_validation_warnings(source, translated, include_terms=True)
                    resolved[index] = (_fallback(source, *warnings) if warnings else TextTranslation(
                        translated, ('machine_translation_unreviewed:机器译文未经人工审核',
                                     'd7f_lexical_review_pending'), 'hybrid_memory',
                        (PhraseEvidence(source, translated, 'd7f-drawing-fields', 'd7f', kind),),
                    ))
                    continue
                if source.strip().upper() in UNRESOLVED_ABBREVIATIONS:
                    resolved[index] = _fallback(source, 'unresolved_drawing_abbreviation')
                    continue
            # D7G顺序组合保留源连接关系，
            # 所有片段都必须有可追溯译法。
            # D7D模板随后只匹配有限参数形状，
            # 捕获尺寸与角度原字面直接带入。
            # 均通过同一enhancement最终校验。
            if self.enable_d7g:
                sequence = compose_sequence(source, self.glossary)
                if sequence is not None:
                    result = self._validate_phrase_enhancement(source, sequence)
                    resolved[index] = (TextTranslation(result.translated_text,
                        (*result.warnings,'machine_translation_unreviewed:机器译文未经人工审核',
                         'd7g_composition_review_pending'),result.route,result.evidence)
                        if translation_is_available(result) else result)
                    continue
            if self.generalization_knowledge is not None:
                generalized = self.generalization_knowledge.resolve_template(source)
                if generalized is not None:
                    enhancement = PhraseEnhancement(
                        generalized.translated_text,
                        (PhraseEvidence(
                            source, generalized.translated_text, generalized.entry_id,
                            "d7d", generalized.match_kind,
                        ),),
                        0, 0, 0,
                    )
                    resolved[index] = self._validate_phrase_enhancement(source, enhancement)
                    continue
            # 短语记忆是最后的专业复用机会，
            # 它可通过回调翻译未解析片段。
            # 回调仍施加token上限与模型守卫，
            # 任一片段失败会放弃该组合。
            # 全部专业路径都未解决才排队模型。
            if self.phrase_memory is not None:
                enhancement = self.phrase_memory.enhance(
                    source, self._translate_phrase_segment,
                )
                if enhancement is not None:
                    resolved[index] = self._validate_phrase_enhancement(source, enhancement)
                    continue
            # 实际模型token数超上限便原文回退，
            # 不截断长英文来提高表面覆盖率。
            # 通过后保留原始完整字符串排队，
            # 由后端一次按顺序处理剩余批次。
            # 规则命中的输入不支付额外模型推理。
            self.load()
            if self.backend.count_input_tokens(source) > self.max_input_tokens:
                resolved[index] = _fallback(source, "source_exceeds_model_limit")
                continue
            model_indices.append(index)
            model_inputs.append(source)

        # 模型返回按model_indices逐项回填，
        # 其它规则结果保持原位置不动。
        # 完整批次必须每个输入都有结果，
        # 不能过滤失败项造成坐标错位。
        # Final重试只在全部基线路线之后，
        # 不会覆盖已经可用的合格结果。
        if model_inputs:
            values = self._translate_model_batch(model_inputs)
            for index, value in zip(model_indices, values):
                resolved[index] = value
        if any(result is None for result in resolved):
            raise TranslationInferenceError("translation batch did not resolve every input")
        if self.enable_final_optimization:
            self._recover_qualified_candidates(source_texts,resolved)
        if getattr(self,'enable_convergence_quality',False):
            from .translation_convergence import repair_entity_count
            for i,(source,result) in enumerate(zip(source_texts,resolved)):
                if result is None or not translation_is_available(result):continue
                # “两种”到“两处”的修复范围有限，
                # 只针对明确实体特征计数语法。
                # 修后再跑全部最终守卫，
                # 不通过仍回退而非保留坏中文。
                # 新增机器证据单独说明计数绑定，
                # 不把措辞修正包装成新人工批准。
                repaired=repair_entity_count(source,result.translated_text)
                if repaired!=result.translated_text:
                    warnings=self._final_validation_warnings(source,repaired,include_terms=True)
                    resolved[i]=(_fallback(source,*warnings) if warnings else replace(result,
                        translated_text=repaired,route='hybrid_memory',
                        warnings=(*result.warnings,'machine_translation_unreviewed:机器译文未经人工审核',
                                  'convergence_physical_count_review_pending'),
                        evidence=(*result.evidence,PhraseEvidence(source,repaired,None,'convergence','physical_entity_count_binding'))))
        return tuple(result for result in resolved if result is not None)

    def _recover_qualified_candidates(self, sources, resolved) -> None:
        """Retry only failed, auditable inputs with a strict same-model budget.

        A known leaf merely makes a retry relevant; it never certifies the
        unknown sentence. Every full hypothesis passes all original and new
        gates. Never retry policy holds or overwrite already available results.
        """
        from .engineering_relations import assess
        method=getattr(self.backend,'translate_candidates',None)
        if method is None or self.qualified_proposals is None:
            return
        # 同模型重试只挑当前失败条目，
        # 策略挂起、混合语言、超限均排除。
        # 源必须有关系覆盖或合资格提案，
        # 且输入不超过96个真实token。
        # 每请求最多挑两条，
        # 候选预算在后端仍受上限限制。
        selected=[]
        for i,(source,current) in enumerate(zip(sources,resolved)):
            if current is None or translation_is_available(current):continue
            if any(w.startswith(('v2_runtime_hold:','mixed_language_constraints_preserved',
                                 'source_exceeds_model_limit')) for w in current.warnings):continue
            if self._final_source_recovery_blocked(source):continue
            if not (assess(source,source).source_supported or self.qualified_proposals.plan(source) is not None):continue
            if self.backend.count_input_tokens(source)>96:continue
            selected.append(i)
            if len(selected)==2:break
        if not selected:return
        try:
            # 额外候选数量仍须与选中输入一致，
            # 推理失败保留原回退并添unavailable。
            # 不在异常时重新联网或换引擎，
            # 也不误报候选生成已成功。
            candidates=method([sources[i] for i in selected])
            if len(candidates)!=len(selected):raise TranslationInferenceError('candidate batch count mismatch')
        except TranslationInferenceError:
            for i in selected:
                resolved[i]=replace(resolved[i],warnings=(*resolved[i].warnings,'bounded_candidates_unavailable'))
            return
        for i,rows in zip(selected,candidates):
            for rank,generation in enumerate(rows):
                candidate=self._validate_generation(sources[i],generation)
                if translation_is_available(candidate):
                    # Mere source keyword relevance is not full semantic
                    # coverage. An unsupported remainder cannot be certified by
                    # a leaf match or by the absence of supplemental warnings.
                    # 已知词叶只证明重试相关性，
                    # 采用还要整句关系matched，
                    # 或完整qualified语法无警告。
                    # 不能凭新增中文含某个术语，
                    # 就让此前未知整句突破守卫。
                    # 候选rank留在警告中供审计。
                    relation=assess(sources[i],candidate.translated_text)
                    qualified=self.qualified_proposals.assess(sources[i],candidate.translated_text)
                    if not (relation.status=='matched' or (qualified.supported and not qualified.warnings)):
                        continue
                    resolved[i]=replace(candidate,warnings=tuple(dict.fromkeys((*candidate.warnings,
                        'machine_translation_unreviewed:机器译文未经人工审核',
                        f'bounded_same_model_candidate:{rank+1}'))))
                    break

    def _final_source_recovery_blocked(self, source: str) -> bool:
        """A new candidate cannot override the original source-ambiguity gates."""
        from .mechanical_instructions import ambiguity_warning
        from .translation_convergence import unsupported_input_warning
        ambiguous=ambiguity_warning(source,self.glossary) if self.enable_professional_quality else ()
        # 旧AST只有一个全局事件槽，
        # 窄例外要求两个完整已知断言，
        # 并且关系明确为AND。
        # 操作原子推断不得使用此例外，
        # 否定范围也不能借此解除。
        # 源歧义判断在后处理重试仍有效。
        if ambiguous==('instruction_ambiguous_trailing_event_scope',):
            plan=self.qualified_proposals.plan(source) if self.qualified_proposals is not None else None
            # The old AST has one global event slot. Two uniquely covered
            # complete reviewed assertions provide explicit local scope instead;
            # never use this exception for inferred instruction atoms/negation.
            if plan is not None and plan.relation=='and' and all(not p.instruction for p in plan.operands):
                ambiguous=()
        return bool(ambiguous
                    or (self.enable_integrated_quality and instruction_scope_ambiguous(source,self.glossary))
                    or (self.enable_convergence_quality and unsupported_input_warning(source)))

    def _resolve_marked_rule(self, source: str) -> TextTranslation | None:
        # Only standalone leading reference stars are detachable. Multiplication,
        # numbering, trailing marks and unknown bodies retain the original path.
        match = re.fullmatch(r"(\*{1,4})\s+([A-Za-z].*)", source.strip(), re.S)
        if match is None:
            return None
        # 只拆一至四个前导引用星号，
        # 英文主体必须从词表完整解析。
        # 重组后以原始带星英文做最终校验，
        # 因此标记遗失仍会被发现。
        # V2证据关联原知识ID而非新造批准。
        marker, body = match.groups()
        rule = self.glossary.resolve(body)
        if rule is None:
            return None
        translated = marker + " " + rule.translated_text
        warnings = self._final_validation_warnings(source, translated, include_terms=True)
        if warnings:
            return _fallback(source, *warnings)
        return TextTranslation(
            translated, (), "hybrid_memory",
            self._v2_rule_evidence(source, rule, match_kind='marked_reviewed_body') or (PhraseEvidence(source, translated, rule.entry_id,
                            "v1" if rule.entry_id.startswith("mgt-") else "d6",
                            "marked_reviewed_body"),),
        )

    # 仅candidate中的V1 mgt词条可关联V2，
    # D6规则不伪装成迁移知识。
    # 找不到真实source_ref则返回空证据，
    # 不按相似中文猜测knowledge_id。
    def _v2_rule_evidence(self, source, rule, *, match_kind='normalized_full_exact'):
        if self.knowledge_mode != 'candidate' or not rule.entry_id.startswith('mgt-'):
            return ()
        k = self.knowledge_runtime.knowledge_for_ref('v1_raw:' + rule.entry_id)
        return (PhraseEvidence(source, rule.translated_text, rule.entry_id, 'knowledge_v2', match_kind, k['id']),) if k else ()

    def _audit_knowledge(self, source, result, *, entry=None, context=None):
        """One extra indexed candidate lookup; never runs OCR or a second model."""
        runtime = self.knowledge_runtime
        if runtime is None:
            return result
        from .knowledge_v2_runtime import KnowledgeUsage
        from .knowledge_v2 import validate_context
        # 审计只增加索引查找与纯校验，
        # 不会再次OCR或运行第二个模型。
        # 已有V1证据可补真实knowledge_id，
        # 不是把未采用的候选说成已命中。
        # 耗时累计到独立knowledge_audit_seconds。
        started = perf_counter()
        evidence = list(result.evidence)
        if self.knowledge_mode == 'candidate':
            for i,e in enumerate(evidence):
                k = runtime.knowledge_for_ref('v1_raw:' + str(e.entry_id))
                if k and e.knowledge_id is None:
                    evidence[i] = replace(e, knowledge_id=k['id'])
        # 全局精确与局部实例分开查询，
        # 实例要求源字面与entry原文一致。
        # 页、框、审核资格再次核对，
        # 避免外部context跨条目借用。
        # reason保留没有授权候选的原因。
        candidate = runtime.exact(source)
        reason = runtime.policy_hold(source) or 'no_authorized_exact_or_instance'
        if candidate is None and context is not None:
            validate_context(context)
            valid = (entry is not None and source == entry.source_text and context.page == entry.page_index + 1 and context.bounds ==
                {k: getattr(entry.bounds,k) for k in ('min_x','min_y','max_x','max_y')})
            if valid:
                records = runtime.review_overlay.match(context, entry.source_text)
                target, reason = runtime.review_overlay.translation_candidate(records)
                if target is not None and runtime.review_eligible(records):
                    from .knowledge_v2 import KnowledgeMatch
                    ref = 'review:' + records[0]['instance_id']
                    k = runtime.knowledge_for_ref(ref)
                    candidate = KnowledgeMatch(k['id'], k['target'], (ref,), 'historical_instance')
        decision = 'held' if runtime.policy_hold(source) else 'no_candidate'
        if candidate:
            rejected = self._final_validation_warnings(source, candidate.target, include_terms=True)
            if self.enable_engineering_symbol_risk:
                rejected = (*rejected, *assess_source_risk(source).warning_codes)
            adopted = any(e.knowledge_id == candidate.knowledge_id and e.origin == 'knowledge_v2' for e in evidence)
            if candidate.scope == 'historical_instance':
                adopted = any(w.startswith('user_review_overlay:') and w.endswith(':used') for w in result.warnings)
            # shadow_candidate仅报告可比较候选，
            # candidate还区分adopted与not_selected。
            # 译文碰巧相同不代表来源已采用，
            # 必须检查真实evidence或overlay使用标记。
            # 拒绝原因来自本次完整最终守卫。
            decision = ('rejected' if rejected else 'shadow_candidate' if self.knowledge_mode == 'shadow' else
                        'adopted' if adopted and result.translated_text == candidate.target else 'not_selected')
            reason = ','.join(rejected) if rejected else ('complete_product_gates_passed' if adopted else 'existing_candidate_priority')
        trace = KnowledgeUsage(self.knowledge_mode, runtime.fingerprint,
            candidate.knowledge_id if candidate else None, candidate.source_refs if candidate else (),
            candidate.scope if candidate else 'evidence_only', decision,
            candidate.target if candidate else None, reason)
        warnings = result.warnings
        if self.knowledge_mode == 'candidate':
            # Gate again after D8/context selection so a held literal cannot
            # be reintroduced by a later candidate or compatibility algorithm.
            hold = runtime.policy_hold(source)
            if hold:
                result = _fallback(source, 'v2_runtime_hold:' + hold)
                warnings = result.warnings
            used = tuple(dict.fromkeys(e.knowledge_id for e in evidence if e.knowledge_id and e.knowledge_id.startswith('MDTK2-')))
            # 含模型片段的组合必须提示机器待审，
            # 粗糙度新组合也显式保留待审。
            # 纯复用无额外文字不代表来源已变成人工，
            # 最终TranslationResult仍统一machine。
            warnings = (*warnings, 'knowledge_runtime:' + self.runtime_identity,
                        *('v2_knowledge:' + kid for kid in used))
        self.knowledge_audit_seconds += perf_counter() - started
        return replace(result, warnings=warnings, evidence=tuple(evidence), knowledge_trace=(trace,))

    # 片段回调只返回字符串或None，
    # 无法携带编辑跨度及待审状态。
    # 因此语义对齐编辑不进入该路径，
    # 仍用原批次API运行一次完整片段。
    # 上层组合对完整原句再独立校验。
    def _translate_phrase_segment(self, source: str) -> str | None:
        if self.knowledge_mode == 'candidate' and self.knowledge_runtime.policy_hold(source):
            return None
        self.load()
        if self.backend.count_input_tokens(source) > self.max_input_tokens:
            return None
        # The memory fragment callback returns a string, so it cannot carry
        # source/target edit spans or pending status. Keep its original path;
        # new editing is restricted to complete model-source requests.
        generations = self.backend.translate_batch((source,))
        if len(generations) != 1:
            raise TranslationInferenceError(
                f"translation result count {len(generations)} does not match input count 1"
            )
        result = self._validate_generation(source, generations[0])
        return result.translated_text if translation_is_available(result) else None

    def _translate_model_batch(self, sources: Sequence[str]) -> tuple[TextTranslation, ...]:
        """An optional local edit can propose Chinese, never weaken a full gate.

        Attention is requested during this same greedy call. No re-decoding,
        training, larger beam, foreign model, or source-derived coordinates.
        The new path never runs for existing exact/memory resolutions.
        """
        # 只有存在受控术语且源不在恢复禁区，
        # 才请求同次贪心生成的注意力。
        # 批次仍全部返回原始generation，
        # 不为局部术语重新解码或扩大beam。
        # 没有编辑资格的条目保持原验证路径。
        editor = self.semantic_editor
        method = getattr(self.backend, 'translate_aligned_batch', None)
        eligible = [bool(editor and editor.terms(s) and not self._final_source_recovery_blocked(s))
                    for s in sources] if editor is not None else []
        use_alignment = method is not None and any(eligible)
        aligned = method(sources) if use_alignment else None
        generations = (tuple(a.generation for a in aligned) if aligned is not None
                       else self.backend.translate_batch(sources))
        if len(generations) != len(sources):
            raise TranslationInferenceError('translation result count does not match input count')
        result = []
        for i, (source, generation) in enumerate(zip(sources, generations)):
            original = self._validate_generation(source, generation)
            if not use_alignment or not eligible[i]:
                result.append(original)
                continue
            # 先保存原始贪心假设作为证据，
            # 局部编辑只提供一个中文提案。
            # 强注意力也可能指向错误动作或对象，
            # 因此不是语义对齐正确证明。
            # 完整关系覆盖与最终守卫同时通过，
            # 才可采用编辑后的整句。
            proposal = editor.edit(source, aligned[i])
            raw = PhraseEvidence(source, generation.text, None, 'machine',
                                 'semantic_edit_raw_greedy', 'semantic-edit.1')
            adopted = None
            if proposal is not None:
                candidate = self._validate_generation(source, replace(generation, text=proposal.text))
                # A strong attention block is not a semantic alignment proof.
                # It can still point to the wrong action/object in a fluent
                # target. Local term presence plus old keyword gates is not
                # enough to lift a previously failed unknown sentence. Require
                # independently complete, already supported relation coverage;
                # do not grow an action blacklist to conceal this limitation.
                from .engineering_relations import assess
                relations = assess(source, candidate.translated_text)
                qualified = (self.qualified_proposals.assess(source, candidate.translated_text)
                             if self.qualified_proposals is not None else None)
                complete = (relations.status == 'matched' or
                            (qualified is not None and qualified.supported and not qualified.warnings))
                if translation_is_available(candidate) and complete:
                    # 编辑证据记录源和目标字符区间，
                    # 以及真实使用的术语ID和替换文字。
                    # 跨度来自模型文本而非图纸坐标，
                    # 不能据此修改OCR区域几何。
                    # 结果仍保留局部术语与全文未确认提示。
                    evidence = tuple(PhraseEvidence(
                        source[e.source_span[0]:e.source_span[1]], e.after,
                        e.term.entry_id, 'machine',
                        f'semantic_edit_source:{e.source_span[0]}:{e.source_span[1]}:'
                        f'target:{e.target_span[0]}:{e.target_span[1]}',
                        'semantic-edit.1') for e in proposal.edits)
                    adopted = replace(candidate, evidence=(*candidate.evidence, raw, *evidence),
                        warnings=(*candidate.warnings, 'semantic_edit_local_terms_only',
                                  'semantic_edit_full_meaning_unconfirmed'))
            result.append(adopted or replace(original, evidence=(*original.evidence, raw),
                warnings=(*original.warnings, 'semantic_edit_no_admissible_edit')))
        return tuple(result)

    # 所有复用或组合输出先去两端空白，
    # 空结果与异常重复直接安全回退。
    # 开完整术语模式时累计最终警告，
    # 旧兼容模式保持原先首失败行为。
    # 只有输出内容与工程守卫都通过，
    # 才把原片段evidence附到hybrid结果。
    def _validate_phrase_enhancement(
        self, source: str, enhancement: PhraseEnhancement,
    ) -> TextTranslation:
        translated = enhancement.translated_text.strip()
        if not translated:
            return _fallback(source, "phrase_memory_output_empty")
        if _has_excessive_repetition(translated):
            return _fallback(source, "phrase_memory_output_repetition")
        validation_warnings = (
            self._final_validation_warnings(source, translated, include_terms=True)
            if self.validate_hybrid_terms
            else self._legacy_validation_warnings(source, translated, include_terms=False)
        )
        if validation_warnings:
            return _fallback(source, *validation_warnings)
        warnings = (
            ("machine_translation_unreviewed:机器译文未经人工审核",)
            if enhancement.model_segment_count or any(
                item.match_kind == 'roughness_limit' for item in enhancement.evidence) else ()
        )
        if any(item.match_kind == 'roughness_limit' for item in enhancement.evidence):
            warnings += ('v2_quality_composition_review_pending',)
        return TextTranslation(
            translated, warnings, "hybrid_memory", enhancement.evidence,
        )

    # 模型健康检查依次拒绝空输出、无文本，
    # 达到长度上限和异常长重复。
    # 长英文只剩很少中文时保守拒绝，
    # 避免模型流畅地遗漏大段说明。
    # 这些启发式不构成专业正确率，
    # 通过后仍运行工程与术语守卫。
    def _validate_generation(self, source: str, generation: Generation) -> TextTranslation:
        translated = generation.text.strip()
        if not translated:
            return _fallback(source, "model_output_empty")
        if re.search(r"[A-Za-z]", source) \
                and re.search(r"[A-Za-z\u4e00-\u9fff]", translated) is None:
            return _fallback(source, "model_output_no_text")
        if generation.hit_output_limit:
            return _fallback(source, "model_output_truncated")
        if _has_excessive_repetition(translated):
            return _fallback(source, "model_output_repetition")
        # 长度比使用英文字符与译文内容字符，
        # 不是逐词语义对齐或人工评分。
        # 只对足够长的英文启用短译检测，
        # 不会用该比例补全缺失内容。
        # 异常保留原文并附可定位原因。
        source_letters = len(re.findall(r"[A-Za-z]", source))
        target_content = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", translated))
        if source_letters >= 80 and target_content < source_letters * 0.20:
            return _fallback(source, "model_output_suspiciously_short")
        validation_warnings = (
            self._final_validation_warnings(source, translated, include_terms=True)
            if self.validate_hybrid_terms
            else self._legacy_validation_warnings(source, translated, include_terms=True)
        )
        if validation_warnings:
            return _fallback(source, *validation_warnings)
        return TextTranslation(
            translated,
            ("machine_translation_unreviewed:机器译文未经人工审核",),
            "model",
        )

    def _final_validation_warnings(
        self, source: str, translated: str, *, include_terms: bool,
    ) -> tuple[str, ...]:
        # 缓存键同时包含完整源目标字符串，
        # 以及影响守卫的开关和知识对象身份。
        # include_terms不同不能复用旧检查，
        # 不同词库实例也不能共享结果。
        # 缓存对象只在本次请求范围有效。
        cache = self._validation_cache.get() if self.enable_v2_quality_optimization or self.reviewed94 is not None else None
        key = (source, translated, include_terms, self.enforce_extended_semantics,
               self.enable_d8a, self.enable_d7f, self.enable_professional_quality,self.enable_reference_quality,self.enable_convergence_quality, id(self.glossary),
               id(self.generalization_knowledge), id(self.professional_knowledge))
        if cache is not None and key in cache:
            return cache[key]
        result = self._compute_validation_warnings(source, translated, include_terms=include_terms)
        if cache is not None:
            cache[key] = result
        return result

    def _compute_validation_warnings(
        self, source: str, translated: str, *, include_terms: bool,
    ) -> tuple[str, ...]:
        # 完整守卫合并多层独立警告，
        # 字面量结构与方向条件先分别核验。
        # 词表要求的术语以及D7E限制，
        # 不能被关系语法matched替代。
        # Professional还验证角色与词义，
        # Reference继续检查明确引用关系。
        literal = validate_immutable_literals(source, translated)
        semantic = validate_engineering_semantics(
            source, translated, extended=self.enforce_extended_semantics,
            allow_numbering_labels=self.enable_d8a,
        )
        terms = self.glossary.validate_terms(
            source, translated, extra_targets=self._extra_term_targets,
        ) if include_terms else ()
        professional = (
            self.professional_knowledge.validate_output(source, translated)
            if self.professional_knowledge is not None else ()
        )
        instruction_warnings = ()
        if self.enable_professional_quality:
            from .mechanical_instructions import validate_instruction, lexical_warnings
            instruction_warnings = (*validate_instruction(source, translated, self.glossary),
                                    *lexical_warnings(source, translated, self.glossary))
        if self.enable_reference_quality:
            from .translation_reference_safety import validate_reference_relations
            instruction_warnings=(*instruction_warnings,*validate_reference_relations(source,translated,self.glossary))
        if self.argos_core is not None:
            instruction_warnings=(*instruction_warnings,*self.argos_core.validate(source,translated))
        # 新覆盖语法只验证它负责的范围，
        # 既有完整已审核表达仍沿原门禁。
        # 多词叶或类型指令才追加qualified检查，
        # 不把所有已知中文改成严格字符串相等。
        # relation警告仍合并进最终拒绝结果。
        if self.enable_final_optimization or self.enable_product_core:
            from .engineering_relations import assess as assess_relations
            relation=assess_relations(source,translated)
            plan=self.qualified_proposals.plan(source)
            # Existing reviewed full-expression equivalents continue under the
            # original gates. The new coverage grammar validates its generated
            # multi-leaf/typed instructions, not every wording of a known exact.
            qualified=(self.qualified_proposals.validate(source,translated)
                if relation.status!='matched' and plan is not None
                and (plan.relation!='single' or any(p.instruction for p in plan.operands)) else ())
            instruction_warnings=(*instruction_warnings,*relation.warnings,*qualified)
        # 按固定层序聚合并去重警告，
        # D7F字段词义与Integrated关系，
        # 各按其真实启用开关加入。
        # 没有任何新层可取消先前警告，
        # 最终空列表仅表示这些守卫未发现问题。
        return tuple(dict.fromkeys((
            *literal.warning_codes,
            *semantic.warning_codes,
            *terms,
            *professional,
            *instruction_warnings,
            *(validate_field_terms(source, translated) if self.enable_d7f else ()),
            *(validate_instruction_relations(source,translated,self.glossary) if self.enable_integrated_quality else ()),
        )))

    # 旧D7B保留首个失败层的诊断口径，
    # 先字面量再基本语义最后术语。
    # D8A编号字段才允许对应NO.解释，
    # 分词器残留▁仍单独拒绝。
    # 此兼容路径不是放宽候选新门禁，
    # 而是保持旧入口的可回归行为。
    def _legacy_validation_warnings(
        self, source: str, translated: str, *, include_terms: bool,
    ) -> tuple[str, ...]:
        """Preserve D7-B's first-failing-gate diagnostics when D7-C is off."""
        literal = validate_immutable_literals(source, translated)
        if not literal.is_valid:
            return literal.warning_codes
        semantic = validate_engineering_semantics(
            source, translated, extended=False,
            allow_numbering_labels=self.enable_d8a,
        )
        if not semantic.is_valid:
            return semantic.warning_codes
        if self.enable_d8a and is_numbering_label(source) and "▁" in translated:
            return ("model_output_tokenizer_artifact",)
        if include_terms:
            return self.glossary.validate_terms(source, translated)
        return ()
