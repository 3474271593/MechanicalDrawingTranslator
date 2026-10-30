# mechanical_drawing_translator/translation_worker.py
"""Run the offline translation service on one reusable Qt background thread."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, QThread, Signal, Slot, QMetaObject, Qt

from .translation_results import TranslationResult
from .translation_service import TranslationConfig, TranslationService
from .translation_targets import TargetEntry
from .user_review_overlay import ReviewContext
from pathlib import Path


# 将入口选项转成服务配置
# V1安全知识始终保留
# D7增强由显式配置控制
# 审核overlay路径独立传入
# V2模式与manifest一并传递
# 质量候选各有独立开关
# product-core也进入服务配置
# semantic-edit仅试验入口可启
# 模型路径由packaged解析
# 不在这里硬编码开发机目录
# 返回服务尚可保持懒加载
# 实际load在Worker所有者线程
def make_translation_service(*, enable_d7e: bool = False,
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
                             enable_semantic_edit: bool = False) -> TranslationService:
    """Build the packaged translator; D7-E is enabled only for an explicit trial."""
    config = TranslationConfig.packaged(
        enable_v1=True,
        enable_phrase_enhancement=enable_d7e,
        enable_d7c_enhancement=enable_d7e,
        enable_d7d_enhancement=enable_d7e,
        enable_d7e_enhancement=enable_d7e,
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
    return TranslationService.from_config(config)


# entries冻结为目标元组
# 其中保留SourceSpan身份
# runtime_identity绑定配置
# 不能混用别的知识包结果
# pdf_path供实例审核匹配
# 图片也传实际源文件路径
@dataclass(frozen=True)
class TranslationRequest:
    request_id: int
    document_id: str
    page_index: int
    entries: tuple[TargetEntry, ...]
    pdf_path: str | None = None
    runtime_identity: str = 'legacy'


class _TranslationWorker(QObject):
    result_ready = Signal(object, object)
    details_ready = Signal(object, object)
    failed = Signal(object, str)
    finished = Signal(object)
    preparation_finished = Signal(str, str)

    def __init__(self, service_factory: Callable[[], TranslationService]) -> None:
        super().__init__()
        self._service_factory = service_factory
        self._service: TranslationService | None = None

    # 缓存服务只属于本Worker
    # 新建后先核验运行身份
    # 不匹配结果不留在缓存
    # 严格prepare不接受缺失身份
    # 旧集成服务可沿用legacy
    # 校验失败要求匹配配置重启
    def _service_for(self, runtime_identity: str, *, strict: bool = False):
        # Do not retain a mismatched factory result. Legacy test/integration
        # services may omit identity; explicit candidate preparation may not.
        service = self._service if self._service is not None else self._service_factory()
        identity = getattr(service, 'runtime_identity', None if strict else 'legacy')
        if identity != runtime_identity:
            raise ValueError('translation runtime identity mismatch; restart with matching configuration')
        self._service = service
        return service

    # 预备加载没有图纸输入
    # 不产生任何页译文
    # 错误以message返回
    # 加载失败不伪造准备成功
    # 后续正常请求仍可重试
    # 始终在模型所有者线程执行
    @Slot(str)
    def prepare(self, runtime_identity: str) -> None:
        """Load only on the existing owner thread, without any document inputs."""
        message = ''
        try:
            self._service_for(runtime_identity, strict=True).load()
        except Exception as exc:
            message = str(exc) or type(exc).__name__
        self.preparation_finished.emit(runtime_identity, message)

    # 送译前再执行源文策略
    # 尺寸与代号以原文结果保留
    # 保留结果仍标machine
    # 先筛可译目标再加载模型
    # 全为保留项可直接返回
    # finally仍然发finished
    # 审核实例由原图上下文定位
    # 页码在ReviewContext改为一始
    # 边界使用原始源坐标
    # 随机OCR ID不是审核ID
    # 多条匹配不擅自选批准项
    # 唯一匹配才填instance_id
    # 复合语句留给服务继续核验
    # 知识运行时返回详细追踪
    # 普通服务可只返回结果
    # 保留项按请求顺序并回
    # 所有异常跨线程变失败信号
    # 细节信号不提升审核来源
    # 细节先于最终结果交付
    # 窗口再次核验完整目标元组
    @Slot(object)
    def process(self, request: TranslationRequest) -> None:
        try:
            details = None
            from .source_text_policy import english_translation_eligibility
            policies={entry.group_id:english_translation_eligibility(entry.source_text) for entry in request.entries}
            entries=tuple(entry for entry in request.entries if policies[entry.group_id].eligible)
            preserved={entry.group_id:TranslationResult(1,entry.document_id,entry.page_index,
                entry.group_id,entry.region_ids,entry.source_text,entry.source_text,'machine',
                ('source_text_preserved','source_policy:'+policies[entry.group_id].reason))
                for entry in request.entries if not policies[entry.group_id].eligible}
            if not entries:
                self.result_ready.emit(request,tuple(preserved[e.group_id] for e in request.entries))
                return
            self._service_for(request.runtime_identity)
            if getattr(self._service, 'review_overlay', None) is not None:
                if not request.pdf_path:
                    raise ValueError('审核资源已加载，但翻译请求缺少实际PDF路径')
                # Keep ambiguous instance_id empty. The existing matcher returns
                # attested IDs from original source/page/bounds, including D8C
                # constituent evidence; random OCR IDs are not review IDs.
                contexts = tuple(ReviewContext(
                    request.pdf_path, entry.page_index + 1,
                    {k: getattr(entry.bounds, k) for k in
                     ('min_x', 'min_y', 'max_x', 'max_y')},
                ) for entry in entries)
                # Resolve an ID only through the existing exact matcher. Do not
                # filter by approval: multiple records must remain ambiguous,
                # and D8C constituents are still resolved by the service.
                bound_contexts = []
                for entry, context in zip(entries, contexts):
                    records = self._service.review_overlay.match(context, entry.source_text)
                    bound_contexts.append(replace(context, instance_id=records[0]['instance_id'])
                                          if len(records) == 1 else context)
                details = self._service.translate_entries_with_details(
                    entries, review_contexts=tuple(bound_contexts),
                )
                results = tuple(item.result for item in details)
            else:
                if getattr(self._service, 'knowledge_runtime', None) is not None:
                    details = self._service.translate_entries_with_details(entries)
                    results = tuple(item.result for item in details)
                else:
                    results = self._service.translate_entries(entries)
            if preserved:
                combined={**preserved,**{result.group_id:result for result in results}}
                results=tuple(combined[e.group_id] for e in request.entries)
        except Exception as exc:
            self.failed.emit(request, str(exc) or type(exc).__name__)
        else:
            if details is not None and getattr(self._service, 'knowledge_runtime', None) is not None:
                self.details_ready.emit(request, details)
            self.result_ready.emit(request, results)
        finally:
            self.finished.emit(request)

    # 模型在所属线程释放引用
    # 主线程不直接清原生后端
    # 线程停止前必须完成释放
    @Slot()
    def release_service(self) -> None:
        """Release the offline backend on its owner thread before that thread stops."""
        self._service = None


class TranslationSession(QObject):
    """Own one worker thread and reject stale/cancelled translation deliveries."""

    result_ready = Signal(object, object)
    details_ready = Signal(object, object)
    failed = Signal(object, str)
    finished = Signal(object)
    preparation_finished = Signal(str, str)
    _execute = Signal(object)
    _prepare = Signal(str)

    # 一个窗口一个翻译会话
    # 一个会话一个常驻Worker
    # 模型可跨页跨图复用
    # 预备加载和翻译同线程串行
    # prepare不会抢占process
    # 活动请求与准备状态分离
    # runtime身份固定于会话
    # 服务严格核验同一身份
    # 信号回到Session主线程
    # 取消结果与细节同时屏蔽
    # 线程结束回收Worker对象
    def __init__(
        self,
        service_factory: Callable[[], TranslationService] = make_translation_service,
        parent: QObject | None = None,
        *, runtime_identity: str = 'legacy',
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = _TranslationWorker(service_factory)
        self.runtime_identity = runtime_identity
        self._worker.moveToThread(self._thread)
        self._active_request: TranslationRequest | None = None
        self._cancelled_request_ids: set[int] = set()
        self._next_request_id = 1
        self._shutdown = False
        self._is_preparing = False
        self._prepared_identity: str | None = None
        self._execute.connect(self._worker.process)
        self._prepare.connect(self._worker.prepare)
        self._worker.result_ready.connect(self._forward_result)
        self._worker.details_ready.connect(self._forward_details)
        self._worker.failed.connect(self._forward_failure)
        self._worker.finished.connect(self._finish_request)
        self._worker.preparation_finished.connect(self._finish_preparation)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    @property
    def active_request(self) -> TranslationRequest | None:
        return self._active_request

    @property
    def is_preparing(self) -> bool:
        return self._is_preparing

    # 预备加载是可选优化
    # 重复预备返回False
    # 准备中不重复排队
    # 已有翻译也不插入准备
    # 同身份已准备无需再加载
    # 后续submit可排在准备后
    # 此返回值不表示质量验收
    def prepare(self) -> bool:
        """Queue one opt-in load; a later submit queues behind it on this worker."""
        if self._shutdown:
            raise RuntimeError('翻译会话已经关闭')
        if (self._is_preparing or self._active_request is not None
                or self._prepared_identity == self.runtime_identity):
            return False
        self._is_preparing = True
        self._prepare.emit(self.runtime_identity)
        return True

    # 关闭或忙碌直接拒绝
    # 空目标不能占用工作线程
    # 复制元组隔离外部容器改动
    # 每条必须匹配文档和页码
    # 生成新ID后登记活动任务
    # 运行身份随请求固定
    # 信号排队送入模型线程
    # 页面队列在窗口层管理
    def submit(
        self, document_id: str, page_index: int, entries: Sequence[TargetEntry],
        *, pdf_path: str | None = None,
    ) -> TranslationRequest:
        if self._shutdown:
            raise RuntimeError("翻译会话已经关闭")
        if self._active_request is not None:
            raise RuntimeError("已有翻译任务正在运行")
        values = tuple(entries)
        if not values:
            raise ValueError("翻译请求没有候选条目")
        if any(
            item.document_id != document_id or item.page_index != page_index
            for item in values
        ):
            raise ValueError("翻译条目与请求页面不一致")
        request = TranslationRequest(
            self._next_request_id, document_id, page_index, values, pdf_path, self.runtime_identity,
        )
        self._next_request_id += 1
        self._active_request = request
        self._execute.emit(request)
        return request

    # 仅屏蔽当前结果交付
    # 不强杀CTranslate2调用
    # 预备加载由独立状态跟踪
    # 关闭仍需等准备结束
    def cancel_active(self) -> None:
        if self._active_request is not None:
            self._cancelled_request_ids.add(self._active_request.request_id)

    # 翻译和准备任一忙则拒绝
    # 空闲后先释放后端
    # BlockingQueued确保释放完成
    # 随后退出线程事件循环
    # wait保证窗口销毁前结束
    def shutdown(self) -> bool:
        if self._active_request is not None or self._is_preparing:
            return False
        if not self._shutdown:
            QMetaObject.invokeMethod(self._worker, "release_service", Qt.BlockingQueuedConnection)
            self._shutdown = True
            self._thread.quit()
            self._thread.wait()
        return True

    # 准备结束不能清翻译请求
    # 排在其后的请求仍有效
    # 只有无错误才记录prepared
    # 失败保留下一次重试机会
    # 通知窗口继续延迟关闭
    @Slot(str, str)
    def _finish_preparation(self, runtime_identity: str, message: str) -> None:
        self._is_preparing = False
        self._prepared_identity = runtime_identity if not message else None
        # Separate completion cannot clear a queued translation request or
        # deliver page results. Failure leaves normal submit/load retry intact.
        self.preparation_finished.emit(runtime_identity, message)

    # 知识追踪也遵守软取消
    # 否则旧细节可能污染新页面
    @Slot(object, object)
    def _forward_details(self, request, details) -> None:
        if request.request_id not in self._cancelled_request_ids:
            self.details_ready.emit(request, details)

    # 取消集合只控制交付
    # 不更改结果数据和来源
    # 窗口仍核验目标是否变化
    @Slot(object, object)
    def _forward_result(
        self, request: TranslationRequest, results: tuple[TranslationResult, ...],
    ) -> None:
        if request.request_id not in self._cancelled_request_ids:
            self.result_ready.emit(request, results)

    # 取消后的旧失败不干扰UI
    # finished不受该屏蔽影响
    @Slot(object, str)
    def _forward_failure(self, request: TranslationRequest, message: str) -> None:
        if request.request_id not in self._cancelled_request_ids:
            self.failed.emit(request, message)

    # 先回收请求取消标记
    # 只清匹配的活动请求
    # 再通知窗口安全推进
    # 不会由prepare路径代替
    @Slot(object)
    def _finish_request(self, request: TranslationRequest) -> None:
        self._cancelled_request_ids.discard(request.request_id)
        if self._active_request == request:
            self._active_request = None
        self.finished.emit(request)
