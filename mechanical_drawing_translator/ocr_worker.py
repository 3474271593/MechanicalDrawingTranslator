# mechanical_drawing_translator/ocr_worker.py
"""在常驻 Qt 工作线程中串行调用 OCR 服务，不接触界面对象。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot, QMetaObject, Qt

from .ocr_service import OcrDocument, OcrService


# 工厂由Worker首次调用
# 不是窗口构造时加载模型
# compact优先于CPU试验标志
# stage3采用8线程与arena
# 并关闭memory pattern
# 常规引擎默认6线程
# 渲染仍固定216 DPI
# 局部复查单独传开关
# 工程字符核验也单独传递
# 缓存目录仅在显式配置时传入
# 正式product未选final缓存
def make_ocr_service(*, enable_refinement: bool = False,
                     enable_engineering_verification: bool = False,
                     enable_cpu_acceleration: bool = False,
                     enable_compact_memory: bool = False,
                     enable_integrated_quality: bool = False,
                     optimized_graph_cache_dir: str | Path | None = None) -> OcrService:
    """使用已验证的安装包模型和 ONNX Runtime CPU 配置。"""
    from .rapidocr_engine import RapidOcrCpuEngine

    cache_options=({'optimized_graph_cache_dir':optimized_graph_cache_dir}
                   if optimized_graph_cache_dir is not None else {})
    return OcrService(
        (RapidOcrCpuEngine(performance_mode=True, compact_memory=True,**cache_options) if enable_compact_memory
         else RapidOcrCpuEngine(performance_mode=True,**cache_options) if enable_cpu_acceleration
         else RapidOcrCpuEngine(**cache_options)), dpi=216, enable_refinement=enable_refinement,
        enable_engineering_verification=enable_engineering_verification,
        enable_integrated_quality=enable_integrated_quality,
    )


# 请求冻结文档与单页身份
# ID只在当前Session递增
# path是实际源文件路径
# page_index保持零起始
# 不用当前UI页码替代请求页
@dataclass(frozen=True)
class OcrRequest:
    """主线程和 OCR 线程之间传递的不可变单页请求。"""

    request_id: int
    path: Path
    page_index: int
    document_id: str


class _OcrWorker(QObject):
    """只在所属工作线程内创建并使用一个 OCR 服务实例。"""

    result_ready = Signal(object, object)
    failed = Signal(object, str)
    finished = Signal(object)

    def __init__(self, service_factory: Callable[[], OcrService]) -> None:
        super().__init__()
        self._service_factory = service_factory
        self._service: OcrService | None = None

    # 槽函数运行在所属工作线程
    # 同一服务跨请求复用
    # 三个ONNX会话无需每页重建
    # 仅提取请求指定的一页
    # 异常转换为失败信号
    # 成功结果先于finished发出
    # finally始终解除忙碌状态
    # 不得在这里访问窗口控件
    # 取消只抑制结果传递
    # ONNX调用仍安全执行到返回
    @Slot(object)
    def process(self, request: OcrRequest) -> None:
        try:
            if self._service is None:
                self._service = self._service_factory()
            result: OcrDocument = self._service.extract(
                request.path, [request.page_index], document_id=request.document_id,
            )
        except Exception as exc:
            self.failed.emit(request, str(exc) or type(exc).__name__)
        else:
            self.result_ready.emit(request, result)
        finally:
            self.finished.emit(request)

    # 在原所有者线程清引用
    # 先释放引擎再停止事件循环
    # 避免已停线程遗留模型所有权
    @Slot()
    def release_service(self) -> None:
        """Drop native model ownership on this worker's thread, after its last task."""
        self._service = None


class OcrSession(QObject):
    """拥有一个常驻线程，并保证窗口内 OCR 服务和 ONNX 会话复用。"""

    result_ready = Signal(object, object)
    failed = Signal(object, str)
    finished = Signal(object)
    _execute = Signal(object)

    # Session对象留在主线程
    # Worker迁入常驻QThread
    # 跨线程信号排队执行
    # 主线程只提交不可变请求
    # 活动请求限制为一个
    # 取消集合只存请求ID
    # finished负责清理活动引用
    # 线程结束时deleteLater
    # 线程父对象仍是Session
    # 模型懒加载而线程立即启动
    def __init__(
        self,
        service_factory: Callable[[], OcrService] = make_ocr_service,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = _OcrWorker(service_factory)
        self._worker.moveToThread(self._thread)
        self._active_request: OcrRequest | None = None
        self._cancelled_request_ids: set[int] = set()
        self._next_request_id = 1
        self._shutdown = False

        self._execute.connect(self._worker.process)
        self._worker.result_ready.connect(self._forward_result)
        self._worker.failed.connect(self._forward_failure)
        self._worker.finished.connect(self._finish_request)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    @property
    def active_request(self) -> OcrRequest | None:
        return self._active_request

    # 停用会话拒绝新请求
    # 忙碌时不排第二个OCR
    # 页队列由窗口逐页推进
    # 先登记活动请求再发信号
    # 返回请求供窗口二次校验
    # 请求ID不会因取消复用
    def submit(self, path: str | Path, page_index: int, document_id: str) -> OcrRequest:
        if self._shutdown:
            raise RuntimeError("OCR 会话已经关闭")
        if self._active_request is not None:
            raise RuntimeError("已有 OCR 任务正在运行")
        request = OcrRequest(
            self._next_request_id, Path(path), page_index, document_id,
        )
        self._next_request_id += 1
        self._active_request = request
        self._execute.emit(request)
        return request

    # 软取消不终止原生调用
    # 仅把当前请求加入屏蔽集合
    # 仍要等待finished清忙碌
    # 不得立刻销毁工作线程
    def cancel_active(self) -> None:
        """软取消结果传递；已进入 ONNX 的调用仍等待其安全返回。"""
        if self._active_request is not None:
            self._cancelled_request_ids.add(self._active_request.request_id)

    # 忙碌时返回False
    # 调用方在finished后重试
    # 空闲后阻塞调用释放槽
    # 此时Worker事件循环仍运行
    # 释放完成再quit和wait
    # 禁止直接terminate推理线程
    def shutdown(self) -> bool:
        """空闲时停止常驻线程；忙碌时由调用方等待 finished 后重试。"""
        if self._active_request is not None:
            return False
        if not self._shutdown:
            QMetaObject.invokeMethod(self._worker, "release_service", Qt.BlockingQueuedConnection)
            self._shutdown = True
            self._thread.quit()
            self._thread.wait()
        return True

    # 取消后不向窗口交付结果
    # 窗口还校验文档和请求身份
    # 双层检查防止切图污染
    @Slot(object, object)
    def _forward_result(self, request: OcrRequest, result: OcrDocument) -> None:
        if request.request_id not in self._cancelled_request_ids:
            self.result_ready.emit(request, result)

    # 已取消任务的异常不再弹报
    # 真正的结束仍由finished处理
    @Slot(object, str)
    def _forward_failure(self, request: OcrRequest, message: str) -> None:
        if request.request_id not in self._cancelled_request_ids:
            self.failed.emit(request, message)

    # 取消标记用完即删除
    # 只清同一个活动请求
    # 结果被屏蔽也发结束信号
    # 窗口据此推进关闭或页队列
    @Slot(object)
    def _finish_request(self, request: OcrRequest) -> None:
        self._cancelled_request_ids.discard(request.request_id)
        if self._active_request == request:
            self._active_request = None
        self.finished.emit(request)
