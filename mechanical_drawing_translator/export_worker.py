# mechanical_drawing_translator/export_worker.py
"""在 Qt 后台线程中执行标注导出，避免阻塞主界面。"""

from collections.abc import Callable

from PySide6.QtCore import QThread, Signal

from .export_service import AnnotationExportService, ExportRequest, ExportResult


class ExportTask(QThread):
    """单次导出任务；服务实例在线程内创建和使用。"""

    result_ready = Signal(object, object)
    failed = Signal(object, str)

    # 请求已包含共享布局
    # Worker不再读取控件状态
    # 工厂保存到线程运行时调用
    # 单次导出只拥有一个请求
    # 正式资格仍由服务层检查
    def __init__(
        self,
        request: ExportRequest,
        service_factory: Callable[[], AnnotationExportService],
        parent=None,
    ):
        super().__init__(parent)
        self.request = request
        self._service_factory = service_factory

    # 在线程内创建导出服务
    # 读写副本不阻塞主界面
    # UI不能直接碰服务句柄
    # 成功与失败携带原请求
    # 窗口据此核验当前任务
    # 捕获文件和解析异常
    # 错误文本交回状态栏
    # 完成清理由QThread.finished触发
    # 此处没有强制取消写盘
    def run(self) -> None:
        try:
            result: ExportResult = self._service_factory().export(self.request)
        except Exception as exc:  # UI 边界必须把文件/解析异常转换为可见状态。
            self.failed.emit(self.request, str(exc))
        else:
            self.result_ready.emit(self.request, result)
