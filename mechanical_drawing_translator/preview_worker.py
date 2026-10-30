# mechanical_drawing_translator/preview_worker.py
"""后台打开独立 QtPdf 文档并渲染当前页的高分辨率预览。"""

from pathlib import Path

from PySide6.QtCore import QSize, QThread, Signal
from PySide6.QtGui import QImage
from PySide6.QtPdf import QPdfDocument


class PdfPreviewTask(QThread):
    result_ready = Signal(str, int, QImage)
    failed = Signal(str, int, str)

    # 单次高清任务保存源身份
    # target_size复制Qt值对象
    # 不传主窗口文档句柄
    # 结果带文档ID和页码
    # 窗口可丢弃过期预览
    def __init__(self, path: str | Path, document_id: str, page_index: int,
                 target_size: QSize, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.document_id = document_id
        self.page_index = page_index
        self.target_size = QSize(target_size)

    # QPdfDocument在线程内创建
    # load和render也在同线程
    # 前后都检查中断请求
    # 中断不强杀正在渲染的原生库
    # 页面数量再次验证
    # 空图作为可见失败
    # 结果仅升级当前页底图
    # 不重建OCR或布局坐标
    # finally无条件关闭PDF
    # 错误信号也遵守中断状态
    def run(self) -> None:
        # QPdfDocument 在本线程创建与关闭，不跨线程调用主窗口持有的文档。
        pdf = QPdfDocument()
        try:
            if self.isInterruptionRequested():
                return
            error = pdf.load(str(self.path))
            if error != QPdfDocument.Error.None_:
                raise ValueError(f"无法打开 PDF 文件（{error.name}）")
            if not 0 <= self.page_index < pdf.pageCount():
                raise ValueError("PDF 页面编号超出范围")
            image = pdf.render(self.page_index, self.target_size)
            if image.isNull():
                raise ValueError("PDF 高清预览渲染失败")
            if not self.isInterruptionRequested():
                self.result_ready.emit(self.document_id, self.page_index, image)
        except (OSError, ValueError) as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(self.document_id, self.page_index, str(exc))
        finally:
            pdf.close()
