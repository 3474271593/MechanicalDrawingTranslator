# mechanical_drawing_translator/document_service.py
"""本地图片和 PDF 的读取与页面渲染，不依赖界面控件。"""

from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QSize, QSizeF
from PySide6.QtGui import QImage

from .models import DocumentInfo
from .image_metadata import RasterMetadata, read_raster

try:
    from PySide6.QtPdf import QPdfDocument
except ImportError as import_error:
    QPdfDocument = None
    PDF_IMPORT_ERROR = str(import_error)
else:
    PDF_IMPORT_ERROR = ""


class DocumentError(Exception):
    """可在界面展示的文档打开或渲染错误。"""


# 句柄由打开操作独占
# UI预览不共享Worker句柄
# 源尺寸不随渲染DPI变化
# 图片已消费EXIF方向
# PDF源尺寸是展示页point
# 关闭后旧请求明确失败
class LoadedDocument:
    """Own one local document handle and expose its displayed source geometry.

    Raster coordinates are EXIF-oriented pixels; PDF coordinates are displayed
    page points. Callers must not confuse pixel density with coordinate scaling.
    Closing releases the handle and makes stale asynchronous requests fail.
    """

    # 图片与PDF句柄二选一
    # 元数据保留色彩和密度信息
    # closed独立标记生命周期
    def __init__(self, info: DocumentInfo, image: QImage | None = None, pdf=None,
                 raster_metadata: RasterMetadata | None = None):
        self.info = info
        self._image = image
        self._pdf = pdf
        self._closed = False
        self.raster_metadata = raster_metadata

    # 先检查页码和关闭状态
    # 图片返回定向后像素尺寸
    # PDF返回页面point尺寸
    # 不能把预览像素当源坐标
    def source_size(self, page_index: int) -> QSizeF:
        """Return layout coordinates, independent from preview render resolution."""
        self._check_page(page_index)
        if self._image is not None:
            return QSizeF(self._image.size())
        return self._pdf.pagePointSize(page_index)

    # 图片直接复用解码结果
    # PDF渲染只生成工作底图
    # 显式target必须非空
    # 默认限制长边与倍率
    # 首帧可在UI快速显示
    # 高清预览由独立线程补充
    # source_size始终不变
    # 渲染失败不返回空图冒充成功
    # 此方法不写入原始图纸
    def render_page(self, page_index: int, target_size: QSize | None = None) -> QImage:
        """Render without changing source files or the annotation coordinate unit.

        Raster data already uses its full oriented resolution. PDF preview size
        is bounded for responsiveness; the page's source size stays in points.
        """
        self._check_page(page_index)
        if self._image is not None:
            return self._image
        page_size = self._pdf.pagePointSize(page_index)
        if page_size.isEmpty():
            raise DocumentError("无法读取 PDF 页面尺寸")
        if target_size is not None:
            if target_size.isEmpty():
                raise DocumentError("PDF 预览渲染尺寸无效")
            image_size = target_size
        else:
            # 首帧最多约 4 MP，避免在 UI 线程渲染大幅面图纸；精细图后台生成。
            scale = min(2.0, 2400.0 / max(page_size.width(), page_size.height()))
            image_size = QSize(max(1, round(page_size.width() * scale)),
                               max(1, round(page_size.height() * scale)))
        image = self._pdf.render(page_index, image_size)
        if image.isNull():
            raise DocumentError(f"PDF 第 {page_index + 1} 页渲染失败")
        return image

    # 关闭检查优先于页号检查
    # 防止访问已释放原生句柄
    # 页码范围是左闭右开
    def _check_page(self, page_index: int) -> None:
        if self._closed:
            raise DocumentError("文档已经关闭")
        if not 0 <= page_index < self.info.page_count:
            raise DocumentError("页面编号超出范围")

    # 允许重复关闭
    # 先释放PDF原生资源
    # 再清空图片和句柄引用
    # 旧异步任务另用独立文档
    def close(self) -> None:
        if not self._closed:
            if self._pdf is not None:
                self._pdf.close()
            self._image = None
            self._pdf = None
            self._closed = True


class DocumentLoader:
    """识别文件类型并创建可浏览的本地文档。"""

    IMAGE_TYPES = {".jpg": "JPG", ".jpeg": "JPG", ".png": "PNG", ".bmp": "BMP"}

    # 扩展名仅决定解析路线
    # 不能证明文件内容有效
    # PDF依赖缺失给可读错误
    # 新UUID对应一次打开
    # PDF失败分支也关闭句柄
    # 图片只接受白名单格式
    # 同一解码器统一EXIF/ICC
    # 避免预览与导出坐标偏移
    # 读取异常转换为DocumentError
    # 源文件保持只读
    # 目录路径不能充当文件
    # 实际颜色限制由解码层拒绝
    @staticmethod
    def open(path: str | Path) -> LoadedDocument:
        """Open a supported local file and translate parser failures for the UI.

        A recognized extension is not evidence of valid contents. PDF resources
        are closed on failure; raster conversions expose warnings in metadata.
        """
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise DocumentError("文件不存在或无法访问")
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            if QPdfDocument is None:
                raise DocumentError(f"当前环境缺少 PySide6.QtPdf，无法预览 PDF：{PDF_IMPORT_ERROR}")
            pdf = QPdfDocument()
            try:
                error = pdf.load(str(file_path))
                if error != QPdfDocument.Error.None_ or pdf.pageCount() < 1:
                    raise DocumentError(f"无法打开 PDF 文件（{error.name}）")
                info = DocumentInfo(str(uuid4()), file_path, "PDF", pdf.pageCount())
                return LoadedDocument(info, pdf=pdf)
            except Exception:
                pdf.close()
                raise
        if suffix not in DocumentLoader.IMAGE_TYPES:
            raise DocumentError("不支持此文件类型；请选择 PDF、JPG、PNG 或 BMP")
        # Preview and export must consume EXIF orientation and density in the
        # same decoder, otherwise a saved annotation can shift on camera images.
        try:
            image, metadata = read_raster(file_path)
        except (OSError, ValueError) as error:
            raise DocumentError(f"图片无法安全读取：{error}") from error
        info = DocumentInfo(str(uuid4()), file_path, DocumentLoader.IMAGE_TYPES[suffix], 1)
        return LoadedDocument(info, image=image, raster_metadata=metadata)
