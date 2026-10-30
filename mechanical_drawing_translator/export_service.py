# mechanical_drawing_translator/export_service.py
"""从原始图纸生成经过回读验证的中文标注副本。"""

from __future__ import annotations

import os
import math
import re
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from PySide6.QtCore import QMarginsF, QSizeF
from PySide6.QtGui import (
    QPageLayout, QPageSize, QPainter, QPdfWriter,
)

from .annotation_models import AnnotationLayout, MaskDecision
from .annotation_painter import paint_annotation_layout
from .annotation_preview import footprint
from .translation_results import TranslationResult
from .font_resources import annotation_font
from .image_metadata import read_raster, write_raster, verify_raster_metadata
from .export_item_policy import is_inert_preserved_source
from .export_verification import (
    displayed_pdf_page_size, verify_pdf_output, verify_raster_output,
)


# 导出请求冻结用户选择的页面、布局和审核结果，不随后台执行中的 UI 变化。
# 源 SHA 与用户打开和识别的原文件字节绑定，防止同路径文件被替换。
# 布局字典键使用源页索引，输出顺序由 page_indices 单独决定。
# reviewed_results 为 human_reviewed 文本提供显式逐项核对依据。
# 输出路径必须与源路径不同，且不允许覆盖既有目标。
# 仅转换 Path 和浅复制映射，值依靠不可变应用契约保持稳定。
# 格式统一大写便于分支判断，不依赖文件扩展名猜测实际编码。
# 请求构造不代表通过正式资格，服务边界仍将完整复核。
@dataclass(frozen=True)
class ExportRequest:
    """Freeze the selected layouts and reviewed results for one export attempt.

    The shallow read-only mappings prevent UI replacement of individual pages
    during background work; their values are immutable application contracts.
    Source SHA binds the request to the bytes actually opened and reviewed.
    """
    source_path: Path
    output_path: Path
    document_id: str
    layouts: Mapping[int, AnnotationLayout]
    page_indices: tuple[int, ...]
    output_format: str
    jpeg_quality: int = 95
    reviewed_results: Mapping[str, TranslationResult] | None = None
    expected_source_sha256: str | None = None

    def __post_init__(self) -> None:
        # frozen 模型只在初始化时用 object.__setattr__ 归一化字段。
        # 外部原字典后续新增/替换条目不会改变本次任务快照。
        # 审核结果映射缺省为空，不能自动从机器预览推导人工批准。
        # 空页面选择直接拒绝，避免生成一个看似成功的零页输出。
        # JPEG 质量是编码参数而非无损开关，范围限制不提升颜色保真等级。
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "layouts", MappingProxyType(dict(self.layouts)))
        object.__setattr__(self, "reviewed_results", MappingProxyType(dict(self.reviewed_results or {})))
        object.__setattr__(self, "output_format", self.output_format.upper())
        if not self.page_indices:
            raise ValueError("export page selection cannot be empty")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")


# verified 只指实现的结构、几何、字体、像素与元数据检查通过。
# 不认证工程翻译正确、任意特殊 PDF 保真或跨设备色彩一致。
# warnings 同时保留布局退避和格式转换损失，不能只回报成功布尔值。
# output_path 只有在独立回读和原子发布之后才作为正式产物返回。
# page_count 是选中页面数，不是独立图纸或正式验收样本数。
@dataclass(frozen=True)
class ExportResult:
    """Report publication/readback and any explicit raster conversion losses.

    ``verified`` means the implemented geometry, metadata and ink checks passed;
    it does not certify engineering translation or arbitrary color workflows.
    """
    output_path: Path
    page_count: int
    warnings: tuple[str, ...]
    verified: bool


# 服务只生成输出副本，原 PDF/图片不被就地修改。
# 顺序为校验资格、临时编码、独立回读、源身份复核、原子发布。
# 任一步失败都不能把半成品当作已验证结果暴露给用户。
# 正式门禁独立于自动预览能力，preview_only 和机器译文仍拒绝。
# 英文无修改全量导出可复制原编码字节，避免无必要重写。
# 图片/PDF 各走其真实能力路径，不把 RGB OCR 工作图当保真基底。
class AnnotationExportService:
    """Write, independently reopen, then atomically publish a separate copy.

    Review eligibility is validated before rendering. A failed check leaves no
    advertised destination, and an existing destination is never overwritten.
    """
    RASTER_FORMATS = {"PNG", "JPG", "JPEG", "BMP"}

    def export(self, request: ExportRequest) -> ExportResult:
        """Bind read/write to source identity and publish only verified bytes.

        The source is hashed before and after work. This detects changes during
        decoding, including an external editor replacing the original drawing.
        A same-directory temporary file permits no-overwrite atomic publication.
        """
        # 解析路径后再比较源与目标，避免相对路径别名直接覆盖原件。
        # 先查源存在、源 SHA 和目标不存在，再执行费时编码。
        # expected SHA 存在时必须与当前文件匹配，不能拿旧 OCR 配新图纸。
        # 目录创建只为用户指定输出位置，不在源代码目录写运行时日志。
        # 文件系统竞态仍可能发生，最后发布阶段还必须再次保证不覆盖。
        source = request.source_path.resolve()
        output = request.output_path.resolve()
        if source == output:
            raise ValueError("output cannot overwrite the original source drawing")
        if not source.is_file():
            raise ValueError("source drawing does not exist")
        from .project_store import source_sha256
        before_sha=source_sha256(source)
        if request.expected_source_sha256 is not None and before_sha != request.expected_source_sha256:
            raise ValueError('source drawing changed since opening; reopen and recognize before export')
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        self._validate_layouts(request)
        # 临时文件与目标同目录，便于最终原子硬链接/重命名发布。
        # mkstemp 先取得唯一名称，再关闭并移除占位，交给目标编码器正常创建。
        # 后缀沿用目标，编码格式仍以 request.output_format 为准。
        # finally 无论成功失败都清理临时文件，避免未验证产物滞留。
        # 这里的临时输出不属于最终打包，也不生成 EXE 或模型资源。
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{output.stem}-", suffix=output.suffix, dir=output.parent,
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        temp_path.unlink(missing_ok=True)
        encoding_warnings: tuple[str, ...] = ()
        try:
            if request.output_format == "PDF":
                if source.suffix.lower() == '.pdf':
                    from pypdf import PdfReader
                    # 只有布局完全无项且选择原文档全部页面，才直接复制 PDF 原字节。
                    # 部分页面选择需要重组页对象，不能把整份 PDF 冒充选中页输出。
                    # 原字节复制不生成覆盖层，回读验证仍执行页数与原内容检查。
                    # 有 items 时即使只是历史关联，也进入正常资格和绘制流程。
                    # visible 映射用于需要标注验证的页，不把无图层页强行要求中文红像素。
                    unmodified = (not any(layout.items for layout in request.layouts.values())
                                  and request.page_indices == tuple(range(len(PdfReader(source).pages))))
                    if unmodified:
                        shutil.copyfile(source,temp_path)
                        encoding_warnings = ('original_encoded_bytes_preserved',)
                    else:
                        self._export_pdf(request, temp_path)
                    visible = {i:layout for i,layout in request.layouts.items() if layout.items}
                    verify_pdf_output(temp_path, source, request.page_indices, visible)
                else:
                    encoding_warnings = self._export_raster_pdf(request, temp_path)
            # 图片编码后独立验证几何与逐项红色墨迹。
            # 栅格元数据回读在 _export_raster 内执行，与像素检查各负其责。
            # 不支持的格式明确报错，不根据扩展名退到另一个未经选择的编码。
            # 编码/回读完成后再次核对原 SHA，期间源变化则不发布。
            # 这保证导出始终对应任务开始时实际读取的那份源内容。
            elif request.output_format in self.RASTER_FORMATS:
                encoding_warnings = self._export_raster(request, temp_path)
                verify_raster_output(temp_path, request.layouts[0])
            else:
                raise ValueError(f"unsupported export format: {request.output_format}")
            if source_sha256(source) != before_sha:
                raise ValueError('source drawing changed during export; no output was published')
            # 临时文件与目标位于同一目录；硬链接创建是原子的，并在目标已存在时
            # 必须失败，因此不会在“检查后、落盘前”的竞态窗口覆盖其他文件。
            # 发布时创建目标硬链接本身具有“不覆盖已有目标”的原子语义。
            # 前面 exists 检查与这里之间若有其它程序创建同名文件，也会失败。
            # 不能用可替换目标的 os.replace，因为它会破坏不覆盖承诺。
            # Windows 不支持硬链接时使用自身不替换已有目标的 rename。
            # 非 Windows 的 rename 语义不同，不能无条件沿用该回退。
            # 发布后清理的是临时名称，不影响已建立的目标硬链接。
            try:
                os.link(temp_path, output)
            except FileExistsError:
                raise
            except OSError:
                if os.name != "nt":
                    raise
                # FAT/exFAT 等 Windows 文件系统可能不支持硬链接；Windows 的
                # rename 不会替换已有目标，仍保持竞态下的“绝不覆盖”保证。
                os.rename(temp_path, output)
        finally:
            temp_path.unlink(missing_ok=True)
        # 只有 try/finally 正常完成才会构造 verified=True 的结果。
        # 保留逐布局警告与真实编码损失，顺序可对应源页面与项目。
        # JPEG 有损、alpha 展平、ICC 转换等不能因回读成功而从警告中删除。
        # 成功表示本合同成立，不替代专业正确率、性能或目标机验收。
        warnings = tuple(warning for layout in request.layouts.values()
                         for warning in layout.warnings) + encoding_warnings
        return ExportResult(output, len(request.page_indices), warnings, True)

    @staticmethod
    def _validate_layouts(request: ExportRequest) -> None:
        """Recheck provenance and final geometry at the export boundary.

        Preview eligibility never grants export authority. A persisted project
        is untrusted input: finite values, glyph bounds, frame containment and
        collisions still require validation even after earlier UI checks.
        """
        # 持久项目属于不可信输入，正式输出边界重新检查每个布局。
        # preview_only 不能通过更改界面显示状态来解除。
        # 关闭遮挡但残留多边形属于不一致，直接拒绝而不是悄悄继续。
        # 文档身份与字典页号必须一致，避免跨页或跨文件混用布局。
        # 待分析蒙版不能进入正式输出，缺证据不是可忽略警告。
        from PySide6.QtGui import QFontMetricsF
        for index, layout in request.layouts.items():
            if layout.preview_only:
                raise ValueError('preview-only annotation layout cannot be exported')
            if not layout.mask_requested and any(item.mask_polygons for item in layout.items):
                raise ValueError('mask disabled but stale mask polygons remain in export layout')
            if layout.document_id != request.document_id or layout.page_index != index:
                raise ValueError("annotation layout does not match export document/page")
            if layout.mask_requested and any(
                item.mask_decision == MaskDecision.PENDING_ANALYSIS for item in layout.items
            ):
                raise ValueError("mask safety must be analyzed before export")
            # 正式翻译仅允许 manual_test/human_reviewed，机器自动预览没有导出资格。
            # 唯一例外是严格无墨迹的已确认保留源关联，不是未审中文。
            # 例外由共享谓词现场重算，伪造 placement=preserved 不足以绕过门禁。
            # 页面宽高还需有限且为正，NaN 会破坏后续范围比较。
            # 同一布局不允许重复 group，避免一条审核结果被多次绘制到不同位置。
            if any(item.provenance not in {"manual_test", "human_reviewed"}
                   and not is_inert_preserved_source(item)
                   for item in layout.items):
                raise ValueError("unreviewed machine translation cannot be exported")
            if not all(math.isfinite(v) and v > 0 for v in (layout.page_width,layout.page_height)):
                raise ValueError('invalid annotation page size')
            occupied=[]
            groups=set()
            for item in layout.items:
                if item.group_id in groups:
                    raise ValueError('duplicate annotation group')
                groups.add(item.group_id)
                # A historical numeric/view/identifier source link is inert,
                # not a reviewed translation. Recompute the strict shared
                # predicate before review and ink checks; placement alone is
                # never authority to bypass either gate.
                if is_inert_preserved_source(item):
                    continue
                # 来源标签本身不代表用户真的审核了当前完整文字。
                # 显式 reviewed_result 必须匹配 schema、来源、文档、页、组、区域和原译文。
                # 人工修改译文字面后不能继续复用旧审核记录。
                # 审核关联缺失或任何字段不一致就拒绝，不降级为可导出机器文本。
                # manual_test 是明确测试来源的既有合同，不在此伪造为 human_reviewed。
                if item.provenance == 'human_reviewed':
                    reviewed=request.reviewed_results.get(item.group_id)
                    # Labels alone do not authorize export. Bind the final text
                    # to the explicit reviewed result passed from the project.
                    if (reviewed is None or reviewed.schema_version != 1 or
                        reviewed.provenance != 'human_reviewed' or
                        (reviewed.document_id,reviewed.page_index,reviewed.group_id,
                         reviewed.region_ids,reviewed.source_text,reviewed.translated_text) !=
                        (layout.document_id,index,item.group_id,item.region_ids,
                         item.source_text,item.translated_text)):
                        raise ValueError(f'reviewed result mismatch: {item.group_id}')
                # 正式输出必须已有完整可绘制行，待布局、详情和失败项不能蒙混过关。
                # 有限字号、角度与框坐标先检查，再做几何和字体度量。
                # 最低六源单位保证可读性，不让人工项目写入超小字绕开拥挤问题。
                # 拼接 lines 与完整译文去空白后比对，防止丢尾句或少工程符号。
                # 允许排版空白差异，不允许替换数字、标点或字面内容。
                if item.placement != 'inline' or not item.lines:
                    raise ValueError(f'annotation needs layout review: {item.group_id}')
                values=(item.font_size,item.rotation_degrees,item.text_bounds.min_x,
                        item.text_bounds.min_y,item.text_bounds.max_x,item.text_bounds.max_y)
                if not all(math.isfinite(v) for v in values) or item.font_size < 6:
                    raise ValueError('invalid or unreadable annotation font/geometry')
                if re.sub(r'\s+','',''.join(item.lines)) != re.sub(r'\s+','',item.translated_text):
                    raise ValueError('annotation text lines do not match reviewed translation')
                # 紧字形布局按实际 ink 与基线原点验证，旧布局按 advance/lineSpacing 验证。
                # 这两套合同对应 painter 的两条绘制分支，不能混用。
                # 字体度量来自统一资源，避免导出时换系统字体造成越框。
                # 度量合法后仍需检查旋转后的页内、图框内与中文间碰撞。
                # 框内字体 fit 只说明字装得下，不说明它不会盖住别的标注。
                if item.text_origin is not None:
                    from .glyph_text_geometry import validate_text_origin
                    validate_text_origin(item)
                else:
                    metrics=QFontMetricsF(annotation_font(item.font_size))
                    if (max(metrics.horizontalAdvance(line) for line in item.lines) >
                        item.text_bounds.max_x-item.text_bounds.min_x+.01 or
                        metrics.lineSpacing()*len(item.lines) >
                        item.text_bounds.max_y-item.text_bounds.min_y+.01):
                        raise ValueError('annotation text exceeds its final layout bounds')
                # 旋转占用是最终输出范围，不只检查保存的未旋转框。
                # 可靠图框存在时比纸张更严格，未知图框不自动扩展输出画布。
                # occupied 只收已验证可绘制项，严格无墨迹保留源关联不消耗位置。
                # 任意两个正式中文占用正交叠都需人工复核。
                # 这里不重新跑 OCR 或重排位置，避免正式导出与已看预览悄悄不同。
                box=footprint(item.text_bounds,item.rotation_degrees)
                if (box.min_x < 0 or box.min_y < 0 or box.max_x > layout.page_width
                        or box.max_y > layout.page_height):
                    raise ValueError('rotated annotation is outside effective page')
                frame=layout.effective_bounds
                if frame is not None and (box.min_x<frame.min_x or box.min_y<frame.min_y
                        or box.max_x>frame.max_x or box.max_y>frame.max_y):
                    raise ValueError('annotation is outside recorded drawing frame')
                if any(min(box.max_x,b.max_x)>max(box.min_x,b.min_x) and
                       min(box.max_y,b.max_y)>max(box.min_y,b.min_y) for b in occupied):
                    raise ValueError('annotation collision requires layout review')
                occupied.append(box)

    @staticmethod
    def _export_raster(request: ExportRequest, temp_path: Path) -> tuple[str, ...]:
        # 图片只有单页，选择和布局键都必须恰为零号页。
        # 重新从源文件读取视觉定向图与元数据，不拿窗口缩略图导出。
        # 源像素宽高需与保存布局一致，不能靠重采样迁就错配。
        # 同格式无项且无 EXIF 转向可直接复制原编码，保留灰度/调色板及 JPEG。
        # 需要注释或转向时必须进入真实编码合同并返回损失信息。
        if request.page_indices != (0,) or set(request.layouts) != {0}:
            raise ValueError("image export supports exactly page 1")
        image, metadata = read_raster(request.source_path)
        layout = request.layouts[0]
        if image.width() != round(layout.page_width) or image.height() != round(layout.page_height):
            raise ValueError("source image dimensions do not match annotation layout")
        from PIL import Image
        with Image.open(request.source_path) as native:
            fmt='JPEG' if request.output_format in {'JPG','JPEG'} else request.output_format
            # An untouched same-format copy preserves JPEG coding, palette,
            # grayscale and ancillary metadata rather than round-tripping RGB.
            if (not layout.items and native.format == fmt and
                    native.getexif().get(274,1)==1):
                shutil.copyfile(request.source_path,temp_path)
                return ('original_encoded_bytes_preserved',)
        # CMYK 工作图已转为 RGB，直接写栅格会丢原通道合同。
        # 因此正式栅格标注保守拒绝，建议使用原生通道 PDF 输出。
        # 复制 QImage 保持直通 alpha 及透明像素背后的 RGB 样本。
        # 预乘存储可能把 alpha=0 背后颜色清零，不能用于声称原样本保留。
        # QPainter 只在输出副本上画共享布局，不改输入源图。
        if metadata.source_mode == 'CMYK':
            raise ValueError('CMYK raster annotation cannot preserve native channels; choose PDF output')
        # Keep straight alpha and untouched RGB samples, including the RGB bytes
        # behind alpha=0. Premultiplied storage would discard that hidden color.
        output = image.copy()
        painter = QPainter(output)
        if not painter.isActive():
            raise ValueError('failed to initialize raster annotation painter')
        try:
            paint_annotation_layout(painter, layout)
        finally:
            painter.end()
        # 编码器按实际格式处理 ICC/EXIF/DPI/alpha，并返回明确能力损失。
        # 随后用 Pillow 复核元数据，Qt 复核尺寸与红字位置。
        # 两个解码器检查不同问题，任一失败都阻止原子发布。
        # JPEG 重编码始终有损，回读成功不会改变这一定性。
        warnings = write_raster(output, metadata, temp_path, request.output_format,
                                request.jpeg_quality)
        # Pixel geometry is checked independently by verify_raster_output. This
        # second decoder checks metadata that Qt's image reader may normalize.
        verify_raster_metadata(temp_path, metadata, request.output_format)
        return warnings

    def _export_raster_pdf(self, request: ExportRequest, temp_path: Path) -> tuple[str,...]:
        """Reuse the reviewed vector overlay on a native-channel image PDF.

        The temporary base is a disposable copy; source JPEG/PNG/BMP is never
        passed through Qt's RGB preview encoder. Existing PDF verification then
        checks that the base image objects remain and captions are embedded.
        """
        # 图片转 PDF 先包装原通道基底，再使用与原 PDF 一致的矢量覆盖流程。
        # 临时基底是副本，直接读取源图片而不经 Qt RGB 预览重编码。
        # CMYK/ICC/透明软蒙版和普通 JPEG DCT 在 native_raster_pdf 中处理。
        # 布局按一像素一 point 与基底尺寸绑定，不套用原 DPI 缩放。
        # 图片输出仍只支持单页，源文件不被覆盖或写入 EXIF。
        from dataclasses import replace
        from .native_raster_pdf import write_native_raster_pdf
        if request.page_indices != (0,) or set(request.layouts) != {0}:
            raise ValueError('image PDF export supports exactly page 1')
        descriptor,name=tempfile.mkstemp(prefix='.native-image-',suffix='.pdf',dir=temp_path.parent)
        os.close(descriptor);base=Path(name)
        try:
            size=write_native_raster_pdf(request.source_path,base)
            layout=request.layouts[0]
            if tuple(map(round,(layout.page_width,layout.page_height))) != size:
                raise ValueError('native image dimensions do not match annotation layout')
            # 带项时叠加矢量层并回读，空项则复制原生基底再回读结构。
            # verify_pdf_output 对比的是本次原通道基底，不是 OCR RGB 工作图。
            # 原生图像对象应保留，新增中文不应新增整页栅格 XObject。
            # 返回 warning 明确 PDF 坐标合同，避免误解为原物理 DPI 完全保持。
            # finally 删除临时基底，失败不留半成品供误用。
            if layout.items:
                self._export_pdf(replace(request,source_path=base),temp_path)
                verify_pdf_output(temp_path,base,(0,),dict(request.layouts))
            else:
                shutil.copyfile(base,temp_path)
                verify_pdf_output(temp_path,base,(0,),{})
            return ('native_image_channels_preserved_in_pdf','pdf_layout_one_pixel_per_point')
        finally:
            base.unlink(missing_ok=True)

    def _export_pdf(self, request: ExportRequest, temp_path: Path) -> None:
        """Merge a vector copy in displayed CropBox coordinates at 72 points/inch.

        Existing PDF content remains under the overlay, including searchable
        English. Rotation normalization is refused when original annotations
        could move; temporary overlay pages are always removed after the merge.
        """
        from pypdf import PdfReader, PdfWriter, Transformation

        # 原 PDF 路径以页对象克隆/合并为主，不重新渲染成整页图像。
        # 页选择先验证范围，避免错误索引导致页顺序或数量不可信。
        # 全量 append 同时导入书签，部分选择则按用户顺序 add_page。
        # 原文元数据尽量复制，原字、矢量线稿和原图像对象留在底层。
        # 容错读取 strict=False 不豁免后续几何、原内容和字体回读。
        reader = PdfReader(str(request.source_path), strict=False)
        if any(index < 0 or index >= len(reader.pages) for index in request.page_indices):
            raise ValueError("PDF export page selection is out of range")
        full_document = request.page_indices == tuple(range(len(reader.pages)))
        writer = PdfWriter()
        if full_document:
            # append 导入全部页面及书签，但不会把未被页面引用的对象一并展开；
            # 对含对象流的工程 PDF，可避免仅因重写而产生数倍体积膨胀。
            writer.append(reader, import_outline=True)
        else:
            for source_index in request.page_indices:
                writer.add_page(reader.pages[source_index])
        if reader.metadata:
            writer.add_metadata(dict(reader.metadata))
        overlay_paths: list[Path] = []
        try:
            for output_index, source_index in enumerate(request.page_indices):
                layout = request.layouts.get(source_index)
                if layout is None:
                    continue
                page = writer.pages[output_index]
                # 每页覆盖层基于显示方向的 CropBox 尺寸，不按 MediaBox 纸张尺寸猜测。
                # Rotate 非零时先把旋转转入内容，随后才可按统一坐标叠加。
                # 含原生 Annots 的旋转页保守拒绝，避免注释外观/几何未同步变换。
                # 原注释存在不能简单删除以“解决”兼容问题。
                # 规范化后再验宽高，检测 pypdf 对特殊页变换的未知影响。
                before = displayed_pdf_page_size(page)
                if abs(before[0] - layout.page_width) > 0.1 or abs(before[1] - layout.page_height) > 0.1:
                    raise ValueError("source PDF display size does not match annotation layout")
                if page.rotation:
                    annotations = page.get("/Annots")
                    if annotations is not None and len(annotations.get_object()) > 0:
                        raise ValueError(
                            "旋转 PDF 页包含原生注释，无法在不移动注释的情况下安全规范化；"
                            "已拒绝导出且未写入目标文件",
                        )
                    page.transfer_rotation_to_content()
                crop = page.cropbox
                normalized = displayed_pdf_page_size(page)
                if abs(normalized[0] - layout.page_width) > 0.1 or abs(normalized[1] - layout.page_height) > 0.1:
                    raise ValueError("rotated PDF CropBox normalization changed display size")
                descriptor, overlay_name = tempfile.mkstemp(
                    prefix=".annotation-overlay-", suffix=".pdf", dir=temp_path.parent,
                )
                os.close(descriptor)
                overlay_path = Path(overlay_name)
                overlay_paths.append(overlay_path)
                self._write_overlay_pdf(overlay_path, layout)
                overlay_page = PdfReader(str(overlay_path)).pages[0]
                # 覆盖层本身为局部显示页，按当前 CropBox 左下原点平移。
                # pypdf 与 Qt 的上下方向已在覆盖 PDF 中处理，这里只补裁剪偏移。
                # over=True 叠在原内容上，底层英文文字仍存在且可搜索。
                # expand=False 保持原页画布，不为越界标注扩大纸张。
                # 遮挡是视觉填色，不能描述为从原 PDF 删除英文对象。
                page.merge_transformed_page(
                    overlay_page,
                    Transformation().translate(float(crop.left), float(crop.bottom)),
                    over=True,
                    expand=False,
                )
            # pypdf 写出时会展开部分压缩对象流；用 Flate 无损压缩每页内容流，
            # 再去重/移除孤儿对象。这里不栅格化页面，也不改变绘图语义。
            # 逐页 Flate 压缩内容流降低重写体积，不把矢量图转栅格。
            # 对象去重和孤儿清理改善存储，不应改变可见绘图语义。
            # 这些结构处理仍需后面的独立回读验证，不能只信 writer 成功。
            # 覆盖层临时文件始终在 finally 清理，单页失败不泄漏后续半成品。
            for page in writer.pages:
                if page.get_contents() is not None:
                    page.compress_content_streams()
            writer.compress_identical_objects(
                remove_duplicates=True, remove_unreferenced=True,
            )
            with temp_path.open("wb") as stream:
                writer.write(stream)
        finally:
            for overlay_path in overlay_paths:
                overlay_path.unlink(missing_ok=True)

    @staticmethod
    def _write_overlay_pdf(path: Path, layout: AnnotationLayout) -> None:
        """Use the shared painter and exact point page size, without rasterizing.

        The embedded application font and 72-DPI writer make one painter unit
        equal one PDF point; monitor/device scaling must not enter this path.
        """
        # 72 DPI 使 painter 单位和 PDF point 一一对应，排除显示器 DPI 干扰。
        # ExactMatch 与零页边距防止 Qt 自动套纸型或加打印边距。
        # FullPageMode 使用完整自定义页面，不因打印机可印范围裁掉中文。
        # 布局宽高已是显示方向尺寸，无需再次根据横竖自动旋转。
        # 共享 painter 复用相同 lines、字号、旋转和基线原点。
        # 应用字体由 Qt 嵌入，后续解析器还要确认确有字体流。
        # 画布设置或 painter 初始化失败明确报错，不发布空白覆盖层。
        writer = QPdfWriter(str(path))
        writer.setResolution(72)
        size = QPageSize(
            QSizeF(layout.page_width, layout.page_height),
            QPageSize.Unit.Point,
            "annotation-overlay",
            QPageSize.SizeMatchPolicy.ExactMatch,
        )
        page_layout = QPageLayout(
            size, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0),
            QPageLayout.Unit.Point,
        )
        page_layout.setMode(QPageLayout.Mode.FullPageMode)
        if not writer.setPageLayout(page_layout):
            raise ValueError("failed to set exact PDF overlay page layout")
        painter = QPainter(writer)
        if not painter.isActive():
            raise ValueError("failed to initialize PDF overlay painter")
        paint_annotation_layout(painter, layout)
        painter.end()
