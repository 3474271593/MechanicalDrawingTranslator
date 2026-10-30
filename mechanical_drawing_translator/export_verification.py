# mechanical_drawing_translator/export_verification.py
"""导出文件的回读、页面几何、标注像素和字体嵌入验证。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtGui import QImageReader

from .annotation_models import AnnotationLayout
from .annotation_preview import footprint
from .export_item_policy import is_inert_preserved_source


# 显示尺寸来自 CropBox，MediaBox 可能包含用户不见的裁剪外内容。
# 直角 Rotate 交换宽高，180 度只改方向不换尺寸。
# 坐标原点偏移不在这里消除，merge 时另用 CropBox 左下平移。
# point 是 PDF 逻辑单位，不受屏幕缩放或系统 DPI 影响。
# 这个函数只回报尺寸，不声称特殊 UserUnit 或其它未知变换已全面支持。
def displayed_pdf_page_size(page) -> tuple[float, float]:
    """Use the visible CropBox and displayed axes, rather than the media sheet.

    PDF points are independent of monitor DPI. A quarter-turn exchanges axes;
    offsets remain the responsibility of the PDF overlay transform.
    """
    crop = page.cropbox
    width, height = float(crop.width), float(crop.height)
    return (height, width) if page.rotation % 180 else (width, height)


# 简单字体描述符可能直接挂在字体对象上。
# CID 复合字体通常把描述符挂在 DescendantFonts 的首个后代。
# 间接对象需要 get_object 解引用，不能只查引用外壳字段。
# 找不到描述符返回 None，让上层无法验证嵌入而拒绝。
# 此检查是结构检查，不验证每一个汉字的语义或视觉笔画正确。
def _font_descriptor(font) -> dict | None:
    """Resolve either a simple font or the first CID descendant's descriptor."""
    value = font.get_object()
    descriptor = value.get("/FontDescriptor")
    if descriptor is not None:
        return descriptor.get_object()
    descendants = value.get("/DescendantFonts") or []
    if descendants:
        return descendants[0].get_object().get("/FontDescriptor", {}).get_object()
    return None


# 寻找 NotoSans 标注字体且存在实际字体文件流键。
# 只看到 /BaseFont 名称不足以证明目标机离线可显示。
# PDF 字体名可能包含子集前缀，使用族名字串匹配而非完全相等。
# 无字体资源页面跳过，不能因它先出现就误判整份 PDF 失败。
# 函数目前确认至少一个嵌入字体，不声称穷举所有页的每个字形。
# 更严格逐项视觉完整性需额外渲染/复核证据。
def pdf_annotation_fonts_embedded(reader) -> bool:
    """至少一个 Noto 标注字体具有实际嵌入字体流。"""
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get_object().get("/Font")
        if fonts is None:
            continue
        for reference in fonts.get_object().values():
            font = reference.get_object()
            base = str(font.get("/BaseFont", ""))
            # 复合字体主对象缺 BaseFont 时，从后代提取真实字体名。
            # 名字中的空格不影响识别，仍必须检查文件流而非仅字体族。
            # FontFile/FontFile2/FontFile3 覆盖常见 Type1/TrueType/OpenType 嵌入方式。
            # 这里不下载系统字体，也不允许依赖查看器在线替代字体。
            descendants = font.get("/DescendantFonts") or []
            if descendants and not base:
                base = str(descendants[0].get_object().get("/BaseFont", ""))
            if "NotoSans" not in base.replace(" ", ""):
                continue
            descriptor = _font_descriptor(font)
            if descriptor and any(descriptor.get(key) is not None
                                  for key in ("/FontFile", "/FontFile2", "/FontFile3")):
                return True
    return False


# 递归统计页面及嵌套 Form 内的图像资源。
# 用于发现矢量中文叠加误变成新的整页栅格图像。
# seen 避免共享资源重复遍历与潜在资源环递归。
# 图像计数是必要结构警戒，数量相等不证明每个原图像字节相等。
# 原图可见完整性仍需与其它验证及专门视觉证据一起判断。
def page_image_xobject_count(page) -> int:
    """统计页面及 Form XObject 内的图像；矢量覆盖层不应增加整页图像。"""
    seen: set[int] = set()

    def count_resources(resources) -> int:
        if resources is None:
            return 0
        value = resources.get_object()
        xobjects = value.get("/XObject")
        if xobjects is None:
            return 0
        total = 0
        # 每次按实际引用对象解出 Subtype，再区分 Image 与 Form。
        # Form 是复用绘图容器，可能继续引用更深层图像资源。
        # 其它 XObject 不计入图像数，避免把矢量表单误算为栅格。
        # 统计不渲染资源，也不修改原 PDF 对象树。
        for reference in xobjects.get_object().values():
            identity = id(reference)
            if identity in seen:
                continue
            seen.add(identity)
            obj = reference.get_object()
            subtype = obj.get("/Subtype")
            if subtype == "/Image":
                total += 1
            elif subtype == "/Form":
                total += count_resources(obj.get("/Resources"))
        return total

    return count_resources(page.get("/Resources"))


# 对每个应绘制项单独寻找局部红墨迹，不用整页有红就代表全部成功。
# 紧邻 footprint 留少量像素容差，容忍渲染抗锯齿与取整。
# 只验证大致位置与可见性，不进行中文 OCR 或完整文本确认。
# 原图已有红字可能满足阈值，因此不能把通过写成字形身份已证实。
# 无墨迹保留源记录通过严格共享谓词识别并跳过。
# 它们不应被要求重新染红，否则会改变原工程字面外观。
def _has_red_pixels(image, layout: AnnotationLayout) -> bool:
    """Require local ink evidence for every item, without claiming OCR proof.

    This smoke check catches absent/displaced annotations. Existing source red
    ink can satisfy it, so it cannot establish complete glyph or text identity.
    Dedicated composite comparisons remain necessary for visual acceptance.
    """
    rgb = image.convert("RGB")
    # 渲染图分辨率可不同于 source point，X/Y 按真实尺寸分别换算。
    # 旋转使用共享 footprint，避免只查未旋转框错过 90 度标注。
    # 查询区域钳制在像素范围内，不因容差访问页外。
    # 红判定是相对通道优势加最低亮度，尽量排除原黑线。
    # 三个像素的低门槛属于缺失检查，不能当专业视觉质量评分。
    sx, sy = rgb.width / layout.page_width, rgb.height / layout.page_height
    for item in layout.items:
        if is_inert_preserved_source(item):
            continue
        box = footprint(item.text_bounds, item.rotation_degrees)
        left = max(0, int(box.min_x * sx) - 4)
        top = max(0, int(box.min_y * sy) - 4)
        right = min(rgb.width, int(box.max_x * sx) + 5)
        bottom = min(rgb.height, int(box.max_y * sy) + 5)
        count = 0
        for y in range(top, bottom):
            for x in range(left, right):
                red, green, blue = rgb.getpixel((x, y))
                if red > 130 and red > green * 1.35 and red > blue * 1.35:
                    count += 1
        if count < 3:
            return False
    return True


# Qt 独立重新打开实际落盘副本，发现空文件、编码失败或错误尺寸。
# 栅格布局单位就是定向后像素，不再施加屏幕缩放或 EXIF 旋转。
# 逐项红色检查防止只有一条中文画出来便误报全页成功。
# ICC/alpha/EXIF/DPI 不在本函数重复猜测，由 Pillow 元数据回读负责。
# 两套检查任意失败都应阻止导出服务最终发布。
# 原图 RGB 原样保留与颜色空间转换也不能只凭红色像素证明。
def verify_raster_output(path: str | Path, layout: AnnotationLayout) -> None:
    """Reopen with Qt, then check each source-coordinate annotation separately.

    A single red annotation cannot certify all other items. Color/EXIF/density
    preservation is checked separately by the metadata decoder at export time.
    """
    reader = QImageReader(str(path))
    image = reader.read()
    if image.isNull():
        raise ValueError(f"exported image cannot be reopened: {reader.errorString()}")
    if image.width() != round(layout.page_width) or image.height() != round(layout.page_height):
        raise ValueError("exported image dimensions changed")
    # 源关联例外重新运行策略，保存的 preserved 标签本身不够。
    # 真正有译文、行或蒙版的项不能借例外跳过像素检查。
    # 搜索范围以最终旋转框为准，并加小的栅格容差。
    # 缺少红色即报位置不可验证，不尝试移动或重新生成布局掩盖问题。
    for item in layout.items:
        if is_inert_preserved_source(item):
            continue
        red = 0
        box = footprint(item.text_bounds, item.rotation_degrees)
        for y in range(max(0, int(box.min_y) - 2), min(image.height(), int(box.max_y) + 3)):
            for x in range(max(0, int(box.min_x) - 2), min(image.width(), int(box.max_x) + 3)):
                color = image.pixelColor(x, y)
                if (color.red() > 130 and color.red() > color.green() * 1.25
                        and color.red() > color.blue() * 1.25):
                    red += 1
        if red < 3:
            raise ValueError("exported image annotation coordinates could not be verified")


# PDF 输出同时用 pypdf 解析结构、PDFium 实际渲染，减少单库自证。
# 页数、显示尺寸、标注可见性、原文字与原内容流都分别检查。
# 结构通过不是任意 PDF 绘图语义完全不变的数学证明。
# 源文件继续只读；检查目标是临时副本，失败不触及原件。
# source_indices 将输出页顺序映回原页，不能拿输出索引直接查原文。
# layouts 可只含有图层页，其它页仍要验证原内容保留。
# 当前函数不做工程翻译准确率和长时稳定性验收。
def verify_pdf_output(
    path: str | Path,
    source_path: str | Path,
    source_indices: Iterable[int],
    layouts: dict[int, AnnotationLayout],
) -> None:
    """Verify the written copy with independent parsing and PDFium rendering.

    Vector text and image-object counts detect destructive raster replacement.
    These are necessary structural checks, not a proof that arbitrary source
    graphics are visually unchanged underneath an intentional overlay.
    """
    import pypdfium2 as pdfium
    from pypdf import PdfReader

    # 先冻结页序列，后续多次遍历不会耗尽生成器。
    # 重新打开源与输出，确保验证的是实际编码文件而非 writer 内存对象。
    # 有布局时需要字体嵌入；完全无标注复制不要求新增中文字体。
    # PDFium 句柄在 finally 关闭，异常页不应泄漏整份文档资源。
    indices = tuple(source_indices)
    source = PdfReader(str(source_path))
    output = PdfReader(str(path))
    if len(output.pages) != len(indices):
        raise ValueError("exported PDF page count changed")
    if layouts and not pdf_annotation_fonts_embedded(output):
        raise ValueError("annotation font is not embedded in exported PDF")
    render_pdf = pdfium.PdfDocument(str(path))
    try:
        for output_index, source_index in enumerate(indices):
            page = output.pages[output_index]
            layout = layouts.get(source_index)
            if layout is not None:
                actual = displayed_pdf_page_size(page)
                if abs(actual[0] - layout.page_width) > 0.1 or abs(actual[1] - layout.page_height) > 0.1:
                    raise ValueError("exported PDF page dimensions do not match annotation layout")
                # 固定二倍渲染提供较清晰局部红色证据，不依赖当前窗口缩放。
                # 实际图尺寸按 source point 再映射，不以整数 PDF 尺寸直接索引。
                # 只要任一中文项缺红墨迹就拒绝，不以其它项成功抵消。
                # 这不是运行 OCR，也不声称检查了所有字形内容与排版美观。
                rendered = render_pdf[output_index].render(scale=2).to_pil()
                if not _has_red_pixels(rendered, layout):
                    raise ValueError("exported PDF annotation coordinates could not be verified")
            # 底层原文必须继续可提取，视觉遮挡不应删除英文对象。
            # 逐原始非空行要求在输出提取文本中出现，检测破坏性栅格替换。
            # 该字符串包含检查不保证字体顺序、每个对象 ID 或原字节不变。
            # 存在原内容流时输出不能丢掉内容流。
            # 若图像 XObject 数量增加，说明本矢量叠加合同可能被破坏，应拒绝。
            # 结构检查不足以检测所有图形变化，专门视觉验收仍独立保留。
            # 所有异常交给导出服务处理，临时副本不会原子发布。
            original_text = source.pages[source_index].extract_text() or ""
            output_text = page.extract_text() or ""
            original_lines = [line.strip() for line in original_text.splitlines() if line.strip()]
            if any(line not in output_text for line in original_lines):
                raise ValueError("original PDF vector text was not preserved")
            if source.pages[source_index].get_contents() is not None and page.get_contents() is None:
                raise ValueError("original PDF page content stream was not preserved")
            if page_image_xobject_count(page) > page_image_xobject_count(source.pages[source_index]):
                raise ValueError("PDF overlay introduced raster image XObjects")
    finally:
        render_pdf.close()
