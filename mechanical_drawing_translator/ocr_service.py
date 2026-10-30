# mechanical_drawing_translator/ocr_service.py
"""与 Qt 无关的 OCR 文档服务；所有坐标均基于原始页面左上角。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import atan2, degrees
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from PIL import Image

from .models import TextBounds, TextRegion
from .ocr_refinement import RefinementReport, refine_pdf_page
from .engineering_character_verification import verify_engineering_region
from .image_metadata import read_raster_pixels


class OcrError(Exception):
    """不可读取的图纸、页面或 OCR 处理错误。"""


# 引擎输出仍在输入像素系
# 四角点保留检测几何
# 方向可由引擎补充
# 缺少方向时由首边估算
# 不能直接当PDF point使用
@dataclass(frozen=True)
class DetectedText:
    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]
    rotation_degrees: float | None = None


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: Image.Image) -> list[DetectedText]: ...


# 此门只判断OCR保留资格
# 不是最终翻译目标判断
# 数字和工程符号仍可保留
# 基线拒绝中日韩混合字面
# integrated允许ASCII混框
# 后续策略再决定是否送译
# 原始字面不在此归一化
def english_candidate(text: str, *, enable_integrated_quality: bool = False) -> bool:
    """候选模式保留混合框完整字面，避免丢失英文及中文否定条件。"""
    if enable_integrated_quality and any(ch.isascii() and ch.isalnum() for ch in text):
        return True
    return any((ch.isascii() and ch.isalnum()) or ch in "Øø°±⌀∅" for ch in text) and not any(
        "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af" for ch in text
    )


# 每页独立记录坐标单位
# PDF旋转另作元数据保存
# render_dpi仅表示工作图
# 原生文字数不等于覆盖率
# ocr_used标记实际走过引擎
# elapsed是本页服务计时
# regions始终保留原始结果
# refinement另外保存候选
# 不可只保留被采用的新文本
@dataclass(frozen=True)
class OcrPage:
    page_index: int
    width: float
    height: float
    coordinate_unit: str
    pdf_rotation_degrees: int | None
    pdf_user_unit: float | None
    render_dpi: int | None
    native_visible_words: int
    native_rejected_words: int
    ocr_used: bool
    elapsed_seconds: float
    regions: tuple[TextRegion, ...]
    refinement: RefinementReport | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# 文档结果可只包含选定页
# page_count仍是源总页数
# pages长度不代表文档页数
# document_id来自调用请求
# input_path供本地追溯
# 不是可上传的数据封装
@dataclass(frozen=True)
class OcrDocument:
    document_id: str
    input_path: str
    engine: str
    page_count: int
    pages: tuple[OcrPage, ...]

    def to_dict(self) -> dict:
        return asdict(self)


# x/y分别按实际栅格比例还原
# 不假设取整后倍率相同
# PDF源单位仍为point
# 图片传入比例为一
# 每个点限制在页面范围
# 小数保留三位供存储
# 四点不足明确报错
# 边界由全部点包围
# 原多边形与矩形同时保存
# 显式阅读方向优先
# 否则从首边计算角度
# 新ID对应此轮检测区域
# 源文本与置信度原样保留
# 不从中文反推缺失英文
def map_detection(
    detection: DetectedText, document_id: str, page_index: int,
    x_scale: float, y_scale: float, width: float, height: float,
) -> TextRegion:
    """将图像像素四边形映射到原始页面点坐标或像素坐标。"""
    points = tuple(
        (round(max(0.0, min(width, float(x) / x_scale)), 3),
         round(max(0.0, min(height, float(y) / y_scale)), 3))
        for x, y in detection.polygon
    )
    if len(points) < 4:
        raise OcrError("OCR 引擎返回的文字框不足四个点")
    angle = (detection.rotation_degrees if detection.rotation_degrees is not None else
             degrees(atan2(points[1][1] - points[0][1],
                           points[1][0] - points[0][0])))
    xs, ys = zip(*points)
    return TextRegion(
        id=str(uuid4()), document_id=document_id, page_index=page_index,
        source_text=detection.text, translated_text="",
        bounds=TextBounds(min(xs), min(ys), max(xs), max(ys)),
        rotation_degrees=round(angle, 2), confidence=float(detection.confidence),
        polygon=points, source="ocr",
    )


class OcrService:
    """优先读取可见 PDF 原生文字，必要时独立高分辨率渲染后 OCR。"""

    IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}

    # 渲染DPI限制72到600
    # 高DPI增加工作像素成本
    # min_native_words只作触发阈值
    # 不是原生覆盖正确性证明
    # force_ocr仍保留原生区域
    # 局部复查默认关闭
    # 字符核验默认也关闭
    # 配置随服务生命周期固定
    def __init__(
        self, engine: OcrEngine, *, dpi: int = 216,
        min_native_words: int = 15, force_ocr: bool = False,
        enable_refinement: bool = False,
        enable_engineering_verification: bool = False,
        enable_integrated_quality: bool = False,
    ):
        if not 72 <= dpi <= 600:
            raise ValueError("OCR 渲染 DPI 应在 72 到 600 之间")
        self.engine = engine
        self.dpi = dpi
        self.min_native_words = min_native_words
        self.force_ocr = force_ocr
        self.enable_refinement = enable_refinement
        self.enable_engineering_verification = enable_engineering_verification
        self.enable_integrated_quality = enable_integrated_quality

    # 先验证文件实际存在
    # 格式范围与文档加载器一致
    # 外部可注入已有文档ID
    # 图片只能请求第零页
    # PDF逐选定页处理
    # OcrError保持原业务错误
    # 其他异常统一包装文件上下文
    # 服务本身不访问Qt控件
    def extract(
        self, path: str | Path, page_indices: list[int] | None = None,
        document_id: str | None = None,
    ) -> OcrDocument:
        source = Path(path).expanduser()
        if not source.is_file():
            raise OcrError(f"图纸不存在或无法访问：{source}")
        if source.suffix.lower() not in self.IMAGE_TYPES | {".pdf"}:
            raise OcrError("仅支持 PDF、JPG、PNG、BMP")
        doc_id = document_id or str(uuid4())
        try:
            if source.suffix.lower() == ".pdf":
                return self._extract_pdf(source, page_indices, doc_id)
            if page_indices is not None and page_indices != [0]:
                raise OcrError("图片仅有第 1 页")
            return self._extract_image(source, doc_id)
        except OcrError:
            raise
        except Exception as exc:
            raise OcrError(f"无法处理图纸 {source.name}：{exc}") from exc

    # 解码与预览共享方向语义
    # 不引入Qt图像对象
    # RGBA先对白纸合成再识别
    # 透明像素隐藏RGB不算墨迹
    # 直接丢alpha会造假笔画
    # 源坐标是定向后完整像素
    # 整图OCR不使用屏幕缩略图
    # 工程核验仅加旁路证据
    # 图片不套PDF旋转或DPI
    # 计时包含解码与识别成本
    def _extract_image(self, path: Path, doc_id: str) -> OcrDocument:
        started = perf_counter()
        # The CPU decoder shares preview/export ICC and EXIF semantics without
        # importing Qt. Annotation coordinates remain oriented source pixels.
        try:
            image, metadata = read_raster_pixels(path)
        except (OSError, ValueError) as error:
            raise OcrError(f"图片无法安全读取：{error}") from error
        if metadata.has_alpha:
            # Hidden RGB behind transparent pixels is not visible drawing ink.
            # Composite onto the viewer's white paper before OCR; simply dropping
            # alpha would invent dark/color marks and change the engine input.
            paper = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(paper, image).convert('RGB')
        width, height = image.size
        detections = self.engine.recognize(image)
        regions = tuple(map_detection(d, doc_id, 0, 1, 1, width, height)
                        for d in detections if english_candidate(
                            d.text, enable_integrated_quality=self.enable_integrated_quality))
        if self.enable_engineering_verification:
            regions = self._verify_regions(image, regions, 1, 1)
        page = OcrPage(0, width, height, "pixel", None, None, None, 0, 0,
                       True, round(perf_counter() - started, 3), regions)
        return OcrDocument(doc_id, str(path), self.engine.name, 1, (page,))

    # PDFium负责渲染与页对象
    # pdfplumber负责原生文字
    # 二者都独立于Qt预览句柄
    # 打开半失败也关闭已建句柄
    # 先核验全部请求页号
    # selected保留调用方顺序
    # 总页数来自实际PDF
    # finally释放两套文档
    # 异常不遗留文件占用
    def _extract_pdf(
        self, path: Path, indices: list[int] | None, doc_id: str,
    ) -> OcrDocument:
        import pdfplumber
        import pypdfium2 as pdfium

        pdf = None
        try:
            pdf = pdfium.PdfDocument(str(path))
            native_pdf = pdfplumber.open(path)
        except Exception as exc:
            if pdf is not None:
                pdf.close()
            raise OcrError(f"PDF 无法打开：{exc}") from exc
        try:
            count = len(pdf)
            selected = list(range(count)) if indices is None else indices
            if any(i < 0 or i >= count for i in selected):
                raise OcrError(f"页码超出范围，PDF 共 {count} 页")
            pages = tuple(self._pdf_page(pdf[i], native_pdf.pages[i], i, doc_id, path)
                          for i in selected)
            return OcrDocument(doc_id, str(path), self.engine.name, count, pages)
        finally:
            native_pdf.close()
            pdf.close()

    # 逐页决定原生与OCR路线
    # 简单无文字页可略过原生解析
    # 不把整份PDF缓存成无文字
    # 扫描图像存在仍触发OCR
    # 少量文字层不足以跳过OCR
    # integrated还核查原生缺口
    # 图像按dpi/72倍率渲染
    # 实际x/y倍率由成图反算
    # 所有检测映回页面point
    # 原生区域先进入结果集合
    # OCR仅去除相同文本重叠项
    # 不同字面不因框重叠而丢弃
    # 局部复查失败保留整页OCR
    # 失败原因进入RefinementReport
    # 原始regions不会被候选覆盖
    # 采用候选另在selected_regions
    # 工程核验可继续核查新候选
    # PDF旋转与UserUnit留作记录
    # 返回计时包括可选复查成本
    # 不能拿纯推理耗时冒充总耗时
    def _pdf_page(self, page, native_page, index: int, doc_id: str,
                  source_path: Path | None = None) -> OcrPage:
        started = perf_counter()
        width, height = page.get_size()
        no_native_text = self.min_native_words > 0 and self._native_text_absent(page)
        native, rejected = ([], 0) if no_native_text else self._visible_words(
            native_page, doc_id, index, width, height,
            enable_integrated_quality=self.enable_integrated_quality,
        )
        # 即使有少量文字层，图纸中的扫描图像仍需 OCR。
        run_ocr = (self.force_ocr or no_native_text or len(native) < self.min_native_words
                   or bool(native_page.images))
        if self.enable_integrated_quality and not run_ocr:
            # A word-count threshold is not coverage proof. Supplement only when
            # the native parser exposes relevant, on-page glyphs outside all
            # accepted word bounds. Paths alone cannot prove outlined text.
            run_ocr = self._native_coverage_incomplete(native_page, native, width, height)
        ocr_regions: list[TextRegion] = []
        image = None
        sx = sy = 1.0
        if run_ocr:
            image = page.render(scale=self.dpi / 72).to_pil().convert("RGB")
            sx, sy = image.width / width, image.height / height
            ocr_regions = [map_detection(d, doc_id, index, sx, sy, width, height)
                           for d in self.engine.recognize(image) if english_candidate(
                               d.text, enable_integrated_quality=self.enable_integrated_quality)]
            if self.enable_engineering_verification:
                ocr_regions = list(self._verify_regions(image, tuple(ocr_regions), sx, sy))
        regions = list(native)
        for candidate in ocr_regions:
            if not any(self._duplicate(candidate, existing) for existing in regions):
                regions.append(candidate)
        refinement = None
        if self.enable_refinement and source_path is not None:
            try:
                refinement=refine_pdf_page(source_path,index,regions,self.engine)
            except Exception as exc:
                # Optional quality work must not erase successful full-page OCR.
                refinement=RefinementReport(tuple(regions),(),{r.id:r.id for r in regions},0,
                                            (f"local_refinement_failed:{type(exc).__name__}",))
        if (self.enable_engineering_verification and image is not None
                and refinement is not None):
            refinement = replace(
                refinement,
                selected_regions=self._verify_regions(
                    image, refinement.selected_regions, sx, sy,
                ),
            )
        return OcrPage(
            index, width, height, "pdf_point", int(page.get_rotation()),
            float(native_page.page_obj.attrs.get("UserUnit", 1)),
            self.dpi if run_ocr else None, len(native), rejected, run_ocr,
            round(perf_counter() - started, 3), tuple(regions),
            refinement,
        )

    # 只检测解析器可见字符缺口
    # 不证明路径文字覆盖
    # 同色彩策略排除近白字
    # 只检查ASCII字母或数字
    # 越页字符不作为缺口
    # 每64 point建立纵向桶
    # 减少逐字对全部框比较
    # 中心落桶后检查完整包含
    # 微小误差容忍为0.01
    # 发现一个缺口就触发补OCR
    # 没有缺口也不等于整图无漏字
    @staticmethod
    def _native_coverage_incomplete(page, native: list[TextRegion],
                                    width: float, height: float) -> bool:
        """Detect a native-parser coverage gap, not general vector text coverage.

        Use the same color visibility policy as native words. This cannot detect
        letters converted to paths, prove arbitrary clipping/paint visibility, or
        establish that OCR will recover a missing glyph. A vertical bucket index
        avoids comparing every page character with every accepted word.
        """
        strips: dict[int, list[TextBounds]] = {}
        for region in native:
            box = region.bounds
            for strip in range(int(box.min_y // 64), int(box.max_y // 64) + 1):
                strips.setdefault(strip, []).append(box)
        for char in page.chars:
            if not any(ch.isascii() and ch.isalnum() for ch in char.get("text", "")):
                continue
            color = char.get("non_stroking_color")
            if (isinstance(color, (tuple, list)) and len(color) >= 3
                    and min(color[:3]) >= 0.94):
                continue
            x0, top, x1, bottom = (float(char[k]) for k in ("x0", "top", "x1", "bottom"))
            if not (0 <= x0 < x1 <= width and 0 <= top < bottom <= height):
                continue
            boxes = strips.get(int(((top + bottom) / 2) // 64), ())
            if not any(box.min_x - 0.01 <= x0 and x1 <= box.max_x + 0.01
                       and box.min_y - 0.01 <= top and bottom <= box.max_y + 0.01
                       for box in boxes):
                return True
        return False

    # 只接受已知非文字对象类型
    # Form可能嵌套文字
    # 因此Form不能快速判无文字
    # 对象数量异常走保守路线
    # 再查一次textpage字符数
    # 两种证据一致才跳过解析
    # textpage总在finally关闭
    # 任何异常返回不确定False
    # 不因优化丢失复杂原生内容
    @staticmethod
    def _native_text_absent(page) -> bool:
        """Prove a simple non-text page; uncertainty always uses the full parser.

        Only path/image/shading objects are safe. Forms may contain nested text,
        so even an apparently empty Form forces the original pdfplumber route.
        A second text-page check guards against an inconsistent object listing.
        No document verdict is cached: mixed PDFs are decided page by page.
        """
        import pypdfium2.raw as raw

        try:
            count = raw.FPDFPage_CountObjects(page)
            if count < 0:
                return False
            allowed = {raw.FPDF_PAGEOBJ_PATH, raw.FPDF_PAGEOBJ_IMAGE,
                       raw.FPDF_PAGEOBJ_SHADING}
            for index in range(count):
                obj = raw.FPDFPage_GetObject(page, index)
                if not obj or raw.FPDFPageObj_GetType(obj) not in allowed:
                    return False
            textpage = page.get_textpage()
            try:
                return textpage.count_chars() == 0
            finally:
                textpage.close()
        except Exception:
            return False

    # 逐区域附加图像核验
    # 单项失败不丢整页结果
    # 异常记录为核验警告
    # 原始source_text保持不动
    # 倍率用于定位工程字符ROI
    # 不授予任何遮挡资格
    @staticmethod
    def _verify_regions(
        image: Image.Image, regions: tuple[TextRegion, ...],
        x_scale: float, y_scale: float,
    ) -> tuple[TextRegion, ...]:
        verified = []
        for region in regions:
            try:
                verified.append(verify_engineering_region(
                    image, region, x_scale=x_scale, y_scale=y_scale,
                ))
            except Exception as exc:
                verified.append(replace(
                    region,
                    verification_warnings=(
                        f"d8c_character_verification_failed:{type(exc).__name__}",
                    ),
                ))
        return tuple(verified)

    # 请求原生词连同字符证据
    # 没有字符不信任词结果
    # 先执行候选文字保留策略
    # 近白色常是扫描隐藏OCR层
    # 默认颜色未知仍可视为候选
    # 这里不是完整绘制可见性证明
    # 文字必须完整位于页面内
    # 原生词没有OCR置信度
    # matrix给出文字阅读角度
    # y轴方向换算使用负号
    # 四角仍用轴对齐词包围框
    # source标为pdf_native
    # 遮挡另需独立背景证据
    @staticmethod
    def _visible_words(page, doc_id: str, index: int, width: float, height: float,
                       *, enable_integrated_quality: bool = False):
        accepted: list[TextRegion] = []
        rejected = 0
        for word in page.extract_words(return_chars=True):
            chars = word.get("chars", [])
            if not chars or not english_candidate(
                    word["text"], enable_integrated_quality=enable_integrated_quality):
                continue
            # 常见扫描 PDF 的隐藏 OCR 层为白色；不能直接当成可见英文。
            colors = [c.get("non_stroking_color") for c in chars]
            visible = any(
                color is None or not isinstance(color, (tuple, list))
                or len(color) < 3 or min(color[:3]) < 0.94
                for color in colors
            )
            x0, top, x1, bottom = (float(word[k]) for k in ("x0", "top", "x1", "bottom"))
            if not visible or x0 < 0 or top < 0 or x1 > width or bottom > height:
                rejected += 1
                continue
            bounds = TextBounds(x0, top, x1, bottom)
            polygon = ((x0, top), (x1, top), (x1, bottom), (x0, bottom))
            matrix = chars[0].get("matrix")
            angle = degrees(atan2(-float(matrix[1]), float(matrix[0]))) if matrix else 0.0
            accepted.append(TextRegion(
                str(uuid4()), doc_id, index, word["text"], "", bounds,
                rotation_degrees=round(angle, 2), confidence=None,
                polygon=polygon, source="pdf_native",
            ))
        return accepted, rejected

    # 先要求字面忽略大小写相同
    # 再计算交叠面积
    # 分母取OCR候选面积
    # 阈值用于原生/OCR去重
    # 不是通用IoU匹配
    # 文本不同保留双方证据
    @staticmethod
    def _duplicate(a: TextRegion, b: TextRegion) -> bool:
        if a.source_text.casefold().strip() != b.source_text.casefold().strip():
            return False
        x_overlap = max(0, min(a.bounds.max_x, b.bounds.max_x) - max(a.bounds.min_x, b.bounds.min_x))
        y_overlap = max(0, min(a.bounds.max_y, b.bounds.max_y) - max(a.bounds.min_y, b.bounds.min_y))
        a_area = max(1, (a.bounds.max_x - a.bounds.min_x) * (a.bounds.max_y - a.bounds.min_y))
        return x_overlap * y_overlap / a_area > 0.5
