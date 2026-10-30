# mechanical_drawing_translator/image_metadata.py
"""Raster decoding/encoding contract shared by preview and reviewed export.

Source pixels stay in the source file. Output coordinates describe the visually
oriented image, so EXIF rotation is consumed exactly once before layout begins.
An RGB ICC profile remains attached to unchanged RGB samples; CMYK needs a real
profile transform and cannot truthfully be described as lossless RGB output.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageCms, ImageOps
from PIL.PngImagePlugin import PngInfo
if TYPE_CHECKING:
    from PySide6.QtGui import QImage


# 元数据对应完成 EXIF 定向后的 RGB(A) 工作样本，不是原编码坐标。
# size 和 dpi 轴必须与当前视觉宽高一致，横竖转向后不能沿用旧轴。
# ICC 解释当前样本，CMYK 转 RGB 后必须改为转换目标 profile。
# exif 是已清理方向与过期结构信息的副本。
# source_mode/source_format 仍保留原始模式，供导出限制和警告使用。
# warnings 把近似转换、模式展开等损失带到最终导出结果。
# png_text 只保留可解释的文本键值，不复制任意编码结构块。
# 不可变记录避免后台编码时被界面替换某一项。
@dataclass(frozen=True)
class RasterMetadata:
    """Metadata for *oriented* RGB(A) pixels, never raw camera coordinates.

    ICC bytes describe the returned samples, not necessarily the original CMYK
    profile. Density axes follow 90-degree rotations. EXIF orientation is reset
    before encoding; copying the original orientation would rotate output twice.
    """

    size: tuple[int, int]
    dpi: tuple[float, float] | None
    icc_profile: bytes | None
    exif: bytes | None
    has_alpha: bool
    warnings: tuple[str, ...] = ()
    source_format: str = ''
    source_mode: str = ''
    png_text: tuple[tuple[str, str], ...] = ()


def validate_raster_sample_depth(path: str | Path) -> None:
    """Inspect encoded depth before a decoder normalizes its mode name.

    Pillow exposes 16-bit RGB/RGBA PNG as RGB/RGBA after reducing samples to
    eight bits. Checking ``image.mode`` alone therefore misses destructive
    normalization. The PNG IHDR depth byte is authoritative for this guard.
    """
    # 从 PNG IHDR 原编码字段判断位深，优先于解码器规范化后的 mode。
    # Pillow 可能把 16 位 RGB/RGBA 直接呈现为八位模式名称。
    # 若只查 mode 就会默默丢精度，再也无法证明原样本保留。
    # 这里拒绝高于八位的 PNG，源文件只读且保留。
    # 该检查不是完整文件验证，后续 Image.open/load 仍负责解析正确性。
    with Path(path).open('rb') as stream:
        header=stream.read(26)
    if (header[:8] == b'\x89PNG\r\n\x1a\n' and len(header)==26
            and header[12:16]==b'IHDR' and header[24]>8):
        raise ValueError(f'unsupported raster bit depth: {header[24]}-bit PNG; source retained')


def read_raster_pixels(path: str | Path) -> tuple[Image.Image, RasterMetadata]:
    """Decode locally without importing Qt, for CPU services and Qt adapters.

    RGB ICC bytes are retained rather than replaced with an assumed sRGB tag.
    Unsupported high-bit-depth images are rejected: silently reducing engineering
    imagery to eight bits would violate the resolution/color fidelity contract.
    """
    # CPU 解码入口避免导入 Qt，OCR 服务可独立复用一致定向与颜色逻辑。
    # 先检查编码位深，再强制 load，使文件错误在受控上下文中出现。
    # 多帧图与未知颜色模式明确拒绝，不擅选第一帧或静默降位。
    # 所有后续修改发生在解码副本和 EXIF 副本，不写入原文件。
    # 图片预览与正式原通道 PDF 嵌入路径有不同颜色合同，应分别调用。
    warnings: list[str] = []
    validate_raster_sample_depth(path)
    with Image.open(path) as original:
        original.load()
        if getattr(original, 'n_frames', 1) != 1:
            raise ValueError('animated/multiple-frame raster is not a single drawing')
        if original.mode not in {'1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'CMYK'}:
            raise ValueError(f'unsupported raster bit depth/color mode: {original.mode}')
        # 布局坐标以用户看到的定向图片为准，原相机像素坐标不直接外露。
        # 方向先读取留给 DPI 轴交换判断，再消费 EXIF 变换。
        # dpi 是物理密度，不是界面缩放，也不参与自动提高像素分辨率。
        # JPEG 缺 JFIF 物理单位时，Pillow 可能把 EXIF XResolution 重复到双轴。
        # 因此优先从可解释的 EXIF 单位与两个独立有理数恢复真实密度。
        orientation = original.getexif().get(274, 1)
        density = original.info.get('dpi')
        source_exif=original.getexif()
        # JPEG's Pillow fallback repeats XResolution for both axes when JFIF
        # has no physical units. Read both authoritative EXIF rationals instead;
        # otherwise non-square density is already lost before orientation.
        if ((density is None or (original.format=='JPEG' and original.info.get('jfif_unit',0)==0))
                and source_exif.get(296) in {2,3}
                and 282 in source_exif and 283 in source_exif):
            # EXIF 单位 3 是每厘米，转换为每英寸时乘 2.54。
            # 必须同时存在 X/Y 分辨率，不能凭单轴猜另一轴。
            # 方向 5..8 会交换视觉坐标轴，因此密度也同步反序。
            # NaN、无穷大、零或负密度不写入合同，避免错误物理尺寸。
            # 原宽高比与像素数由 exif_transpose 独立保持，不用 DPI 重采样。
            factor=2.54 if source_exif[296]==3 else 1.0
            density=(float(source_exif[282])*factor,float(source_exif[283])*factor)
        dpi = None
        if density is not None:
            values = tuple(float(v) for v in density[:2])
            if len(values) == 2 and all(math.isfinite(v) and v > 0 for v in values):
                dpi = values[::-1] if orientation in {5, 6, 7, 8} else values
        # EXIF 方向只消费一次，后续输出不能再携带旧旋转标记。
        # 缩略图偏移和长度指向原编码内容，重编码后会过期，必须移除。
        # 普通文档/相机 EXIF 尽量保留，但不复制失效的结构定位信息。
        # 视觉像素宽高同时更新，防止外部阅读器读到自相矛盾尺寸。
        # 不能把处理后的图再按原 Orientation 旋转一次。
        image = ImageOps.exif_transpose(original)
        exif = image.getexif()
        # Retain ordinary camera/document EXIF but remove stale thumbnails and
        # structural offsets. They refer to the old encoded image, not this copy.
        for tag in (274, 513, 514):
            if tag in exif:
                del exif[tag]
        if 40962 in exif:
            exif[40962] = image.width
        if 40963 in exif:
            exif[40963] = image.height
        # PixelX/YDimension normally live in ExifIFD, not IFD0. Keeping the
        # historical encoded dimensions there makes an otherwise normalized
        # Orientation=1 output internally contradictory after a quarter-turn.
        # PixelX/YDimension 常位于 ExifIFD，检查 IFD0 还不够。
        # 转向后必须同步更新嵌套尺寸，否则元数据和实际像素方向冲突。
        # 读取并写回同一嵌套字典，保留其它无关 EXIF 标签。
        # 顶层 ImageWidth/ImageLength 若存在也要跟随视觉尺寸。
        if 34665 in exif:
            nested=exif.get_ifd(34665)
            if 40962 in nested:
                nested[40962]=image.width
            if 40963 in nested:
                nested[40963]=image.height
            exif[34665]=nested
        for tag,value in ((256,image.width),(257,image.height)):
            if tag in exif:
                exif[tag]=value
        # 密度已经统一为每英寸，写回 EXIF 也必须标 ResolutionUnit=2。
        # 若沿用原每厘米单位，下一阅读器会再次乘 2.54 导致错误。
        # 只更新原先存在的分辨率标签，不虚构缺失元数据。
        # 图像方向规范化与密度轴交换同时完成，保持非方形密度一致。
        if dpi is not None:
            if 282 in exif:
                exif[282] = dpi[0]
            if 283 in exif:
                exif[283] = dpi[1]
            # Pillow's info['dpi'] is always per inch. If the original EXIF
            # used centimetres, retaining ResolutionUnit=3 with these values
            # would multiply the physical density by 2.54 on the next reader.
            if 282 in exif or 283 in exif:
                exif[296]=2
        # ICC 必须解释当前样本通道，不能仅作为不透明附件原样搬运。
        # CMYK 有 profile 时执行真实颜色管理转换到 sRGB，记录转换警告。
        # RGB 样本已有 RGB profile 时保留原 profile，不假定全是 sRGB。
        # profile 无法解析或不匹配时拒绝，不带错误标签继续输出。
        # 灰度 ICC 等非 RGB profile 在本 RGB 工作图入口目前保守拒绝。
        # 原生灰度 PDF 嵌入的通道匹配规则在 native_raster_pdf 单独实现。
        icc = original.info.get('icc_profile')
        if icc:
            try:
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                color_space = profile.profile.xcolor_space.strip()
                if original.mode == 'CMYK':
                    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
                    image = ImageCms.profileToProfile(image, profile, srgb, outputMode='RGB')
                    icc = srgb.tobytes()
                    warnings.append('cmyk_icc_converted_to_srgb')
                elif color_space != 'RGB':
                    raise ValueError('profile color space does not match RGB output')
            except Exception as error:
                raise ValueError(f'ICC profile cannot be applied safely: {error}') from error
        # 无 ICC 的 CMYK 无法获得受色彩管理保证的颜色还原。
        # 工作图可做近似 RGB 转换，但必须保留警告，不能宣称无损。
        # 正式 CMYK 栅格标注因此由导出服务拒绝，建议原通道 PDF 路径。
        # 调色板、灰度和二值展开 RGB 也显式记录模式变化。
        # 透明性依据真实 alpha 或 transparency 信息保留，不只查看源后缀名。
        elif original.mode == 'CMYK':
            image = image.convert('RGB')
            warnings.append('cmyk_without_icc_approximate_rgb_conversion')
        has_alpha = image.mode in {'RGBA', 'LA'} or 'transparency' in image.info
        if original.mode in {'1', 'L', 'LA', 'P'}:
            warnings.append('raster_mode_expanded_to_rgb')
        image = image.convert('RGBA' if has_alpha else 'RGB')
        # 保留原 format/mode 便于导出判断是否可直接复制或需拒绝。
        # PNG 文本来自解码器解析出的键值，不从文件拼接未经验证块。
        # 返回 image.copy 脱离 with 中原文件与临时图像的生命周期。
        # 此处未运行任何模型，所有数据仍在本地内存处理。
        metadata = RasterMetadata(image.size, dpi, icc, exif.tobytes() if exif else None,
                                  has_alpha, tuple(warnings), original.format or '',
                                  original.mode, tuple(getattr(original, 'text', {}).items()))
        # Detached Pillow storage remains valid after the original file closes.
        return image.copy(), metadata


def read_raster(path: str | Path) -> tuple[QImage, RasterMetadata]:
    """Adapt the shared oriented pixels to detached, color-tagged Qt storage.

    Qt color-space support is checked here, not in CPU-only decoding. This keeps
    OCR services independent of GUI imports and exposes preview-specific limits.
    """
    # Qt 适配层只在需要 GUI 图像时导入，CPU 解码保持解耦。
    # 字节构造 QImage 借用缓冲区，必须 copy 后才能离开函数。
    # RGB 和 RGBA stride 按真实通道数计算，不用假设四通道。
    # Qt 还需单独验证 ICC 支持，Pillow 可解析不等于 Qt 可显示。
    # setColorSpace 只附解释标签，不在这里重复执行 CMYK 转换。
    from PySide6.QtGui import QColorSpace, QImage
    image, metadata = read_raster_pixels(path)
    # QImage constructed from bytes borrows memory. copy() makes its lifetime
    # independent from Pillow and from this function's temporary byte buffer.
    fmt = QImage.Format.Format_RGBA8888 if metadata.has_alpha else QImage.Format.Format_RGB888
    stride = image.width * (4 if metadata.has_alpha else 3)
    qt_image = QImage(image.tobytes(), image.width, image.height, stride, fmt).copy()
    if metadata.icc_profile:
        space = QColorSpace.fromIccProfile(metadata.icc_profile)
        if not space.isValid():
            raise ValueError('ICC profile is unsupported by Qt preview')
        qt_image.setColorSpace(space)
    if metadata.dpi is not None:
        # Qt 的物理密度单位是点/米，与 Pillow DPI 不同。
        # 除以 0.0254 完成每英寸到每米换算，X/Y 各自保留。
        # 这里只设置元数据，不缩放像素或影响布局保存坐标。
        qt_image.setDotsPerMeterX(round(metadata.dpi[0] / .0254))
        qt_image.setDotsPerMeterY(round(metadata.dpi[1] / .0254))
    return qt_image, metadata


def write_raster(image: QImage, metadata: RasterMetadata, path: Path,
                 output_format: str, jpeg_quality: int) -> tuple[str, ...]:
    """Encode an annotated copy without silently dropping density or ICC.

    PNG preserves RGB(A) samples outside painted pixels. JPEG always re-encodes
    lossily. BMP has no portable alpha/ICC contract in this encoder and flattens
    onto white; its changes are returned to the caller as persistent warnings.
    """
    from PySide6.QtGui import QImage
    # 输出格式由明确参数选择，JPG 统一到 Pillow 的 JPEG 编码名。
    # 警告从解码一路传递，编码阶段再追加自己的能力损失。
    # QImage 先转直通 RGBA，避免把预乘 alpha 误当原始颜色通道。
    # frombytes 必须使用 bytesPerLine，Qt 行对齐并不一定等于宽乘四。
    # 编码只写目标副本，图纸原件从不进入写入分支。
    fmt = 'JPEG' if output_format in {'JPG', 'JPEG'} else output_format
    warnings = list(metadata.warnings)
    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    # bytesPerLine is authoritative even when Qt inserts row alignment padding.
    data = Image.frombytes('RGBA', (rgba.width(), rgba.height()), bytes(rgba.constBits()),
                           'raw', 'RGBA', rgba.bytesPerLine())
    options = {}
    # 只在 PNG 中恢复文本键值，使用编码器重新生成合法 iTXt。
    # 不复制旧 IDAT、尺寸或结构块，它们可能引用已经改变的像素布局。
    # 转其它格式时明确提示文本元数据无法携带。
    # 密度仍单独写入，各格式整数精度差异由回读容差体现。
    if fmt == 'PNG' and metadata.png_text:
        # Preserve human document identifiers as text, without copying arbitrary
        # PNG structural chunks that may reference the old encoded pixel layout.
        text = PngInfo()
        for key, value in metadata.png_text:
            text.add_itxt(key, value)
        options['pnginfo'] = text
    elif metadata.png_text:
        warnings.append('png_text_not_stored_in_other_format')
    if metadata.dpi is not None:
        options['dpi'] = metadata.dpi
    # PNG/JPEG 能保留 ICC 和已规范化的 EXIF，写入解释当前样本的版本。
    # BMP 本编码路径没有可移植 ICC 合同，不可静默丢标签还称同色。
    # 丢 profile 前把 RGB 真正转换为 sRGB，并明确报告 profile 不被存储。
    # alpha 先拆下再恢复，防止颜色转换库误处理透明通道。
    # 此转换不是原通道无损保留，用户应从 warnings 看到区别。
    if fmt in {'PNG', 'JPEG'}:
        if metadata.icc_profile:
            options['icc_profile'] = metadata.icc_profile
        if metadata.exif:
            options['exif'] = metadata.exif
    elif metadata.icc_profile:
        # Never strip an arbitrary RGB profile and pretend its samples are sRGB.
        # Convert to sRGB before losing ICC on BMP; report the conversion/loss.
        profile = ImageCms.ImageCmsProfile(io.BytesIO(metadata.icc_profile))
        srgb = ImageCms.createProfile('sRGB')
        alpha = data.getchannel('A')
        data = ImageCms.profileToProfile(data.convert('RGB'), profile, srgb, outputMode='RGB')
        data.putalpha(alpha)
        warnings.append('bmp_icc_converted_to_srgb_profile_not_stored')
    # 本路径 JPEG/BMP 透明性不能保留，明确合成白纸并报告。
    # PNG RGBA 保留 alpha，不因导出而先铺白底。
    # JPEG 重新编码始终是有损，即使 quality=100 也不称无损。
    # subsampling=0 降低颜色抽样损失，但不能消除 JPEG 量化误差。
    # BMP 的 EXIF 也无法按本编码合同保存，另报 warning。
    # 只有保存成功后返回警告，由导出服务与布局诊断一起报告。
    if fmt in {'JPEG', 'BMP'} and metadata.has_alpha:
        background = Image.new('RGBA', data.size, (255, 255, 255, 255))
        data = Image.alpha_composite(background, data)
        warnings.append('alpha_flattened_on_white')
    if fmt != 'PNG' or not metadata.has_alpha:
        data = data.convert('RGB')
    if fmt == 'JPEG':
        options.update(quality=jpeg_quality, subsampling=0)
        warnings.append('jpeg_reencoded_lossy')
    if fmt == 'BMP' and metadata.exif:
        warnings.append('bmp_exif_not_stored')
    data.save(path, format=fmt, **options)
    return tuple(warnings)


def verify_raster_metadata(path: Path, metadata: RasterMetadata, output_format: str) -> None:
    """Read back density, orientation, alpha and the exact retained profile.

    This checks metadata consistency, not perceptual equivalence across arbitrary
    monitors or CMYK printing workflows. PNG/BMP density is integer pixels/metre;
    JPEG density is integer pixels/inch, hence the documented format tolerance.
    """
    # 使用独立 Pillow 回读输出，不只信任编码器未抛错。
    # 像素尺寸和 Orientation=1 首先确保不会再次旋转或错用布局。
    # 密度比较按格式量化能力给容差，PNG/BMP 为整数每米，JPEG 常为整数每英寸。
    # ICC 在能携带的格式中要求字节一致，不只比较 profile 名称。
    # PNG 源含 alpha 时输出必须还有 A 通道，否则透明性丢失。
    # 这些检查证明元数据一致性，不等于跨显示器/印刷流程颜色终验。
    with Image.open(path) as image:
        image.load()
        if image.size != metadata.size or image.getexif().get(274, 1) != 1:
            raise ValueError('exported raster orientation/dimensions changed')
        if metadata.dpi is not None:
            actual = image.info.get('dpi')
            tolerance = .51 if output_format in {'JPG', 'JPEG'} else .03
            if actual is None or any(abs(a-b) > tolerance for a,b in zip(actual, metadata.dpi)):
                raise ValueError('exported raster density changed')
        if output_format in {'PNG', 'JPG', 'JPEG'} and image.info.get('icc_profile') != metadata.icc_profile:
            raise ValueError('exported raster ICC profile changed')
        if output_format == 'PNG' and metadata.has_alpha and 'A' not in image.getbands():
            raise ValueError('exported PNG transparency was lost')
