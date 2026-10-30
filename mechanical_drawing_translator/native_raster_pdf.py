# mechanical_drawing_translator/native_raster_pdf.py
"""Create a PDF image object from original channels, never the OCR RGB preview.

JPEG with ordinary orientation retains its encoded DCT stream verbatim. Other
inputs are oriented once and stored with lossless Flate. CMYK channels and ICC
are retained; transparent samples use a separate soft mask. One layout pixel is
one PDF point so vector captions share the exact source coordinate contract.
"""
from pathlib import Path
from PIL import Image, ImageOps, ImageCms
import io
from pypdf import PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, EncodedStreamObject,
                          DictionaryObject, NameObject, NumberObject)
from .image_metadata import validate_raster_sample_depth

# 图片转 PDF 的正式保真基底，输入直接来自源文件而非 OCR/Qt RGB 预览。
# 一原像素对应一 PDF point，使已有像素布局能直接叠加矢量中文。
# 这种坐标合同不声称沿用原图 DPI 对应的物理打印尺寸。
# 支持的源模式和位深明确受限，未知格式组合应拒绝而不默默降位。
# 目标是临时输出副本，源文件始终只读。
# JPEG 普通方向可保留原 DCT 编码；其它情况保留解码后原通道样本。
# 带 EXIF 旋转的 JPEG 会定向后以 Flate 存储，避免第二次旋转。
# RGB 预览和 CMYK 原生通道输出属于不同路径，不能混用 profile。
def write_native_raster_pdf(source: Path, destination: Path) -> tuple[int,int]:
    """Embed native decoded samples or the untouched JPEG coding in one page.

    ICC is validated against the native channel model before embedding. An
    arbitrary RGB preview profile must never be attached to a CMYK image. The
    Adobe JPEG transform inverts CMYK samples, hence the explicit Decode array.
    EXIF-oriented JPEGs use lossless decoded samples to avoid a second rotation.
    """
    # PNG 的编码位深须先读 IHDR，Pillow mode 可能已把 16 位 RGB 降为 8 位。
    # 检查放在解码前，避免降位后才误认为输入本来只有八位。
    # 多帧图不能自动选择第一帧冒充整张图纸。
    # 支持的二值、灰度、调色板在下文明确展开为 PDF 支持通道。
    # 本函数不做 OCR、不查翻译资格，资格由导出服务先行检查。
    validate_raster_sample_depth(source)
    with Image.open(source) as original:
        original.load()
        if getattr(original,'n_frames',1)!=1 or original.mode not in {'1','L','LA','P','RGB','RGBA','CMYK'}:
            raise ValueError('unsupported raster bit depth/color mode for PDF')
        # EXIF 定向只消费一次，后续尺寸就是布局使用的视觉尺寸。
        # LA/RGBA 或调色板透明先分离 alpha，避免 PDF 失去透明信息。
        # 颜色样本放主图像，alpha 放独立灰度软蒙版。
        # 二值扩成八位灰度是明确样本表示变化，不伪称原编码字节保持。
        # 调色板无透明时展开 RGB，索引色表不直接复用为错误颜色空间。
        image=ImageOps.exif_transpose(original)
        alpha=None
        if image.mode in {'RGBA','LA'} or 'transparency' in image.info:
            image=image.convert('RGBA');alpha=image.getchannel('A');image=image.convert('RGB')
        elif image.mode=='P':image=image.convert('RGB')
        elif image.mode=='1':image=image.convert('L')
        # 通道数同时决定 Device 色空间和 ICC 流的 /N。
        # profile 真实颜色空间必须与输出样本通道一致。
        # 不能给 CMYK 样本贴 RGB/sRGB 标签来绕过色彩转换。
        # 灰度 ICC 只有在灰度样本路径才匹配，透明灰度展开 RGB 后不能误贴。
        # profile 无法解析或不匹配时拒绝输出，保留源文件供用户选择其它流程。
        channels={'L':1,'RGB':3,'CMYK':4}[image.mode]
        icc=original.info.get('icc_profile')
        if icc:
            profile=ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if profile.profile.xcolor_space.strip() != {'L':'GRAY','RGB':'RGB','CMYK':'CMYK'}[image.mode]:
                raise ValueError('ICC profile does not match native PDF channels')
        # 新 PDF 页只包装原图像对象，中文覆盖层随后由共享矢量 painter 生成。
        # 页面宽高用定向后像素数，不用屏幕缩放或工作图缩略尺寸。
        # 普通方向 JPEG 直接作为 /DCTDecode 流，保留原压缩字节。
        # EXIF 转向或其它格式则以 Flate 压缩实际样本，不进行 JPEG 二次编码。
        # 这两类保留合同不同，不能把重新编码 JPEG 称为无损。
        writer=PdfWriter();page=writer.add_blank_page(width=image.width,height=image.height)
        keep_jpeg=original.format=='JPEG' and original.getexif().get(274,1)==1
        if keep_jpeg:
            obj=EncodedStreamObject();obj._data=source.read_bytes();obj[NameObject('/Filter')]=NameObject('/DCTDecode')
        else:
            obj=DecodedStreamObject();obj.set_data(image.tobytes());obj=obj.flate_encode()
        # 主图像明确写入宽高与八位通道精度，PDF 查看器不再猜测样本布局。
        # 色空间缺省按真实通道数选择 Gray/RGB/CMYK。
        # ICC 存在时用 ICCBased 包装，并保留 Device 色空间作为 Alternate。
        # ICC 流单独 Flate 压缩只改变编码，不改 profile 字节语义。
        # 所有对象由当前 writer 注册，不能引用已关闭文档的间接对象。
        obj.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Image'),
                    NameObject('/Width'):NumberObject(image.width),NameObject('/Height'):NumberObject(image.height),
                    NameObject('/BitsPerComponent'):NumberObject(8)})
        color=NameObject({1:'/DeviceGray',3:'/DeviceRGB',4:'/DeviceCMYK'}[channels])
        if icc:
            profile=DecodedStreamObject();profile.set_data(icc);profile[NameObject('/N')]=NumberObject(channels)
            profile[NameObject('/Alternate')]=color
            color=ArrayObject([NameObject('/ICCBased'),writer._add_object(profile.flate_encode())])
        obj[NameObject('/ColorSpace')]=color
        # Adobe CMYK JPEG 的样本反相约定用 /Decode 显式表达。
        # 只有保留 DCT 编码且有 Adobe 标记时使用，不能对所有 CMYK 一律反相。
        # 解码后的 Flate CMYK 已是实际通道样本，不再重复这次 Decode。
        # 这保证原压缩 JPEG 与经解码路径各自遵守正确通道解释。
        if keep_jpeg and image.mode=='CMYK' and 'adobe' in original.info:
            obj[NameObject('/Decode')]=ArrayObject([NumberObject(v) for v in (1,0)*4])
        # 软蒙版与颜色图像使用完全相同的定向后尺寸。
        # alpha 存为八位 DeviceGray，零透明与全不透明由 PDF 软蒙版解释。
        # 不先合成白纸，因此透明区域仍可在支持的查看器中保持透明。
        # 主图颜色与软蒙版分离，不把 alpha 当第四个 RGB 颜色通道。
        if alpha is not None:
            mask=DecodedStreamObject();mask.set_data(alpha.tobytes());mask=mask.flate_encode()
            mask.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Image'),
                         NameObject('/Width'):NumberObject(image.width),NameObject('/Height'):NumberObject(image.height),
                         NameObject('/BitsPerComponent'):NumberObject(8),NameObject('/ColorSpace'):NameObject('/DeviceGray')})
            obj[NameObject('/SMask')]=writer._add_object(mask)
        # 资源字典只注册实际源图像，绘制流使用 /SourceImage 名称引用它。
        # 矩阵把单位图像映射到完整源像素宽高，q/Q 隔离图形状态。
        # 没有把原图放大到 OCR 工作分辨率，也没有用截图重建原页。
        # Producer/Subject 描述包装合同，不能当作专业翻译或视觉终验凭证。
        # with 保证写入流关闭，随后调用者才能独立重新打开验证。
        # 返回视觉尺寸供导出服务与已保存布局核对，错配就拒绝叠字。
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/XObject'):DictionaryObject({NameObject('/SourceImage'):writer._add_object(obj)})})
        stream=DecodedStreamObject();stream.set_data(f'q {image.width} 0 0 {image.height} 0 0 cm /SourceImage Do Q'.encode('ascii'))
        page[NameObject('/Contents')]=writer._add_object(stream)
        writer.add_metadata({'/Producer':'MechanicalDrawingTranslator native image object',
                             '/Subject':'Native pixels retained; PDF layout uses one pixel per point'})
        with destination.open('wb') as output:writer.write(output)
        return image.size
