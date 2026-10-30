# mechanical_drawing_translator/font_resources.py
"""加载随程序分发、允许再分发的中文标注字体。"""

from functools import lru_cache
from pathlib import Path
from .resource_paths import package_resource

from PySide6.QtGui import QFont, QFontDatabase, QRawFont


# 字体随项目提供，运行时不下载，也不依赖目标机安装某个中文字体。
# 哈希常量记录预期资源身份，实际加载函数本身不执行哈希核验。
# 资源完整性与许可证需在交付/审计路径另外核对。
# OFL 文件和来源说明属于字体分发必需证据，不能只带 OTF。
FONT_SHA256 = "faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9"


# 定位相对模块资源目录，不硬编码开发机路径或当前工作目录。
# 入口脚本从不同目录启动时也应定位同一份打包字体。
# 该路径只读资源，不用于保存用户图纸或运行时数据。
def font_resource_path() -> Path:
    return package_resource('fonts', 'NotoSansSC-Regular.otf')


# 应用字体注册一次后复用族名，避免每条标注重复加载大字体。
# 缓存的是 Qt 注册结果，不缓存翻译资格或布局背景证据。
# 需要有效 Qt 应用环境；离线 CPU 解码无需导入此 GUI 资源模块。
# 字体缺失或 Qt 注册失败明确报错，不能静默落到系统替代字体。
@lru_cache(maxsize=1)
def load_annotation_font() -> str:
    path = font_resource_path()
    if not path.is_file():
        raise RuntimeError(f"packaged annotation font is missing: {path}")
    # 族名以 Qt 对真实字体的识别结果为准，不靠文件名猜测。
    # 注册返回负值或族名为空都表示该资源不可用。
    # 把错误交给预览/导出边界处理，防止布局度量和实际字体不同。
    # 返回第一个注册族后，所有字号共享同一字体资源。
    font_id = QFontDatabase.addApplicationFont(str(path))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if not families:
        raise RuntimeError("packaged annotation font could not be loaded")
    return families[0]


# 字号使用 setPixelSize，使源坐标度量不受系统 point/DPI 换算干扰。
# PDF writer 配置为 72 DPI 时一个 painter 单位恰为一个 PDF point。
# round 是 Qt 整数字号接口要求，适配与绘制必须调用同一函数。
# 禁用 hinting 减少设备特定像素对齐对几何的一致性影响。
# 这不宣称不同平台字体渲染逐像素相同，仍需目标机视觉验证。
def annotation_font(pixel_size: float) -> QFont:
    if pixel_size <= 0:
        raise ValueError("annotation font size must be positive")
    font = QFont(load_annotation_font())
    font.setPixelSize(max(1, round(pixel_size)))
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


# 检查实际 glyph 索引，避免 CID CJK OTF 的 supportsCharacter 假阴性。
# 空白不需要可见字形，其它字符索引零表示缺字。
# 缺字集合排序后统一报错，不删除未知工程符号继续排版。
# 字体可用与翻译专业正确是两个独立问题。
# 同一检查供自动布局、紧字形和正式排版复用，保证缺字拒绝一致。
def ensure_font_supports(text: str, pixel_size: float = 16) -> None:
    raw = QRawFont.fromFont(annotation_font(pixel_size))
    if not raw.isValid():
        raise RuntimeError("packaged annotation font is not usable")
    # Qt/FreeType 对 CID-keyed CJK OTF 的 supportsCharacter() 存在假阴性；
    # glyph index 为 0 才表示实际缺字，这也与最终 QPainter 使用的映射一致。
    missing = sorted({character for character in text if not character.isspace()
                      and raw.glyphIndexesForString(character) == [0]})
    if missing:
        raise ValueError(f"packaged annotation font lacks glyphs: {''.join(missing)}")
