# mechanical_drawing_translator/models.py
"""与界面缩放无关的文档和文字区域数据结构。"""

from dataclasses import asdict, dataclass
from pathlib import Path


# 一次打开生成文档UUID
# UUID不同于文件内容哈希
# 项目恢复可还原旧UUID
# path只表示本地源图
# 页数由实际解析器提供
# 显示名称不携带父目录
@dataclass(frozen=True)
class DocumentInfo:
    """一次打开操作对应的本地图纸元数据。"""

    id: str
    path: Path
    file_type: str
    page_count: int

    @property
    def name(self) -> str:
        return self.path.name


# 所有边界均为源坐标
# PDF使用展示页point
# 图片使用定向后像素
# 这里不包含视图缩放
# 矩形是轴对齐包围框
# 旋转文字另存polygon
# 反向边界立即拒绝
# 有限数值还由载入层检查
@dataclass(frozen=True)
class TextBounds:
    """原始图片像素或 PDF 页面的点坐标，不是屏幕坐标。"""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    # 仅检查区间方向
    # 不在这里裁剪到页面
    # 页面尺寸由几何对象掌握
    def __post_init__(self) -> None:
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError("文字区域坐标范围无效")


# 冻结对象保留OCR原文
# 译文不覆盖source_text
# 原始区域ID用于追溯
# 页码采用零起始索引
# confidence可为空
# 原生PDF没有OCR置信度
# polygon保留检测形状
# bounds供布局粗筛
# rotation表示阅读方向
# source标识原生或OCR
# verified文本是旁路证据
# 它不是原字面的替代存档
# 核验来源和警告一起保存
# 局部复识别通过新ID追溯
@dataclass(frozen=True)
class TextRegion:
    """后续 OCR 与翻译模块交换的可序列化文字对象。"""

    id: str
    document_id: str
    page_index: int
    source_text: str
    translated_text: str
    bounds: TextBounds
    rotation_degrees: float = 0.0
    confidence: float | None = None
    status: str = "待处理"
    polygon: tuple[tuple[float, float], ...] = ()
    source: str = "ocr"
    verified_source_text: str | None = None
    verification_source: str | None = None
    verification_confidence: float | None = None
    verification_warnings: tuple[str, ...] = ()

    # 转换为纯数据容器
    # 嵌套边界随asdict展开
    # 不包含任何Qt对象
    def to_dict(self) -> dict:
        """返回可直接交给 json.dumps 的纯数据。"""
        return asdict(self)

