# mechanical_drawing_translator/annotation_models.py
"""预览与导出共享的不可变中文标注布局模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import TextBounds


# 状态表达本次布局的遮挡分析阶段，不是永久授权凭证。
# NOT_REQUESTED 表示用户未申请，不能保留可绘制蒙版。
# PENDING_ANALYSIS 必须在正式导出前消解。
# SAFE 需要调用链上的独立原生对象证据，单独存储的枚举不够。
# UNSAFE_GEOMETRY 与背景不支持分开，便于解释保留英文的原因。
# 旧项目读取到 SAFE 后也必须重新走当前规则，不能直接恢复擦除。
class MaskDecision(str, Enum):
    NOT_REQUESTED = "not_requested"
    PENDING_ANALYSIS = "pending_analysis"
    SAFE = "safe"
    UNSAFE_GEOMETRY = "unsafe_geometry"
    UNSUPPORTED_BACKGROUND = "unsupported_background"


# 一页的身份和尺寸与 UI 视口分离，缩放只发生在绘制变换层。
# pdf_point 表示显示方向上的 PDF point，pixel 表示定向后原图像素。
# 页索引从零开始；显示页码由界面单独加一。
# crop_origin 保留 PDF 裁剪原点信息，不能与左上角显示坐标混为一谈。
# frozen 防止后台任务途中被 UI 修改同一个页面对象。
# 此模型的轻量验证不代替导出边界对有限数值的完整检查。
@dataclass(frozen=True)
class PageGeometry:
    document_id: str
    page_index: int
    width: float
    height: float
    coordinate_unit: str
    crop_origin_x: float = 0.0
    crop_origin_y: float = 0.0

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("page index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("page dimensions must be positive")
        if self.coordinate_unit not in {"pdf_point", "pixel"}:
            raise ValueError("unsupported page coordinate unit")


# 多边形顶点始终使用所属页面源单位。
# 不存屏幕像素，避免保存后在另一缩放比例下错位。
# tuple 保留顶点顺序；遮挡绘制器按顺序连线。
# 多边形数量与原区域数量可以不同，不应凭数量推断审核资格。
Polygon = tuple[tuple[float, float], ...]


# 单项串联不可变原文关联、译文、排版结果和遮挡诊断。
# group_id 是译文组身份，region_ids 仍指向原始识别区域。
# source_polygons 记录定位证据，mask_polygons 记录候选视觉覆盖。
# 两者不能互换；空蒙版可与可显示中文同时存在。
# text_bounds 是未旋转文字框，最终占用需结合 rotation_degrees。
# lines 是已经度量并换行的完整译文，不由 painter 二次拆分。
# placement 决定是否有墨迹，待布局记录仍保留列表关联和译文。
# provenance 描述翻译来源，人工拖动位置不会把机器译文变人工审核。
@dataclass(frozen=True)
class AnnotationItem:
    group_id: str
    region_ids: tuple[str, ...]
    source_text: str
    translated_text: str
    provenance: str
    source_polygons: tuple[Polygon, ...]
    text_bounds: TextBounds
    font_size: float
    lines: tuple[str, ...]
    rotation_degrees: float
    mask_polygons: tuple[Polygon, ...]
    mask_decision: MaskDecision
    warnings: tuple[str, ...] = ()
    fallback_text_bounds: TextBounds | None = None
    placement: str = 'inline'
    display_index: int = 0
    # New tight-ink layouts store their first baseline in the unrotated box;
    # None preserves the historical ascent/line-spacing painter exactly.
    # 显式原点是第一条基线相对未旋转文字框左上角的偏移。
    # 紧字形适配可能得到正负 bearing，故它不是简单的固定 ascent。
    # 只有紧字形布局填写该值，旧项目 None 继续原有绘制约定。
    # 改变原点会改变实际墨迹范围，恢复及绘制前必须再验证。
    text_origin: tuple[float, float] | None = None
    # Missing fields in old data retain the historical white glyph mask. A
    # measured canvas records its exact paper RGB and immutable proof receipt.
    # 填色只描述覆盖副本的颜色，不证明纸面背景可被擦除。
    # 历史灰底画布数据保留用于追溯，当前生产不接受其 OCR 擦除授权。
    # evidence 是诊断收据文本，不可替代独立背景再验证。
    # 缺省白色是兼容存储值，不表示扫描件原纸一定纯白。
    mask_fill: tuple[int,int,int] = (255,255,255)
    mask_evidence: tuple[str,...] = ()

    def __post_init__(self) -> None:
        # 明确排除 bool，防止 True/False 冒充 RGB 整数通道。
        # 三通道都必须在 0..255，避免 QColor 隐式截断坏项目数据。
        # 收据必须是字符串元组，保持保存格式稳定。
        # 模型只检查形状和基础值；证据真实性由使用边界判断。
        # 文本原点还需要有限性，NaN 会使普通大小比较失效。
        # 构造成功不代表 item 可以导出，也不代表蒙版有权绘制。
        if (not isinstance(self.mask_fill,tuple) or len(self.mask_fill)!=3
            or any(type(v)is not int or not 0<=v<=255 for v in self.mask_fill)):
            raise ValueError('Invalid annotation mask fill')
        if not isinstance(self.mask_evidence,tuple) or any(not isinstance(v,str)for v in self.mask_evidence):
            raise ValueError('Invalid annotation mask evidence')
        if self.text_origin is not None:
            import math
            if (not isinstance(self.text_origin,tuple) or len(self.text_origin)!=2
                or any(not isinstance(v,(int,float)) or isinstance(v,bool)
                       or not math.isfinite(v) for v in self.text_origin)):
                raise ValueError('Invalid annotation text origin')


# 布局是整页快照，预览、导出复用同一几何而不重新猜位置。
# preview_only 是独立门禁，机器自动预览必须保留它。
# display_mode 的 legacy/chinese 是旧数据兼容，不增设当前第三入口。
# 正式界面使用英文与中英对照，安全遮挡是双语的独立选项。
# layout_version 让旧几何缓存失效，不等于翻译模型或知识资源版本。
# effective_bounds 只保存可靠测得的图框，未知时为 None。
# frame_evidence 解释图框方法与拒绝原因，不能只保存一个成功布尔值。
# preview_width 保留历史接口数据，不能用于扩张源画布输出。
@dataclass(frozen=True)
class AnnotationLayout:
    document_id: str
    page_index: int
    page_width: float
    page_height: float
    coordinate_unit: str
    font_family: str
    mask_requested: bool
    items: tuple[AnnotationItem, ...]
    preview_only: bool = False
    preview_width: float = 0.0
    # Older project objects default to legacy rendering, never masquerade as
    # the stricter in-place Chinese replacement mode.
    display_mode: str = 'legacy'
    layout_version: str = 'legacy-v1'
    effective_bounds: TextBounds | None = None
    frame_evidence: tuple[str,...] = ()

    def __post_init__(self) -> None:
        if self.effective_bounds is not None:
            import math
            # 记录图框必须严格包含于纸张且有正面积。
            # 检查有限数值后再比较，避免无穷大和 NaN 绕过边界。
            # 图框是额外收紧条件，缺少它不意味着可画到纸张外。
            # 此处不根据 OCR 文字范围推断图框，测量交给专门检测模块。
            b=self.effective_bounds
            if (not isinstance(b,TextBounds) or not all(math.isfinite(v) for v in
                    (b.min_x,b.min_y,b.max_x,b.max_y)) or b.min_x<0 or b.min_y<0
                    or b.max_x>self.page_width or b.max_y>self.page_height
                    or b.min_x>=b.max_x or b.min_y>=b.max_y):
                raise ValueError('Invalid recorded drawing frame')

    # 聚合顺序跟随 items 顺序，保证界面和导出报告可对应。
    # 警告不在模型内去重或改写，保留每项产生原因的上下文。
    # 这个属性只读已有诊断，不重新运行背景分析或翻译。
    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(warning for item in self.items for warning in item.warnings)
