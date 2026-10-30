# mechanical_drawing_translator/drawing_view.py
"""图纸场景、缩放和平移，以及原图坐标到场景坐标的转换。"""

import math
from typing import Mapping

from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsItemGroup, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsScene, QGraphicsView,
)

from .annotation_models import AnnotationLayout
from .annotation_painter import paint_annotation_layout
from .models import TextRegion


# 场景项只展示不可变布局
# 不在paint里重新换行
# 预览和导出复用共享painter
# 鼠标缩放不会重算文字位置
class _AnnotationGraphicsItem(QGraphicsItem):
    """场景只缩放共享布局；文字换行、位置和遮挡均不在预览重算。"""

    def __init__(self, layout: AnnotationLayout, scene_size: QSizeF, parent=None):
        super().__init__(parent)
        self.layout = layout
        self._scene_size = QSizeF(scene_size)

    def boundingRect(self) -> QRectF:
        return QRectF(0,0,self._scene_size.width(),self._scene_size.height())

    # 先保存外部画笔状态
    # 按源页到场景比例缩放
    # 共享painter仍接收源坐标
    # 绘制结束恢复状态
    # 避免污染后续场景项
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.save()
        painter.scale(self._scene_size.width() / self.layout.page_width,
                      self._scene_size.height() / self.layout.page_height)
        paint_annotation_layout(painter, self.layout)
        painter.restore()


class DrawingView(QGraphicsView):
    """统一预览图片与 PDF 渲染页，额外保留两层标注场景组。"""

    zoom_changed = Signal(float)
    group_activated = Signal(str)
    annotation_moved = Signal(str, float, float)

    # 底图与文字采用同一场景
    # source_size保留源坐标尺寸
    # page_size固定初始场景像素
    # 高清图替换不改变page_size
    # 视图transform另负责屏幕缩放
    # 底图位于z零
    # OCR/中文/焦点/诊断逐层在上
    # ScrollHandDrag用于平移
    # 鼠标锚点让缩放围绕指针
    # fit_mode表示随窗口自动适应
    # 手动缩放后退出fit_mode
    # inspection布局可在英文模式存在
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drawingView")
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._source_size = QSizeF()
        self._page_size = QSizeF()
        self._has_page = False
        self._fit_mode = False
        self._content_rect = QRectF()
        self._drawing_content_rect = QRectF()
        self._page_item: QGraphicsPixmapItem | None = None
        self._overlay_groups: dict[str, QGraphicsItemGroup] = {}
        self._ocr_items: dict[str, QGraphicsPolygonItem] = {}
        self._ocr_colors: dict[str, QColor] = {}
        self._focused_ocr_ids: tuple[str, ...] = ()
        self._translation_item: _AnnotationGraphicsItem | None = None
        self._inspection_layout = None
        self._diagnostics_visible = False
        self._press_position = None
        self.setMouseTracking(True)

    # 取视图矩阵水平缩放
    # 100%对应场景初始像素
    # 不等同PDF 72 DPI物理大小
    @property
    def zoom_percent(self) -> float:
        return self.transform().m11() * 100.0

    # 返回当前底图实际像素
    # 高清替换后此值会变
    # 场景几何保持原来的page_size
    @property
    def page_raster_size(self) -> QSize:
        return self._page_item.pixmap().size() if self._page_item else QSize()

    # 取底图而非含中文的截图
    # 防止红字被当源工程线
    # 透明图先合成白纸背景
    # 隐藏RGB不应当可见墨迹
    # 返回QImage供本地安全分析
    def page_image(self) -> QImage:
        """返回当前底图副本，供遮挡安全分析使用。"""
        if self._page_item is None:
            return QImage()
        original=self._page_item.pixmap().toImage()
        if not original.hasAlphaChannel():return original
        paper=QImage(original.size(),QImage.Format.Format_RGB32);paper.fill(QColor('white'))
        painter=QPainter(paper);painter.drawImage(0,0,original);painter.end()
        return paper

    # 倍率乘DPR得到设备需求
    # 再取两倍采样改善细节
    # 长边最多6000像素
    # 总像素最多2400万
    # 两种约束取较严格值
    # 保持宽高比例不变
    # 不会据此改变源PDF尺寸
    # 高分屏不能无限分配内存
    def pdf_render_size(self, dpr: float | None = None) -> QSize:
        """按当前画面倍率和设备像素比计算完整 PDF 页的渲染像素。"""
        if not self._has_page:
            return QSize()
        ratio = self.devicePixelRatioF() if dpr is None else dpr
        if ratio <= 0:
            raise ValueError("设备像素比必须大于 0")
        scale = self.transform().m11() * ratio * 2.0
        width = max(1, round(self._page_size.width() * scale))
        height = max(1, round(self._page_size.height() * scale))
        # 只保存当前页一张高分辨率图，限制单张约 92 MiB 的 ARGB 像素。
        factor = min(1.0, 6000.0 / max(width, height),
                     math.sqrt(24_000_000 / (width * height)))
        if factor < 1.0:
            width = max(1, int(width * factor))
            height = max(1, int(height * factor))
        return QSize(width, height)

    # 新pixmap设备比固定一
    # 场景比例显式反向补偿
    # 高分辨率图仍占原场景矩形
    # 源框和中文位置均不重算
    # 视图缩放和相机位置保持
    # 空图不能替代有效底图
    def replace_page_image(self, image: QImage) -> None:
        """只替换底图像素，保持 PDF 页面、OCR 框和视图变换的场景坐标。"""
        if not self._has_page or image.isNull() or self._page_item is None:
            raise ValueError("预览图像或页面无效")
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(1.0)
        self._page_item.setPixmap(pixmap)
        self._page_item.setTransform(QTransform.fromScale(
            self._page_size.width() / pixmap.width(),
            self._page_size.height() / pixmap.height(),
        ))

    # 切页先销毁旧场景对象
    # 源尺寸与首帧像素分开保存
    # 白纸矩形在底图下方
    # 透明源预览仍有白底
    # 可见内容范围仅影响取景
    # 四类叠加层按z值排序
    # 场景范围仍限完整纸面
    # 最后按内容适应窗口
    # 这里不生成翻译或审核结果
    def show_page(self, image: QImage, source_size: QSizeF) -> None:
        if image.isNull() or source_size.isEmpty():
            raise ValueError("预览图像或原始页面尺寸无效")
        self.clear_page()
        scene = self.scene()
        self._page_item = scene.addPixmap(QPixmap.fromImage(image))
        self._page_item.setZValue(0)
        paper=scene.addRect(QRectF(0,0,image.width(),image.height()),QPen(Qt.PenStyle.NoPen),QColor('white'))
        paper.setZValue(-1)
        self._page_size = QSizeF(image.size())
        self._source_size = QSizeF(source_size)
        self._drawing_content_rect = self._visible_content_rect(image)
        self._content_rect = QRectF(self._drawing_content_rect)
        self._has_page = True
        scene.setSceneRect(QRectF(0, 0, image.width(), image.height()))
        # 后续模块可单独向场景组增加 OCR 框和中文标注。
        for name, z_value in (("ocr", 10), ("translation", 20), ('focus',30),('diagnostics',40)):
            group = QGraphicsItemGroup()
            group.setZValue(z_value)
            scene.addItem(group)
            self._overlay_groups[name] = group
        self.fit_page()

    # 返回矩形副本防外部改动
    # 相机范围不反写源数据
    # 叠加项不能制造纸外边栏
    @property
    def content_scene_rect(self) -> QRectF:
        """Stable drawing-content extent; overlays never change camera fitting."""
        return QRectF(self._content_rect)

    # 白色扫描边距只从取景剔除
    # 原始像素和页面不裁剪
    # 透明图按白纸判断墨迹
    # 按bytesPerLine处理行填充
    # RGB三通道任一较暗即算墨迹
    # 248阈值用于相机而非遮挡
    # 空白页保持整页范围
    # 稀疏标注至少取四分之一页
    # 额外留少量视野边距
    # 结果始终与纸面求交
    @staticmethod
    def _visible_content_rect(image: QImage) -> QRectF:
        """Exclude white scanner/page margins without clipping any visible ink.

        This raster rectangle affects only the camera. Source coordinates and
        page/annotation bounds remain unchanged. A sparse caption is given a
        minimum page-relative field of view to avoid extreme magnification.
        """
        import numpy as np
        if image.hasAlphaChannel():
            paper=QImage(image.size(),QImage.Format.Format_RGB32);paper.fill(QColor('white'))
            painter=QPainter(paper);painter.drawImage(0,0,image);painter.end();image=paper
        rgb=image.convertToFormat(QImage.Format.Format_RGB888)
        a=np.frombuffer(rgb.constBits(),dtype=np.uint8,count=rgb.sizeInBytes()).reshape(rgb.height(),rgb.bytesPerLine())[:,:rgb.width()*3].reshape(rgb.height(),rgb.width(),3)
        ink=(a<248).any(axis=2)
        ys=np.flatnonzero(ink.any(axis=1));xs=np.flatnonzero(ink.any(axis=0))
        page=QRectF(0,0,image.width(),image.height())
        if not len(xs) or not len(ys):return page
        rect=QRectF(float(xs[0]),float(ys[0]),float(xs[-1]-xs[0]+1),float(ys[-1]-ys[0]+1))
        min_w=image.width()*.25;min_h=image.height()*.25
        center=rect.center();rect.setWidth(max(rect.width(),min_w));rect.setHeight(max(rect.height(),min_h));rect.moveCenter(center)
        padding=max(4.,min(image.width(),image.height())*.008)
        return rect.adjusted(-padding,-padding,padding,padding).intersected(page)

    # 上层按名称取得固定层
    # 添加内容仍需源到场景转换
    # 不要把源PDF point直接当场景像素
    def overlay_layer(self, name: str) -> QGraphicsItemGroup:
        """返回标注场景组，图纸像素坐标与其场景坐标一致。"""
        return self._overlay_groups[name]

    # 只重建OCR这一层
    # 先清旧ID到图元映射
    # 候选红色疑难橙色排除灰色
    # show_all控制排除项可见性
    # 优先绘制原始polygon
    # 缺polygon才使用bounds四角
    # 逐点映射到固定场景尺寸
    # cosmetic画笔不随放大变粗
    # 提示显示原文/来源/置信度
    # 不能从颜色推断专业正确
    def show_ocr_regions(
        self, regions: list[TextRegion] | tuple[TextRegion, ...],
        categories: Mapping[str, str] | None = None, *, show_all: bool = True,
    ) -> None:
        """红色表示翻译候选，橙色待确认，全部模式用灰色展示排除项。"""
        layer = self.overlay_layer("ocr")
        for child in layer.childItems():
            self.scene().removeItem(child)
        self._ocr_items.clear()
        self._ocr_colors.clear()
        self._focused_ocr_ids = ()
        palette = {
            "translatable": QColor("#dc2626"),
            "review_required": QColor("#d97706"),
            "non_translatable": QColor("#94a3b8"),
        }
        for region in regions:
            category = categories.get(region.id, "translatable") if categories else "translatable"
            if category == "non_translatable" and not show_all:
                continue
            source_points = region.polygon or (
                (region.bounds.min_x, region.bounds.min_y),
                (region.bounds.max_x, region.bounds.min_y),
                (region.bounds.max_x, region.bounds.max_y),
                (region.bounds.min_x, region.bounds.max_y),
            )
            polygon = QPolygonF([self.source_to_scene(QPointF(x, y))
                                 for x, y in source_points])
            item = QGraphicsPolygonItem(polygon, layer)
            color = palette[category]
            pen = QPen(color, 1.5)
            pen.setCosmetic(True)
            item.setPen(pen)
            score = "—" if region.confidence is None else f"{region.confidence:.2f}"
            item.setToolTip(f"{region.source_text}\n来源：{region.source} · 置信度：{score}"
                            f"\n筛选：{category}")
            self._ocr_items[region.id] = item
            self._ocr_colors[region.id] = color

    # 先移除旧中文场景项
    # 布局本身不在这里修改
    # 诊断层读取同一布局对象
    # 无布局时保留原底图
    # 场景边界保持完整纸面
    # 实际inline中文可扩展取景范围
    # 待布局项不制造相机边距
    # 页外历史项不纳入视野
    # fit_mode下必要时重新适应
    # 布局模式决定内容而非视图标签
    def show_translation_layout(self, layout: AnnotationLayout | None) -> None:
        layer = self.overlay_layer("translation")
        previous_width=self.sceneRect().width()
        for child in layer.childItems():
            self.scene().removeItem(child)
        self._translation_item = None
        self._diagnostic_layout=layout
        self._refresh_diagnostics()
        previous_content=QRectF(self._content_rect)
        self._content_rect=QRectF(self._drawing_content_rect)
        self.scene().setSceneRect(QRectF(0,0,self._page_size.width(),self._page_size.height()))
        if layout is None:
            if self._fit_mode and previous_content!=self._content_rect:self.fit_page()
            return
        if (layout.document_id and
                (layout.page_width <= 0 or layout.page_height <= 0)):
            raise ValueError("translation layout page size is invalid")
        item = _AnnotationGraphicsItem(layout, self._page_size, layer)
        self._translation_item = item
        self._inspection_layout=layout
        self.scene().setSceneRect(item.boundingRect())
        # Include actual visible Chinese within the paper. Legacy preview_width,
        # pending text and outside-page items never create a camera margin.
        from .annotation_preview import footprint
        for annotation in layout.items:
            if annotation.placement!='inline' or not annotation.lines:continue
            b=footprint(annotation.text_bounds,annotation.rotation_degrees)
            if b.min_x<0 or b.min_y<0 or b.max_x>layout.page_width or b.max_y>layout.page_height:continue
            a=self.source_to_scene(QPointF(b.min_x,b.min_y));z=self.source_to_scene(QPointF(b.max_x,b.max_y))
            self._content_rect=self._content_rect.united(QRectF(a,z).adjusted(-4,-4,4,4)).intersected(self.sceneRect())
        if self._fit_mode and previous_content!=self._content_rect:self.fit_page()

    # 点击检查可独立于中文绘制
    # 英文模式也能查原文关联
    # 该引用不自动创建可见图层
    def set_inspection_layout(self,layout) -> None:
        self._inspection_layout=layout

    # 仅切诊断轮廓可见性
    # 不改变共享布局或导出文件
    # 不会提升任何mask资格
    def set_diagnostics_visible(self,visible) -> None:
        self._diagnostics_visible=bool(visible);self._refresh_diagnostics()

    # 轮廓单独放诊断层
    # 无布局或隐藏时不创建图元
    # 紫色表示已有mask多边形
    # 橙色源框表示未授遮挡权
    # 青色框表示中文实际footprint
    # cosmetic虚线便于不同倍率检查
    # 这些线不进入共享painter
    # 因此不会随正式导出保存
    def _refresh_diagnostics(self):
        """Outline-only scene diagnostics never enter the shared export layout."""
        layer=self._overlay_groups.get('diagnostics')
        if layer is None:return
        for child in layer.childItems():self.scene().removeItem(child)
        layer.setVisible(self._diagnostics_visible)
        layout=getattr(self,'_diagnostic_layout',None)
        if not self._diagnostics_visible or layout is None:return
        from .annotation_preview import footprint
        for item in layout.items:
            polygons=item.mask_polygons or item.source_polygons
            color=QColor('#7c3aed') if item.mask_polygons else QColor('#d97706')
            for polygon in polygons:
                outline=QGraphicsPolygonItem(QPolygonF([self.source_to_scene(QPointF(x,y)) for x,y in polygon]),layer)
                pen=QPen(color,1.5);pen.setCosmetic(True);pen.setStyle(Qt.PenStyle.DashLine)
                outline.setPen(pen);outline.setToolTip('已验证遮挡范围' if item.mask_polygons else '未授权遮挡；保留源文')
            if item.lines:
                b=footprint(item.text_bounds,item.rotation_degrees)
                outline=QGraphicsPolygonItem(QPolygonF([self.source_to_scene(QPointF(x,y)) for x,y in
                    ((b.min_x,b.min_y),(b.max_x,b.min_y),(b.max_x,b.max_y),(b.min_x,b.max_y))]),layer)
                pen=QPen(QColor('#0891b2'),1);pen.setCosmetic(True);outline.setPen(pen)
                outline.setToolTip('中文实际文字范围')

    # 只控制OCR场景组
    # 中文和焦点层独立
    # 底图像素完全不变
    def set_ocr_visible(self,visible) -> None:
        if 'ocr' in self._overlay_groups:self._overlay_groups['ocr'].setVisible(visible)

    # 输入必须先还原到源坐标
    # 优先命中已绘中文
    # 旋转文字逆旋转后测矩形
    # 绕文字框中心旋转
    # 未绘中文不能用虚框抢点击
    # 其次测试原始多边形
    # OddEvenFill兼容多边形形状
    # 返回整个注释项供组关联
    # 不凭最近距离猜未命中目标
    def group_at_source(self,point: QPointF):
        layout=self._inspection_layout
        if layout is None:return None
        # Text first, then original polygons. Inspection is independent of mode.
        if self._translation_item is not None:
            for item in layout.items:
                if item.placement != 'inline' or not item.lines:continue
                b=item.text_bounds;cx=(b.min_x+b.max_x)/2;cy=(b.min_y+b.max_y)/2
                transform=QTransform();transform.translate(cx,cy);transform.rotate(-item.rotation_degrees);transform.translate(-cx,-cy)
                p=transform.map(point)
                if QRectF(b.min_x,b.min_y,b.max_x-b.min_x,b.max_y-b.min_y).contains(p):return item
        for item in layout.items:
            if any(QPolygonF([QPointF(x,y) for x,y in polygon]).containsPoint(point,Qt.FillRule.OddEvenFill)
                   for polygon in item.source_polygons):return item
        return None

    # 记录屏幕按下点区分拖动
    # 当前鼠标先mapToScene
    # 再scene_to_source还原
    # 点在中文框才启动拖动
    # 仅记录起点暂不修改布局
    # 其他位置保留手形平移行为
    def mousePressEvent(self,event) -> None:
        self._press_position=event.position()
        self._drag_annotation=None
        if self._has_page and event.button()==Qt.MouseButton.LeftButton:
            source=self.scene_to_source(self.mapToScene(event.position().toPoint()))
            item=self.group_at_source(source)
            if item is not None and self._translation_item is not None:
                b=item.text_bounds
                if QRectF(b.min_x,b.min_y,b.max_x-b.min_x,b.max_y-b.min_y).contains(source):
                    self._drag_annotation=(item,source)
                    event.accept(); return
        super().mousePressEvent(event)

    # 位移超过五逻辑像素算拖动
    # 拖动终点还原为源坐标
    # 发出的是新源框左上角
    # 主窗口统一安全验证与锁定
    # 短点击只发group_activated
    # 平移后不误触详情选择
    # 不会在鼠标事件直接写项目
    def mouseReleaseEvent(self,event) -> None:
        drag=getattr(self,'_drag_annotation',None)
        if drag is not None and event.button()==Qt.MouseButton.LeftButton:
            self._drag_annotation=None
            item,start=drag
            if self._press_position is not None and (event.position()-self._press_position).manhattanLength()>5:
                end=self.scene_to_source(self.mapToScene(event.position().toPoint()))
                self.annotation_moved.emit(item.group_id,item.text_bounds.min_x+end.x()-start.x(),item.text_bounds.min_y+end.y()-start.y())
            else:self.group_activated.emit(item.group_id)
            event.accept();return
        super().mouseReleaseEvent(event)
        if not self._has_page or event.button()!=Qt.MouseButton.LeftButton:return
        if self._press_position is not None and (event.position()-self._press_position).manhattanLength()>5:return
        item=self.group_at_source(self.scene_to_source(self.mapToScene(event.position().toPoint())))
        if item is not None:self.group_activated.emit(item.group_id)

    # 悬停命中复用源坐标逻辑
    # 提示显示英中与前三条告警
    # unavailable明确说明保留英文
    # 没有命中就清空旧提示
    def mouseMoveEvent(self,event) -> None:
        super().mouseMoveEvent(event)
        if not self._has_page:return
        item=self.group_at_source(self.scene_to_source(self.mapToScene(event.position().toPoint())))
        self.setToolTip('' if item is None else f'英文：{item.source_text}\n中文：{item.translated_text}\n'+
                        ('未成功翻译；保留英文' if item.placement=='unavailable' else '自动翻译预览')+
                        '\n'+ '\n'.join(item.warnings[:3]))

    def focus_ocr_region(self, region_id: str) -> None:
        self.focus_ocr_regions((region_id,))

    # 先恢复上次焦点框样式
    # 按原Region ID找全部成员
    # 共享框不会变成假字符框
    # 焦点轮廓置于独立高层
    # 蓝框与类别颜色分离
    # 联合包围范围用于居中
    # 仅移动相机不改变倍率
    # 缺失原始ID安全跳过
    def focus_ocr_regions(self, region_ids: tuple[str, ...]) -> None:
        """分组条目高亮其全部原始框，并定位到联合区域。"""
        for old_id in self._focused_ocr_ids:
            item = self._ocr_items.get(old_id)
            if item is None:
                continue
            pen = item.pen()
            pen.setColor(self._ocr_colors[old_id])
            pen.setWidthF(1.5)
            item.setPen(pen)
        self._focused_ocr_ids = region_ids
        focus=self._overlay_groups.get('focus')
        if focus is not None:
            for child in focus.childItems():self.scene().removeItem(child)
        bounds = QRectF()
        for region_id in region_ids:
            item = self._ocr_items.get(region_id)
            if item is None:
                continue
            pen = item.pen()
            pen.setColor(self._ocr_colors[region_id])
            pen.setWidthF(2.5)
            item.setPen(pen)
            bounds = bounds.united(item.sceneBoundingRect())
            if focus is not None:
                outline=QGraphicsPolygonItem(item.polygon(),focus)
                selected=QPen(QColor('#2563eb'),2);selected.setCosmetic(True);outline.setPen(selected)
        if not bounds.isNull():
            self.centerOn(bounds.center())

    # 源坐标乘初始场景/源尺寸
    # 水平和竖向分别换算
    # 不把视图zoom重复乘进去
    # 屏幕映射由QGraphicsView负责
    # 高清底图替换不改变此比例
    def source_to_scene(self, point: QPointF) -> QPointF:
        """原图像素或 PDF 点坐标映射到渲染场景坐标。"""
        if not self._has_page:
            raise ValueError("尚未加载图纸")
        return QPointF(point.x() * self._page_size.width() / self._source_size.width(),
                       point.y() * self._page_size.height() / self._source_size.height())

    # 与source_to_scene互为反变换
    # 先由Qt从屏幕得到场景点
    # 再还原PDF point或源像素
    # 保存拖动位置时使用该结果
    # 未加载页面不能换算
    def scene_to_source(self, point: QPointF) -> QPointF:
        if not self._has_page:
            raise ValueError("尚未加载图纸")
        return QPointF(point.x() * self._source_size.width() / self._page_size.width(),
                       point.y() * self._source_size.height() / self._page_size.height())

    # scene.clear会销毁旧图元
    # 随后清所有Python索引引用
    # 焦点/诊断/检查布局同时失效
    # 源尺寸和场景尺寸都清空
    # fit_mode退出避免无页适应
    # 视图矩阵恢复单位变换
    # 通知倍率恢复100%
    def clear_page(self) -> None:
        self.scene().clear()
        self._page_item = None
        self._overlay_groups.clear()
        self._ocr_items.clear()
        self._ocr_colors.clear()
        self._focused_ocr_ids = ()
        self._translation_item = None
        self._inspection_layout = None
        self._diagnostic_layout = None
        self._source_size = QSizeF()
        self._page_size = QSizeF()
        self._has_page = False
        self._fit_mode = False
        self.resetTransform()
        self.zoom_changed.emit(100.0)

    def zoom_in(self) -> None:
        self._zoom_by(1.25)

    def zoom_out(self) -> None:
        self._zoom_by(0.8)

    # 允许5%到1600%范围
    # 超过边界不应用缩放
    # 手动缩放退出自动适应
    # 只改视图矩阵
    # 倍率信号会触发高清预览防抖
    def _zoom_by(self, factor: float) -> None:
        if not self._has_page:
            return
        next_scale = self.transform().m11() * factor
        if 0.05 <= next_scale <= 16.0:
            self._fit_mode = False
            self.scale(factor, factor)
            self.zoom_changed.emit(self.zoom_percent)

    # 重置的是场景显示变换
    # 不重采样原图
    # 也不改PDF坐标单位
    # 随后窗口可按需补高清图
    def actual_size(self) -> None:
        if self._has_page:
            self._fit_mode = False
            self.resetTransform()
            self.zoom_changed.emit(self.zoom_percent)

    # 按可见内容而非纸外布局取景
    # 内容为空退回完整场景
    # 关闭滚动条避免尺寸反馈震荡
    # KeepAspectRatio保持图纸比例
    # fit_mode使窗口变化继续适应
    # 居中后广播实际倍率
    def fit_page(self) -> None:
        if self._has_page:
            self._fit_mode = True
            rect=self._content_rect if not self._content_rect.isEmpty() else self.sceneRect()
            # Remove scroll-bar feedback while fitInView measures the viewport;
            # this avoids alternating scales during resize and page transitions.
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(rect.center())
            self.zoom_changed.emit(self.zoom_percent)

    # Ctrl加滚轮才进行倍率缩放
    # 普通滚轮沿用滚动行为
    # 每步放大1.25或缩小0.8
    # 事件消费避免重复滚动
    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._zoom_by(1.25 if event.angleDelta().y() > 0 else 0.8)
            event.accept()
        else:
            super().wheelEvent(event)

    # 先让Qt更新视口尺寸
    # 仅fit_mode自动重新适应
    # 用户手动倍率不会被窗口改动覆盖
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_page()
