# mechanical_drawing_translator/main_window.py
"""主窗口布局及文档、导航与预览组件之间的协调。"""

import json
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSizeF, QTimer, Qt,QSignalBlocker
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter, QToolBar,
    QVBoxLayout, QWidget,QComboBox,
)

from .document_service import DocumentError, DocumentLoader, LoadedDocument
from .drawing_view import DrawingView
from .annotation_layout import AnnotationLayoutEngine
from .annotation_models import AnnotationLayout, PageGeometry, MaskDecision
from .export_dialog import ExportDialog
from .export_service import AnnotationExportService, ExportRequest, ExportResult
from .export_worker import ExportTask
from .mask_safety import analyze_mask_safety
from .models import TextRegion
from .ocr_refinement import RefinementReport
from .ocr_service import OcrDocument, OcrService
from .ocr_worker import OcrRequest, OcrSession, make_ocr_service
from .panels import FilePanel, TranslationPanel
from .preview_worker import PdfPreviewTask
from .translation_results import (
    STAGE1_PROVENANCE,
    TranslationResult,
    validate_translation_results,
    preview_translation_available,
)
from .translation_targets import DecisionStore, SelectionResult, TargetEntry, select_targets
from .translation_d8c import merge_reviewed_instructions
from .translation_glossary import Glossary, default_v1_glossary_path
from .translation_worker import (
    TranslationRequest,
    TranslationSession,
    make_translation_service,
)
from .product_workflow import ProductWorkflow


class MainWindow(ProductWorkflow, QMainWindow):
    """只负责界面交互；文件解析与 PDF 渲染由服务层处理。"""

    FILE_FILTER = "图纸文件 (*.pdf *.jpg *.jpeg *.png *.bmp);;PDF 文件 (*.pdf);;图片文件 (*.jpg *.jpeg *.png *.bmp)"

    # 窗口是各工作流协调者
    # ProductWorkflow提供项目与审核
    # RecoveryWorkflow提供本地草稿
    # 业务区域始终采用源坐标
    # UI控件只表达用户意图
    # 配置在窗口构造时固定
    # 运行身份绑定翻译知识组合
    # 缓存namespace与运行身份核对
    # OCR原始缓存和复查缓存分开
    # 目标缓存保存分组及SourceSpan
    # 译文缓存按组ID索引
    # 布局缓存还区分模式与遮挡
    # 预览工作底图另有独立缓存
    # canonical底图不跟缩放变化
    # 遮挡证据不等于OCR结果
    # 常驻Session在主线程创建
    # 各Worker迁入独立线程
    # 模型由工厂懒加载
    # 一窗口复用一套OCR会话
    # 一窗口复用一个翻译后端
    # _ocr_task只是不可变请求
    # 不是可被强杀的线程对象
    # 单次高清任务单独管理
    # 导出任务也有独立生命周期
    # closing阻止新任务和旧交付
    # auto_translate_key串起OCR后翻译
    # 旧分类只在显式注入时读取
    # 首个预览防抖定时120毫秒
    # 左右面板与画布由Splitter布局
    # 翻译列表选择可定位原框
    # 图纸点击反向选择翻译组
    # 最后创建项目与审核菜单
    # 控件创建完成再接恢复工作流
    def __init__(
        self,
        ocr_service_factory: Callable[[], OcrService] = make_ocr_service,
        translation_service_factory=make_translation_service,
        decision_store: DecisionStore | None = None,
        export_service_factory: Callable[[], AnnotationExportService] = AnnotationExportService,
        *, enable_semantic_grouping: bool = False,
        enable_ocr_refinement: bool = False,
        enable_d8c_semantic: bool = False,
        translation_runtime_identity: str = 'legacy',
        reviewed94_knowledge=None,
        enable_integrated_quality: bool = False,
        enable_professional_quality: bool = False,
        enable_convergence_quality: bool = False,
        convergence_glossary=None,
        enable_runtime_preload: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("MechanicalDrawingTranslator · 机械图纸翻译")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 650)
        self._document: LoadedDocument | None = None
        self._enable_semantic_grouping = enable_semantic_grouping
        self._enable_ocr_refinement = enable_ocr_refinement
        self._enable_d8c_semantic = enable_d8c_semantic
        self._translation_runtime_identity = translation_runtime_identity
        self._reviewed94_knowledge=reviewed94_knowledge
        self._enable_integrated_quality=enable_integrated_quality
        self._enable_professional_quality=enable_professional_quality
        self._enable_convergence_quality=enable_convergence_quality
        self._convergence_glossary=convergence_glossary
        self._enable_runtime_preload = enable_runtime_preload
        self._translation_cache_namespace = translation_runtime_identity
        self._d8c_glossary = (
            Glossary.load(default_v1_glossary_path()) if enable_d8c_semantic else None
        )
        self._refinement_cache: dict[tuple[str,int],RefinementReport] = {}
        self.current_page = 0
        self._ocr_session = OcrSession(ocr_service_factory, self)
        self._ocr_session.result_ready.connect(self._ocr_ready)
        self._ocr_session.failed.connect(self._ocr_failed)
        self._ocr_session.finished.connect(self._ocr_finished)
        self._ocr_task: OcrRequest | None = None
        self._translation_session = TranslationSession(translation_service_factory, self,
            runtime_identity=translation_runtime_identity)
        self._translation_session.result_ready.connect(self._translation_ready)
        self._translation_session.details_ready.connect(self._knowledge_ready)
        self._translation_session.failed.connect(self._translation_failed)
        self._translation_session.finished.connect(self._translation_finished)
        self._translation_session.preparation_finished.connect(self._translation_prepared)
        self._translation_task: TranslationRequest | None = None
        self._preview_task: PdfPreviewTask | None = None
        self._preview_pending = False
        self._ocr_cache: dict[tuple[str, int], tuple[TextRegion, ...]] = {}
        self._target_cache: dict[tuple[str, int], SelectionResult] = {}
        self._translation_cache: dict[tuple[str, int], dict[str, TranslationResult]] = {}
        self._translation_knowledge_cache = {}
        self._layout_cache: dict[tuple[str, int, bool], AnnotationLayout] = {}
        self._preview_source_cache = {}
        self._canonical_source_cache = {}
        self._preview_mask_cache = {}
        self._mask_task = None
        self._auto_translate_key = None
        self._use_legacy_decisions = decision_store is not None
        self._layout_engine = AnnotationLayoutEngine()
        self._export_service_factory = export_service_factory
        self._export_task: ExportTask | None = None
        self._decision_store = decision_store or DecisionStore()
        self._closing = False
        self._screen_fit_connected = False
        self.actions: dict[str, QAction] = {}
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._start_pdf_preview)

        self.file_panel = FilePanel()
        self.drawing_view = DrawingView()
        self.translation_panel = TranslationPanel()

        center = QWidget()
        center.setObjectName("previewPanel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(12, 12, 12, 12)
        center_layout.setSpacing(8)
        preview_title = QLabel("图纸预览")
        preview_title.setObjectName("panelHeading")
        center_layout.addWidget(preview_title)
        self.preview_hint = QLabel("打开 PDF 或图片后在此查看图纸 · Ctrl + 滚轮缩放 · 拖动平移")
        self.preview_hint.setObjectName("emptyState")
        center_layout.addWidget(self.preview_hint)
        center_layout.addWidget(self.drawing_view, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        for widget in (self.file_panel, center, self.translation_panel):
            splitter.addWidget(widget)
        splitter.setSizes([190, 900, 350])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._create_actions()
        self._set_view_actions_enabled(False)
        self._create_menu_and_toolbar()
        self._create_status_bar()
        self.file_panel.page_selected.connect(self.show_page)
        self.drawing_view.zoom_changed.connect(self._update_zoom)
        self.drawing_view.zoom_changed.connect(self._schedule_pdf_preview)
        self.translation_panel.region_selected.connect(self._focus_region)
        self.translation_panel.show_all_changed.connect(self._display_cached_regions)
        self.drawing_view.group_activated.connect(self._inspect_group)
        self._initialize_product_workflow()

    # 原生边框尺寸在显示后可信
    # screenChanged只连接一次
    # Qt屏幕尺寸已经是逻辑像素
    # 不能再乘或除DPR
    # 布局适屏延迟到事件循环
    # 不改变图纸源坐标
    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Native frame margins become reliable only after the window is shown.
        # Qt screen geometry is already logical: applying DPR a second time
        # would shrink a 200% desktop twice and invalidate source coordinates.
        handle = self.windowHandle()
        if handle is not None and not self._screen_fit_connected:
            handle.screenChanged.connect(self._schedule_window_fit)
            self._screen_fit_connected = True
        self._schedule_window_fit()

    # 单次排队让Qt先更新屏幕几何
    # 多屏切换复用同一适屏逻辑
    # 不在原生显示回调中递归缩放
    def _schedule_window_fit(self, *_args) -> None:
        QTimer.singleShot(0, self._fit_window_to_screen)

    # 最大化与全屏交给系统
    # 普通窗口约束在可用工作区
    # 留八个逻辑像素边距
    # 边框额外尺寸从客户区扣除
    # 小屏可降低通常最小尺寸
    # 避免标题栏和底部控件出屏
    # 移动量按frame与pos差计算
    # 仅调整窗口不缩放源图数据
    def _fit_window_to_screen(self) -> None:
        if not self.isVisible() or self.isMaximized() or self.isFullScreen():
            return
        screen = self.screen()
        if screen is None:
            return
        available = screen.availableGeometry().adjusted(8, 8, -8, -8)
        frame = self.frameGeometry()
        extra_width = max(0, frame.width() - self.width())
        extra_height = max(0, frame.height() - self.height())
        width = max(1, available.width() - extra_width)
        height = max(1, available.height() - extra_height)
        # Keep the usual minimum where space permits, but a logical work area
        # smaller than it must still expose the bottom controls and title bar.
        self.setMinimumSize(min(1000, width), min(650, height))
        self.resize(min(self.width(), width), min(self.height(), height))
        frame = self.frameGeometry()
        x = min(max(frame.x(), available.left()), available.right() - frame.width() + 1)
        y = min(max(frame.y(), available.top()), available.bottom() - frame.height() + 1)
        self.move(self.pos().x() + x - frame.x(), self.pos().y() + y - frame.y())

    # 统一字典维护菜单和工具栏动作
    # 一个动作可被多处控件引用
    # 打开Ctrl+O与退出Ctrl+Q
    # 缩放快捷键都只操作视图变换
    # 适应窗口为Ctrl+0
    # 实际大小为Ctrl+1
    # 软件设置仍是禁用占位
    # 识别当前页保留工具菜单入口
    # 自动翻译负责完整链路
    # 取消是软取消处理
    # 人工测试译文动作保留底层接口
    # 正常UI把该动作隐藏
    # 显示中文标注是可勾选动作
    # 预览刷新不重新触发翻译
    # 正式导出初始禁用
    def _create_actions(self) -> None:
        def make(key: str, title: str, slot=None, shortcut=None, enabled=True) -> QAction:
            action = QAction(title, self)
            if slot:
                action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.setEnabled(enabled)
            self.actions[key] = action
            return action

        make("open", "打开图纸", self.choose_document, "Ctrl+O")
        make("exit", "退出程序", self.close, "Ctrl+Q")
        make("zoom_in", "放大", self.drawing_view.zoom_in, "Ctrl++")
        make("zoom_out", "缩小", self.drawing_view.zoom_out, "Ctrl+-")
        make("fit", "适应窗口", self.drawing_view.fit_page, "Ctrl+0")
        make("actual", "实际大小", self.drawing_view.actual_size, "Ctrl+1")
        make("settings", "软件设置（后续阶段）", enabled=False)
        make("about", "关于软件", self._show_about)
        make("recognize", "识别当前页", self.recognize_current_page, enabled=False)
        make("load_translations", "载入人工测试译文…", self.choose_translation_results,
             enabled=False)
        make("translate", "自动翻译", self.translate_current_page, enabled=False)
        make('cancel','取消处理',self.cancel_automatic_translation,enabled=False)
        make("annotations", "显示中文标注", enabled=False)
        self.actions["annotations"].setCheckable(True)
        self.actions["annotations"].toggled.connect(self._refresh_annotation_preview)
        self.actions['load_translations'].setVisible(False)
        make("export", "导出标注副本…", self._choose_annotation_export, enabled=False)

    # 菜单区分文件/视图/工具/帮助
    # 工具栏提供打开/翻译/取消
    # 显示下拉只有两个正式模式
    # 默认索引一是中英对照
    # 旧chinese不再提供选择入口
    # 安全遮挡是独立复选框
    # 创建时不勾选
    # 仅中英对照允许生效
    # 未验证时英文与红字都保留
    # 遮挡状态标签显示资格计数
    # 导出仍走同一动作门禁
    # 放大缩小适应位于工具栏右侧
    # 工具栏固定不可拖走
    # 辅助名称描述语言显示方式
    def _create_menu_and_toolbar(self) -> None:
        menus = (
            ("文件", ("open", "exit")),
            ("视图", ("zoom_in", "zoom_out", "fit", "actual")),
            ("工具", ("recognize", "settings")),
            ("帮助", ("about",)),
        )
        for title, keys in menus:
            menu = self.menuBar().addMenu(title)
            for key in keys:
                menu.addAction(self.actions[key])
        toolbar = QToolBar("常用操作", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for key in ("open", "translate",'cancel'):
            toolbar.addAction(self.actions[key])
        toolbar.addSeparator()
        toolbar.addWidget(QLabel('显示：'))
        self.display_mode_selector=QComboBox()
        self.display_mode_selector.setObjectName('translationDisplayMode')
        self.display_mode_selector.addItems(['英文原文','中英对照'])
        self.display_mode_selector.setCurrentIndex(1)
        self.display_mode_selector.setAccessibleName('图纸语言显示方式')
        self.display_mode_selector.currentIndexChanged.connect(self._display_mode_changed)
        toolbar.addWidget(self.display_mode_selector)
        self.cover_checkbox = QCheckBox("安全遮挡英文")
        self.cover_checkbox.setToolTip('仅中英对照生效；只遮挡通过原生文字安全验证的英文，未通过时保留原文和红色中文。')
        self.cover_checkbox.setEnabled(False)
        self.cover_checkbox.toggled.connect(self._mask_option_changed)
        toolbar.addWidget(self.cover_checkbox)
        self.mask_status=QLabel('安全遮挡默认关闭')
        self.mask_status.setObjectName('maskSafetyStatus')
        self.mask_status.setWordWrap(True)
        self.mask_status.setMaximumWidth(360)
        toolbar.addWidget(self.mask_status)
        for key in ("export",):
            toolbar.addAction(self.actions[key])
        toolbar.addSeparator()
        for key in ("zoom_in", "zoom_out", "fit"):
            toolbar.addAction(self.actions[key])

    # 无图时禁用缩放动作
    # 仅控制视图相关按钮
    # 不决定OCR或导出资格
    def _set_view_actions_enabled(self, enabled: bool) -> None:
        for key in ("zoom_in", "zoom_out", "fit", "actual"):
            self.actions[key].setEnabled(enabled)

    # 主状态显示当前操作结果
    # 文件状态只显示基本名
    # 页状态采用一始显示
    # 倍率是视图变换百分比
    # 永久控件位于状态栏右侧
    def _create_status_bar(self) -> None:
        self.state_status = QLabel("就绪")
        self.file_status = QLabel("未打开文件")
        self.page_status = QLabel("— / —")
        self.zoom_status = QLabel("100%")
        bar = self.statusBar()
        bar.addWidget(self.state_status, 1)
        bar.addPermanentWidget(self.file_status)
        bar.addPermanentWidget(self.page_status)
        bar.addPermanentWidget(self.zoom_status)

    # 用户选择路径后先保护修改
    # 恢复草稿失败不能默认丢编辑
    # 放弃需显式选择Discard
    # 打开成功才询问恢复草稿
    # 程序化载入不弹该选择
    # 解码失败保留旧文档
    # 已放弃但替换失败需重标dirty
    # 不能让旧编辑失去关闭保护
    def choose_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开图纸", "", self.FILE_FILTER)
        if path:
            discarded_live = False
            if self._project_dirty and not self.flush_recovery_draft():
                choice = QMessageBox.question(self, '未保存的修改',
                    '恢复草稿无法写入。可取消打开并另存项目；是否明确放弃本次未保存修改？',
                    QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel)
                if choice != QMessageBox.StandardButton.Discard:
                    return
                self.discard_unsaved_live_changes()
                discarded_live = True
            if self.open_document(path):
                # Only an explicit UI open prompts. Programmatic project load
                # and recovery replay remain nonmodal and preserve old drafts.
                self.choose_recovery_draft()
            elif discarded_live:
                # A failed decode leaves the old document visible. Its edits
                # must retain close/recovery protection until replacement
                # actually succeeds, even after the preceding discard choice.
                self.mark_project_dirty('failed_source_switch')

    # 先打开并渲染替代文档
    # 首帧成功才进入替换阶段
    # 失败关闭临时句柄
    # 当前源和预览保持可用
    # dirty草稿未安全落盘就拒绝换图
    # 记录新源实际SHA
    # 停止旧批次并清人工布局
    # 结果身份回到当前处理配置
    # 软取消旧管线但不强杀模型
    # 停止旧预览防抖
    # 所有文档级结果缓存清空
    # 新canonical只留首帧
    # 遮挡开关重新默认关闭
    # 重新显示原底图与源尺寸
    # 文件面板恢复第一页
    # 旧文档句柄最后释放
    # 常驻OCR/翻译Session继续复用
    # 新文档ID隔离旧任务回调
    # 重置恢复状态并检测旧草稿
    # 返回True只表示成功打开
    def open_document(self, path: str | Path, show_error: bool = True) -> bool:
        """成功加载并渲染首张页面后才替换当前图纸。"""
        replacement = None
        try:
            replacement = DocumentLoader.open(path)
            first_image = replacement.render_page(0)
            source_size = replacement.source_size(0)
        except (DocumentError, OSError) as error:
            if replacement:
                replacement.close()
            self.state_status.setText("打开失败")
            if show_error:
                QMessageBox.warning(self, "无法打开图纸", str(error))
            return False
        previous = self._document
        if (previous is not None and not self._loading_project
                and self._project_dirty and not self.flush_recovery_draft()):
            replacement.close()
            self.state_status.setText('恢复草稿无法写入；请先保存或明确放弃本次未保存修改，再打开其它图纸')
            return False
        from .project_store import source_sha256
        self._source_identity_sha = source_sha256(replacement.info.path)
        self._batch_active = False
        self._manual_layouts = {}
        self._legacy_display_state = None
        self._review_events = []
        self._machine_suggestions = {}
        self._result_runtime_identity=self._translation_runtime_identity
        self._page_result_runtime={}
        self._cancel_pipeline()
        self._preview_timer.stop()
        self._preview_pending = False
        self._document = replacement
        self.current_page = 0
        self._ocr_cache.clear()
        self._refinement_cache.clear()
        self._target_cache.clear()
        self._translation_cache.clear()
        self._translation_knowledge_cache.clear()
        self._layout_cache.clear()
        self._preview_source_cache.clear()
        self._canonical_source_cache = {0:first_image}
        self._preview_mask_cache.clear()
        self.actions["annotations"].setChecked(False)
        self.actions["annotations"].setEnabled(False)
        self.actions["load_translations"].setEnabled(False)
        self.actions["export"].setEnabled(False)
        self.cover_checkbox.setChecked(False)
        self.cover_checkbox.setEnabled(False)
        self.drawing_view.show_page(first_image, source_size)
        self._set_view_actions_enabled(True)
        self.actions["recognize"].setEnabled(self._ocr_task is None)
        self.file_panel.set_document(replacement.info)
        self.translation_panel.set_entries(())
        self._update_annotation_actions()
        self.preview_hint.hide()
        self.file_status.setText(replacement.info.name)
        self.page_status.setText(f"1 / {replacement.info.page_count}")
        self.state_status.setText(self._busy_status() if self._ocr_task else "图纸已加载")
        if previous:
            previous.close()
        self.reset_project_recovery_state()
        return True

    # 校验有效页码后才切换
    # 同页调用无需重新渲染
    # 用户手动切页终止批次剩余队列
    # 批次内部切页带专用标志
    # 先软取消旧自动翻译意图
    # 失败时恢复导航选择
    # 新底图保留当前源尺寸
    # canonical记初始渲染而非放大图
    # current_page成功后才更新
    # 切页状态进入项目草稿
    # 已有目标/译文立即从缓存显示
    # 刷新布局不重新推理
    # 旧任务完成仍需身份门禁
    def show_page(self, page_index: int) -> None:
        if self._document is None or not 0 <= page_index < self._document.info.page_count:
            return
        if page_index == self.current_page:
            return
        if not getattr(self, '_batch_switch', False):
            self._batch_active=False
            self._batch_queue=[]
        self._cancel_pipeline()
        try:
            image = self._document.render_page(page_index)
            source_size = self._document.source_size(page_index)
        except DocumentError as error:
            self.state_status.setText("页面加载失败")
            QMessageBox.warning(self, "无法显示页面", str(error))
            self.file_panel.set_current_page(self.current_page)
            return
        self._remember_canonical_source(page_index,image)
        self.drawing_view.show_page(image, source_size)
        self.current_page = page_index
        self.mark_project_dirty('page')
        self.file_panel.set_current_page(page_index)
        self.page_status.setText(f"{page_index + 1} / {self._document.info.page_count}")
        self.state_status.setText(self._busy_status() if self._ocr_task else "图纸已加载")
        self._display_cached_regions()
        self._refresh_annotation_preview()

    # 导出优先显示
    # 其次显示翻译页
    # OCR请求页号来自任务快照
    # 旧图仍在识别需明确提示
    # 不能把当前页号套给旧任务
    def _busy_status(self) -> str:
        if self._export_task is not None:
            return "正在后台导出中文标注副本…"
        if self._translation_task is not None:
            return f"正在翻译第 {self._translation_task.page_index + 1} 页…"
        task = self._ocr_task
        if task is None or self._document is None:
            return "图纸已加载"
        if task.document_id != self._document.info.id:
            return "上一图纸仍在识别，请等待完成…"
        return f"正在识别第 {task.page_index + 1} 页…"

    # 同一时刻只提交一个OCR
    # 导出/遮挡忙也不并发开识别
    # closing后禁止新任务
    # 请求固定源路径/页/文档ID
    # 提交后立即禁用识别动作
    # 模型初始化仍在Worker完成
    # 按钮状态按真实忙碌重算
    def recognize_current_page(self) -> None:
        if (self._document is None or self._ocr_task is not None
                or self._export_task is not None or self._closing or self._mask_task is not None):
            return
        info = self._document.info
        page_index = self.current_page
        task = self._ocr_session.submit(info.path, page_index, info.id)
        self._ocr_task = task
        self.actions["recognize"].setEnabled(False)
        self.state_status.setText(f"正在识别第 {page_index + 1} 页…")
        self._update_annotation_actions()

    # 先核对活动请求对象
    # 再核对文档ID与关闭状态
    # 旧图OCR不能写进新图缓存
    # regions保存未经覆盖的原始结果
    # 局部复查报告独立保存
    # 旧分类仅显式注入时读取
    # 判定读取失败继续保留OCR
    # 从有效区域重新筛选目标
    # 对应旧译文全部失效
    # 旧知识追踪和布局也失效
    # 预览源与遮挡证据一并清理
    # 保存为dirty等待草稿写
    # 只有当前页更新可见列表
    # 状态数量按区域分类口径
    # 该数量不代表句级准确率
    def _ocr_ready(self, request: OcrRequest, result: OcrDocument) -> None:
        if request != self._ocr_task:
            return
        if self._closing or self._document is None or result.document_id != self._document.info.id:
            return
        warning = ""
        for page in result.pages:
            key = (result.document_id, page.page_index)
            self._ocr_cache[key] = page.regions
            if page.refinement is not None:
                self._refinement_cache[key] = page.refinement
            else:
                self._refinement_cache.pop(key,None)
            try:
                overrides = self._decision_store.overrides_for(
                    self._document.info.path, page.regions,
                ) if self._use_legacy_decisions else {}
            except (OSError, ValueError) as exc:
                overrides = {}
                warning = f"；人工判定读取失败：{exc}"
            self._target_cache[key] = self._select_page_targets(key,overrides)
            self._translation_cache.pop(key, None)
            self._translation_knowledge_cache.pop(key, None)
            self._layout_cache.pop((key[0], key[1], False), None)
            self._layout_cache.pop((key[0], key[1], True), None)
            self._invalidate_preview_page(key)
        self.mark_project_dirty('ocr')
        if any(page.page_index == self.current_page for page in result.pages):
            self._display_cached_regions()
            self._update_annotation_actions()
            selection = self._target_cache[(result.document_id, self.current_page)]
            self.state_status.setText(
                f"第 {self.current_page + 1} 页：原始 {len(selection.raw_regions)}"
                f" · 需翻译 {selection.counts['translatable']}"
                f" · 自动尝试 {selection.counts['review_required']}"
                f" · 排除 {selection.counts['non_translatable']}{warning}"
            )

    # 失败也必须属于当前请求
    # 旧图旧页异常不覆盖当前状态
    # 界面给出可见失败提示
    # 不凭空创建空成功缓存
    def _ocr_failed(self, request: OcrRequest, message: str) -> None:
        if (not self._closing and self._document is not None
                and request == self._ocr_task
                and self._document.info.id == request.document_id
                and self.current_page == request.page_index):
            self.state_status.setText(f"识别失败：{message}")
            self.translation_panel.hint_label.setText(f"识别失败：{message}")
            self.translation_panel.hint_label.show()

    # 只有匹配请求才清忙碌
    # 关闭等待在此继续
    # 按钮启用依赖其他任务状态
    # auto_translate_key代表未取消意图
    # 文档与当前页都一致才续翻译
    # 目标存在才提交下个阶段
    # 空或失败页通知批次记账
    # 旧图结束不推进新图翻译
    # 不能在result_ready抢先启下一OCR
    # finished才是资源安全结束点
    def _ocr_finished(self, request: OcrRequest) -> None:
        task = self._ocr_task
        if task != request:
            return
        self._ocr_task = None
        if self._closing:
            self.close()
        else:
            self.actions["recognize"].setEnabled(
                self._document is not None and self._export_task is None,
            )
            self._update_annotation_actions()
            expected=self._auto_translate_key
            if (expected is not None and self._document is not None
                    and expected==(self._document.info.id,self.current_page)):
                self._auto_translate_key=None
                if expected in self._target_cache:
                    self.translate_current_page()
                if self._batch_active and self._translation_task is None:
                    self._batch_page_finished(request.page_index,
                        failed=expected not in self._target_cache)
                return
            if (task is not None and self._document is not None
                    and (task.document_id != self._document.info.id
                         or task.page_index != self.current_page)):
                self.state_status.setText("图纸已加载")

    # 运行身份不匹配整层缓存失效
    # 译文与知识追踪一起清理
    # 布局/底图/遮挡也需重建
    # OCR原始缓存不由此删除
    # 当前调用返回False提示先停止
    # 不能拿旧知识结果冒充新配置
    def _ensure_translation_cache_identity(self) -> bool:
        if self._translation_cache_namespace == self._translation_runtime_identity:
            return True
        self._translation_cache.clear()
        self._translation_knowledge_cache.clear()
        self._layout_cache.clear()
        self._preview_source_cache.clear()
        self._preview_mask_cache.clear()
        self._translation_cache_namespace = self._translation_runtime_identity
        return False

    # 显示路径先验缓存namespace
    # 没有目标时清列表和图层
    # 全部OCR与候选视图可切换
    # 全部视图可能用原区域ID
    # 候选视图使用语义组ID
    # 译文按组关联而非按行号
    # 布局缓存区分preview标记
    # 复查和知识证据交详情面板
    # 原始框仍来自SelectionResult
    # 分类颜色只表达筛选结果
    # 英文模式通常隐藏OCR框
    # 显式全部OCR仍可诊断显示
    # 刷新不创建新的推理任务
    def _display_cached_regions(self) -> None:
        self._ensure_translation_cache_identity()
        if self._document is None:
            return
        key = (self._document.info.id, self.current_page)
        selection = self._target_cache.get(key)
        if selection is None:
            self.translation_panel.set_entries(())
            self.drawing_view.show_ocr_regions(())
            self.drawing_view.show_translation_layout(None)
            self.drawing_view.set_inspection_layout(None)
            self._update_annotation_actions()
            return
        show_all = self.translation_panel.show_all_checkbox.isChecked()
        entries = selection.all_entries if show_all else selection.candidate_entries
        translations = self._translation_cache.get(key, {})
        if not translations:
            self.drawing_view.show_translation_layout(None)
            self.drawing_view.set_inspection_layout(None)
        layout = None
        if self.actions["annotations"].isChecked():
            mode=self._display_mode_name();mask=mode=='bilingual' and self.cover_checkbox.isChecked()
            layout = self._layout_cache.get((key[0],key[1],mask,mode,'preview')) or self._layout_cache.get((key[0],key[1],mask,mode))
        self.translation_panel.set_entries(entries, selection, translations, layout,
                                          refinement=self._refinement_cache.get(key),
                                          knowledge_details=self._translation_knowledge_cache.get(key))
        self.drawing_view.show_ocr_regions(
            selection.raw_regions, selection.category_by_region, show_all=show_all,
        )
        self.drawing_view.set_ocr_visible(show_all or self._display_mode_name()=='bilingual')
        self._update_annotation_actions()

    # 这是底层兼容控制接口
    # 正常工具栏没有独立开关
    # 先重建所有已缓存页目标
    # 重建成功后再切全局标志
    # 旧翻译请求软取消
    # 新分组不复用旧译文
    # 知识追踪与布局全部清除
    # 原始OCR保持不变
    # 关闭标注避免残留旧句界
    # 随后按新目标刷新列表
    def set_semantic_grouping(self, enabled: bool) -> None:
        """Rebuild targets atomically; old machine results cannot cross modes."""
        enabled = bool(enabled)
        if enabled == self._enable_semantic_grouping:
            return
        rebuilt = {}
        if self._document is not None:
            for key, raw in self._ocr_cache.items():
                overrides = self._decision_store.overrides_for(self._document.info.path, raw) if self._use_legacy_decisions else {}
                rebuilt[key] = self._select_page_targets(key,overrides,semantic=enabled)
        self._enable_semantic_grouping = enabled
        if self._translation_task is not None:
            self._translation_session.cancel_active()
        self._target_cache = rebuilt
        self._translation_cache.clear()
        self._translation_knowledge_cache.clear()
        self._layout_cache.clear()
        self._preview_source_cache.clear()
        self._preview_mask_cache.clear()
        self.actions["annotations"].setChecked(False)
        self.drawing_view.show_translation_layout(None)
        self._display_cached_regions()
        self.state_status.setText("OCR-3B语义句界试运行已启用" if enabled else "已恢复原分组方式")

    # 原始缓存始终可回滚
    # 启用时选择refinement有效区域
    # 旧人工分类经parent映射转给新ID
    # 语义开关决定句界恢复
    # integrated/professional还受语义总开关约束
    # D8C可按安全词汇组合指令
    # 合并后保留每个源候选关联
    # all_entries不能只按region_id替换
    # 不同SourceSpan可能共享原框
    # 去重应按group_id
    # 被排除原始行继续保留
    # reviewed94用实际源路径限定实例
    # 可保留聚合前semantic_units
    # convergence再做受控完整知识恢复
    # 最后返回新的SelectionResult
    # OCR原始字面不在此覆盖
    def _select_page_targets(self,key,overrides,*,semantic=None,refined=None):
        raw=self._ocr_cache[key]
        enabled=self._enable_ocr_refinement if refined is None else refined
        report=self._refinement_cache.get(key) if enabled else None
        selected=report.selected_regions if report is not None else raw
        decisions=({r.id:overrides[parent] for r in selected
                    if (parent:=report.parent_by_region_id[r.id]) in overrides}
                   if report is not None else overrides)
        semantic_enabled=self._enable_semantic_grouping if semantic is None else semantic
        selection = select_targets(selected,decisions,enable_semantic_grouping=semantic_enabled,
            enable_integrated_quality=self._enable_integrated_quality and semantic_enabled,
            enable_professional_quality=getattr(self,'_enable_professional_quality',False) and semantic_enabled)
        if self._enable_d8c_semantic and self._d8c_glossary is not None:
            merged, evidence = merge_reviewed_instructions(
                selection.candidate_entries, self._d8c_glossary,
            )
            all_entries = selection.all_entries
            if evidence:
                sources = {entry.group_id: entry for entry in selection.candidate_entries}
                replacements = {}
                affected_regions = set()
                for item in evidence:
                    action = sources[item.action_group_id]
                    obj = sources[item.object_group_id]
                    region_ids = (*action.region_ids, *obj.region_ids)
                    replacement = next(entry for entry in merged if entry.region_ids == region_ids)
                    replacements[action.group_id] = replacement
                    replacements[obj.group_id] = replacement
                    affected_regions.update(region_ids)
                # The all-OCR view normally has one row per raw region. Expand
                # affected rows back to source candidates before replacing by
                # group ID: distinct semantic spans can share the same region.
                # Preserve excluded raw rows and the original raw-row order.
                display_entries = []
                emitted = set()
                for original in all_entries:
                    if (original.category == 'non_translatable'
                            or not affected_regions.intersection(original.region_ids)):
                        display_entries.append(original)
                        continue
                    for source in selection.candidate_entries:
                        if not set(original.region_ids).intersection(source.region_ids):
                            continue
                        displayed = replacements.get(source.group_id, source)
                        if displayed.group_id not in emitted:
                            display_entries.append(displayed)
                            emitted.add(displayed.group_id)
                all_entries = tuple(display_entries)
            selection = SelectionResult(
                selection.raw_regions, selection.decisions, merged,
                all_entries, selection.counts,
            )
        if self._reviewed94_knowledge is not None and self._document is not None:
            selection=self._reviewed94_knowledge.recover_selection(selection,self._document.info.path,
                preserve_semantic_units=self._enable_integrated_quality)
        if getattr(self,'_enable_convergence_quality',False) and getattr(self,'_convergence_glossary',None) is not None:
            from .translation_convergence import recover_reviewed_selection
            selection=recover_reviewed_selection(selection,self._convergence_glossary)
        return selection

    # 切换只使用已缓存复查候选
    # 不会在此重新运行局部OCR
    # 先按新有效区域重建目标
    # parent身份供人工分类继承
    # 软取消旧翻译结果交付
    # 原/新SourceSpan不能混用
    # 所有译文与布局需重新生成
    # 原始OCR缓存继续保留
    # 关闭标注后显示当前有效输入
    def set_ocr_refinement(self,enabled: bool) -> None:
        """Apply or roll back cached image-backed candidates, never old spans."""
        enabled=bool(enabled)
        if enabled==self._enable_ocr_refinement: return
        rebuilt={}
        if self._document is not None:
            for key,raw in self._ocr_cache.items():
                overrides=self._decision_store.overrides_for(self._document.info.path,raw) if self._use_legacy_decisions else {}
                rebuilt[key]=self._select_page_targets(key,overrides,refined=enabled)
        self._enable_ocr_refinement=enabled
        self._translation_session.cancel_active()
        self._target_cache=rebuilt
        self._translation_cache.clear();self._translation_knowledge_cache.clear();self._layout_cache.clear()
        self._preview_source_cache.clear()
        self._preview_mask_cache.clear()
        self.actions["annotations"].setChecked(False)
        self.drawing_view.show_translation_layout(None)
        self._display_cached_regions()
        self.state_status.setText("OCR-3C候选已启用" if enabled else "已恢复原始OCR输入")

    # 一次点击表示完整翻译意图
    # 先确认运行身份未变化
    # 任一互斥任务忙时不重复提交
    # 已有非空译文直接显示缓存
    # 不会重复调用模型
    # 没有目标时先记自动意图
    # 可选prepare与OCR分线程重叠
    # OCR完成再真正提交翻译
    # 无候选明确保留原图
    # 提交带当前目标完整元组
    # pdf_path提供审核实例上下文
    # UI保持可打开其他图纸
    # 打开时旧请求会软取消
    # 状态栏显示本次请求页
    def translate_current_page(self) -> None:
        """One user intent chains missing OCR, translation and source-coordinate preview."""
        if not self._ensure_translation_cache_identity():
            return
        if (self._document is None or self._translation_task is not None
                or self._ocr_task is not None or self._export_task is not None
                or self._closing or self._mask_task is not None):
            return
        key = (self._document.info.id, self.current_page)
        if self._translation_cache.get(key):
            self._show_chinese_preview()
            self.state_status.setText('已显示当前页缓存译文；未重复识别或翻译')
            return
        selection = self._target_cache.get(key)
        if selection is None:
            self._auto_translate_key=key
            if self._enable_runtime_preload:
                self._translation_session.prepare()
            self.recognize_current_page()
            return
        if not selection.candidate_entries:
            self.state_status.setText('当前页未检测到可翻译英文；保留原图')
            return
        try:
            request = self._translation_session.submit(
                key[0], key[1], selection.candidate_entries,
                pdf_path=str(self._document.info.path),
            )
        except (RuntimeError, ValueError) as exc:
            self.state_status.setText(f"无法开始翻译：{exc}")
            return
        self._translation_task = request
        self.actions["open"].setEnabled(True)
        self.actions["recognize"].setEnabled(False)
        self._update_annotation_actions()
        self.state_status.setText(f"正在翻译第 {self.current_page + 1} 页…")

    # 知识细节也要校验活动请求
    # 运行身份/文档/页均须一致
    # 目标元组变化则丢弃
    # 仅接受machine结果细节
    # 不把知识命中标成人工批准
    # 追踪按组存入当前页缓存
    def _knowledge_ready(self, request, details) -> None:
        if (request != self._translation_task or self._closing or self._document is None
                or request.runtime_identity != self._translation_runtime_identity
                or request.document_id != self._document.info.id or request.page_index != self.current_page):
            return
        selection = self._target_cache.get((request.document_id, request.page_index))
        if selection is None or tuple(selection.candidate_entries) != request.entries:
            return
        validate_translation_results(request.entries, (item.result for item in details),
                                     allowed_provenance=frozenset({'machine'}))
        self._translation_knowledge_cache[(request.document_id, request.page_index)] = {
            item.result.group_id: item.detail.knowledge_trace for item in details}

    # 先拒绝不同运行身份
    # 再拒绝旧请求与已关闭窗口
    # 文档和当前页双重隔离
    # 目标完整元组还必须未变化
    # 相同页码不足以证明结果有效
    # 验证group/region/source身份
    # Worker只可交machine来源
    # 缺条目补显式不可用回退
    # 不能把缺结果静默删除
    # 回退仍保留原英文
    # 记录该页实际运行身份
    # 结果变化使全部预览派生缓存失效
    # 刷新列表后显示中文预览
    # 当前语言选择不被强行切换
    # 保留原文和回退分别计数
    # 红字出现不等于人工审核
    # 状态提示布局可能仍待调整
    def _translation_ready(
        self, request: TranslationRequest, results: tuple[TranslationResult, ...],
    ) -> None:
        if request.runtime_identity != self._translation_runtime_identity:
            self._ensure_translation_cache_identity()
            return
        if request != self._translation_task or self._closing or self._document is None:
            return
        if (request.document_id != self._document.info.id
                or request.page_index != self.current_page):
            return
        current_selection = self._target_cache.get(
            (request.document_id, request.page_index),
        )
        if (current_selection is None
                or tuple(current_selection.candidate_entries) != request.entries):
            self.state_status.setText("翻译目标已变更，已忽略过期机器译文")
            return
        try:
            validated = validate_translation_results(
                request.entries, results, allowed_provenance=frozenset({"machine"}),
            )
        except ValueError as exc:
            self.state_status.setText(f"机器译文校验失败：{exc}")
            return
        key = (request.document_id, request.page_index)
        for entry in request.entries:
            if entry.group_id not in validated:
                validated[entry.group_id]=TranslationResult(1,entry.document_id,entry.page_index,
                    entry.group_id,entry.region_ids,entry.source_text,entry.source_text,'machine',
                    ('translation_unavailable','translation_result_missing'))
        self._translation_cache[key] = validated
        self._page_result_runtime[str(request.page_index)]=self._translation_runtime_identity
        self.mark_project_dirty('translation')
        self._layout_cache.pop((key[0], key[1], False), None)
        self._layout_cache.pop((key[0], key[1], True), None)
        self._invalidate_preview_page(key)
        self._display_cached_regions()
        self._show_chinese_preview()
        preserved_count=sum('reviewed94:no_translation_required' in r.warnings for r in validated.values())
        fallback_count=sum(not preview_translation_available(r) and 'reviewed94:no_translation_required' not in r.warnings for r in validated.values())
        self.state_status.setText(
            f"自动翻译完成：{len(validated)-fallback_count-preserved_count} 条中文，{preserved_count} 条保留原文，{fallback_count} 条未成功翻译；"
            "译文已保存到列表；图面显示以当前模式和布局状态为准，点击区域查看详情"
        )

    # 仅报告当前页当前请求异常
    # 不让旧任务失败覆盖新图状态
    # 没有成功缓存就不伪报完成
    def _translation_failed(self, request: TranslationRequest, message: str) -> None:
        if (not self._closing and request == self._translation_task
                and self._document is not None
                and request.document_id == self._document.info.id
                and request.page_index == self.current_page):
            self.state_status.setText(f"翻译失败：{message}")

    # 结束点才清活动翻译引用
    # 关闭等待在此重新尝试
    # 识别按钮仍检查OCR/导出状态
    # 批处理按是否有缓存判失败
    # 每页完成后排队继续
    # 不由成功信号直接递归推进
    def _translation_finished(self, request: TranslationRequest) -> None:
        if request != self._translation_task:
            return
        self._translation_task = None
        if self._closing:
            self.close()
            return
        self.actions["open"].setEnabled(True)
        self.actions["recognize"].setEnabled(
            self._document is not None and self._ocr_task is None
            and self._export_task is None,
        )
        self._update_annotation_actions()
        self._batch_page_finished(request.page_index,
            failed=(request.document_id, request.page_index) not in self._translation_cache)

    # prepare没有页结果
    # 不能在此清除翻译请求
    # OCR完成才触发自动送译
    # 准备错误由正式请求重试报告
    # 这里只负责唤醒延迟关闭
    def _translation_prepared(self, runtime_identity: str, message: str) -> None:
        """Preparation has no page output; only unblock a deferred window close.

        OCR completion remains the sole automatic translation trigger. A load
        error is retried/reported by the subsequent normal translation request.
        """
        if self._closing:
            self.close()

    # 遗留人工测试接口仍可程序调用
    # 正常UI动作已隐藏禁用
    # JSON只用于明确测试来源
    # 不是日常自动翻译入口
    def choose_translation_results(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "载入人工测试译文", "", "翻译结果 JSON (*.json)",
        )
        if path:
            self.load_translation_results(path)

    # 完整读入后一次性替换缓存
    # 只接受manual_test专用kind
    # schema_version限定一
    # 根对象和results数组都验证
    # 结果必须属于当前文档UUID
    # 每页须已有对应OCR目标
    # 按源身份逐项验证
    # 任一页失败保留现有预览
    # 成功后清旧知识追踪
    # 所有受影响页派生布局失效
    # 人工测试来源不等同machine
    # 也不能用此给真实机器批量批准
    def load_translation_results(self, path: str | Path, show_error: bool = True) -> bool:
        """原子载入并严格校验人工译文；失败时保留现有预览。"""
        try:
            if self._document is None:
                raise ValueError("请先打开对应图纸")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("翻译结果根节点必须是对象")
            if payload.get("schema_version") != 1:
                raise ValueError("不支持的翻译结果 schema_version")
            if payload.get("kind") != "manual_test_translation_results":
                raise ValueError("当前阶段只接受明确标记的人工测试译文")
            raw_results = payload.get("results")
            if not isinstance(raw_results, list) or not raw_results:
                raise ValueError("翻译结果 results 必须是非空数组")
            results = [TranslationResult.from_dict(item) for item in raw_results]
            grouped: dict[tuple[str, int], list[TranslationResult]] = {}
            document_id = self._document.info.id
            for result in results:
                if result.document_id != document_id:
                    raise ValueError("译文 document_id 与当前图纸不一致")
                grouped.setdefault((document_id, result.page_index), []).append(result)
            validated_pages: dict[tuple[str, int], dict[str, TranslationResult]] = {}
            for key, page_results in grouped.items():
                selection = self._target_cache.get(key)
                if selection is None:
                    raise ValueError(f"第 {key[1] + 1} 页尚无对应 OCR 翻译目标")
                validated_pages[key] = validate_translation_results(
                    selection.candidate_entries, page_results,
                )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.state_status.setText(f"测试译文载入失败：{exc}")
            if show_error:
                QMessageBox.warning(self, "无法载入人工测试译文", str(exc))
            return False

        self._translation_cache.update(validated_pages)
        for page_key in validated_pages:
            self._translation_knowledge_cache.pop(page_key, None)
        for document_id, page_index in validated_pages:
            self._layout_cache.pop((document_id, page_index, False), None)
            self._layout_cache.pop((document_id, page_index, True), None)
            self._invalidate_preview_page((document_id,page_index))
        self.actions["annotations"].setEnabled(True)
        self.actions["annotations"].setChecked(True)
        self._refresh_annotation_preview()
        self.state_status.setText(f"已载入 {len(results)} 条人工测试译文（非机器翻译）")
        return True

    # PDF使用源页面point尺寸
    # 图片使用定向后源像素尺寸
    # 不读屏幕pixmap尺寸
    # 文档与页号都进入几何契约
    # 预览导出共享这套坐标
    def _page_geometry(self, page_index: int) -> PageGeometry:
        if self._document is None:
            raise ValueError("尚未打开图纸")
        size: QSizeF = self._document.source_size(page_index)
        unit = "pdf_point" if self._document.info.file_type == "PDF" else "pixel"
        return PageGeometry(
            self._document.info.id, page_index, size.width(), size.height(), unit,
        )

    # 索引只映射两个正式模式
    # 无第三个中文原位入口
    # 历史chinese只在恢复兼容层处理
    def _display_mode_name(self):
        return ('english','bilingual')[self.display_mode_selector.currentIndex()]

    # 只缓存初始标准渲染
    # 高清缩放图不能混入
    # 按页索引隶属当前文档
    # 恢复UUID变化不影响该底图
    # 采用插入顺序保留最近三页
    # 控制大幅面工作图内存
    def _remember_canonical_source(self,page_index,image):
        # Page indices belong to the currently opened immutable document. A
        # project restore may rebind its UUID, so this cache is document-scoped
        # rather than keyed by the temporary open_document UUID. Keep three
        # bounded initial renders; view zoom/high-DPI rasters never enter it.
        self._canonical_source_cache.pop(page_index,None)
        self._canonical_source_cache[page_index]=image
        while len(self._canonical_source_cache)>3:
            self._canonical_source_cache.pop(next(iter(self._canonical_source_cache)))

    # 所有布局从源坐标生成
    # 旧chinese调用降为双语
    # 该兼容调用不继承遮挡
    # 遮挡只在bilingual生效
    # 缓存键同时区分模式与preview
    # 布局算法版本变化立即失效
    # 缓存命中也可补请求背景证据
    # 遮挡证据只在OCR空闲后提交
    # 源底图取canonical标准渲染
    # 不使用随缩放变化的高清图
    # 背景和verified集合分别传递
    # 补充蒙版字段不自动授权擦除
    # 人工位置作为固定障碍
    # 自动布局始终先不请求遮挡
    # 中文位置独立于遮挡是否通过
    # 套人工布局后再加双语mask
    # 因此遮挡失败不隐藏红字
    # preview可显示machine结果
    # 正式分支先筛当前可译英文
    # 非审核来源一律拒绝
    # 逐项结果身份仍再校验
    # 正式布局复用预览几何
    # 没有可绘中文拒绝导出
    # 待布局/detail/unavailable都阻止正式输出
    # 只有通过门禁才清preview_only
    # 缓存旧SAFE不能替当前背景验证
    def _layout_for_page(self, page_index: int, mask_requested: bool, *, for_preview=False,display_mode=None) -> AnnotationLayout:
        self._ensure_translation_cache_identity()
        if self._document is None:
            raise ValueError("尚未打开图纸")
        document_id = self._document.info.id
        mode=display_mode or self._display_mode_name()
        # Historical callers may still carry the retired identifier. Only the
        # adjacent bilingual layout is generated, with no inherited erasure.
        if mode=='chinese':mode='bilingual';mask_requested=False
        mask_requested=bool(mask_requested and mode=='bilingual')
        cache_key = ((document_id,page_index,mask_requested,mode,'preview') if for_preview else
                     (document_id,page_index,mask_requested,mode))
        cached = self._layout_cache.get(cache_key)
        from .layout_identity import LAYOUT_ALGORITHM_VERSION
        if cached is not None and cached.layout_version!=LAYOUT_ALGORITHM_VERSION:
            self._layout_cache.pop(cache_key,None)
            cached=None
        if cached is not None:
            if (for_preview and mask_requested and self._document.info.file_type=='PDF'
                    and (document_id,page_index) not in self._preview_mask_cache
                    and self._ocr_task is None):
                selection=self._target_cache.get((document_id,page_index))
                if selection is not None:self._request_mask_evidence((document_id,page_index),selection)
            return cached
        target_key = (document_id, page_index)
        selection = self._target_cache.get(target_key)
        translations = self._translation_cache.get(target_key)
        if selection is None or not translations:
            raise ValueError(f"第 {page_index + 1} 页没有可用人工审核译文")
        if for_preview:
            if target_key not in self._preview_source_cache:
                image=self._canonical_source_cache.get(page_index)
                if image is None:
                    image=self._document.render_page(page_index)
                    self._remember_canonical_source(page_index,image)
                self._preview_source_cache[target_key]=image
            background=None;verified=frozenset();supplemental=None;supplemental_reasons=None
            if mask_requested and self._document.info.file_type=='PDF' and self._ocr_task is None:
                if target_key not in self._preview_mask_cache:
                    self._request_mask_evidence(target_key,selection)
                else:
                    evidence=self._preview_mask_cache[target_key]
                    if hasattr(evidence,'raster_masks'):
                        background=evidence.background;verified=evidence.native_verified
                        supplemental=evidence.raster_masks;supplemental_reasons=evidence.rejections
                    else:background,verified=evidence
            from .annotation_preview import footprint
            fixed_obstacles=tuple(TextRegion('manual:'+item.group_id,document_id,page_index,'','',
                footprint(item.text_bounds,item.rotation_degrees))
                for entry in selection.candidate_entries
                for item in (self._manual_for_group(entry.group_id,mode),)
                if item is not None and mode!='chinese')
            # The mode is part of the immutable layout and its cache identity.
            # Never relabel a nearby bilingual caption as an in-place Chinese replacement.
            layout=self._layout_engine.layout_preview(self._page_geometry(page_index),
                selection.candidate_entries,translations,
                source_image=self._preview_source_cache[target_key],mask_requested=False,
                obstacles=(*tuple({r.id:r for r in (*self._ocr_cache.get(target_key,()),*selection.raw_regions)}.values()),*fixed_obstacles),
                mask_background=background,verified_mask_groups=verified,display_mode=mode,
                supplemental_masks=supplemental,supplemental_reasons=supplemental_reasons)
            layout=self._apply_manual_layouts(layout,self._preview_source_cache[target_key])
            if mode=='bilingual':
                from .annotation_preview import apply_bilingual_masks
                layout=apply_bilingual_masks(self._layout_engine,layout,selection.candidate_entries,
                    mask_requested,background,verified,selection.raw_regions)
            self._layout_cache[cache_key]=layout
            return layout
        export_results=self._export_results_for_page(page_index)
        if not export_results:
            raise ValueError(f'第 {page_index+1} 页没有可导出的英文注释译文')
        if any(item.provenance not in STAGE1_PROVENANCE for item in export_results.values()):
            raise ValueError("机器译文尚未完成人工审核，不能用于标注或导出")
        validate_translation_results(
            selection.candidate_entries, export_results.values(),
            allowed_provenance=STAGE1_PROVENANCE,
        )
        from dataclasses import replace
        preview=self._layout_for_page(page_index,mask_requested,for_preview=True,display_mode=mode)
        if not any(item.lines for item in preview.items):
            raise ValueError(f'第 {page_index+1} 页没有可导出的中文标注')
        blocked=[item.group_id for item in preview.items
                 if item.placement not in {'inline','preserved'}]
        if blocked:
            raise ValueError(f'第 {page_index+1} 页有 {len(blocked)} 条译文尚未安全布置：'+', '.join(blocked[:4]))
        layout=replace(preview,preview_only=False)
        self._layout_cache[cache_key] = layout
        return layout

    # 英文模式只显示底图
    # annotations勾选还需实际译文
    # 英文仍可建立检查用双语布局
    # 检查布局不等于绘制图层
    # 布局失败清除可见中文层
    # 无译文时也清检查身份
    # 双语才启用遮挡复选框
    # 状态更新与表格显示使用同一结果
    # 此刷新不改变机器审核来源
    def _refresh_annotation_preview(self, checked: bool | None = None) -> None:
        if self._document is None:
            return
        key = (self._document.info.id, self.current_page)
        translations = self._translation_cache.get(key)
        mode=self._display_mode_name()
        show = mode!='english' and self.actions["annotations"].isChecked() and bool(translations)
        if translations:
            try:
                layout = self._layout_for_page(
                    self.current_page, self.cover_checkbox.isChecked() and mode=='bilingual',for_preview=True,
                )
            except ValueError as exc:
                self.drawing_view.show_translation_layout(None)
                self.state_status.setText(f"标注预览失败：{exc}")
            else:
                inspection=(self._layout_for_page(self.current_page,False,for_preview=True,display_mode='bilingual')
                            if mode=='english' else layout)
                self.drawing_view.set_inspection_layout(inspection)
                self.drawing_view.show_translation_layout(layout if show else None)
        else:
            self.drawing_view.show_translation_layout(None)
            self.drawing_view.set_inspection_layout(None)
        self.cover_checkbox.setEnabled(mode=='bilingual')
        self._update_mask_status()
        self._display_cached_regions()
        self._update_annotation_actions()

    # 只读状态不能再发后台任务
    # 避免取消后立刻自动重试
    # 英文状态说明完整原图
    # 关闭开关说明全部英文保留
    # 安全计数只算真正inline中文
    # SAFE还需非空mask多边形
    # 候选数不等于全部OCR数
    # result_ready先于finished
    # 任务引用清除后才显示结束
    # 验证失败仍保留英文和红字
    def _update_mask_status(self):
        """Read current evidence state without scheduling another worker.

        result_ready can precede QThread.finished. Reporting completion must
        happen after the worker reference is cleared, including cancellation;
        refreshing the entire preview here would immediately retry a canceled
        request rather than respecting the user's cancel operation.
        """
        mode=self._display_mode_name()
        key=(self._document.info.id,self.current_page) if self._document else None
        if mode=='english':self.mask_status.setText('英文原文：显示完整原图')
        elif not self.cover_checkbox.isChecked():self.mask_status.setText('安全遮挡已关闭；保留全部英文')
        else:
            current=self._layout_cache.get((*key,True,'bilingual','preview')) if key else None
            candidates=[i for i in current.items if i.lines and i.placement=='inline'] if current else []
            safe=sum(i.mask_decision==MaskDecision.SAFE and bool(i.mask_polygons) for i in candidates)
            self.mask_status.setText((f'候选 {len(candidates)}；安全通过/已遮挡 {safe}；未通过 {len(candidates)-safe}' if safe else
                '本页没有通过安全验证、可自动遮挡的英文，已保留原文。')+('（正在验证）' if self._mask_task else ''))

    # 按钮状态来自真实任务与缓存
    # 先检查当前英文结果审核资格
    # 历史尺寸结果不应阻塞注释
    # 任一后台任务忙则关闭互斥动作
    # 取消不能中断导出写盘
    # 人工测试导入一直禁用
    # 显示中文要求有译文且空闲
    # 导出还必须有已计算安全布局
    # 只准inline或preserved项
    # 至少一条真正可绘中文
    # 英文模式不提供中文导出
    # 审核通过不能补齐待布局
    # 状态计算不发遮挡任务
    # 防止控件刷新递归启动后台
    def _update_annotation_actions(self) -> None:
        if self._document is None:
            return
        key = (self._document.info.id, self.current_page)
        has_targets = key in self._target_cache
        has_translations = bool(self._translation_cache.get(key))
        try:
            export_results=self._export_results_for_page(self.current_page)
            reviewed=bool(export_results) and all(item.provenance in STAGE1_PROVENANCE
                for item in export_results.values())
        except ValueError:
            reviewed=False
        idle = (self._ocr_task is None and self._translation_task is None
                and self._export_task is None and self._mask_task is None and not self._closing)
        self.actions["load_translations"].setEnabled(False)
        self.actions["translate"].setEnabled(idle)
        self.actions['cancel'].setEnabled(not idle and self._export_task is None and not self._closing)
        self.actions["annotations"].setEnabled(has_translations and idle)
        # Reviewing wording does not resolve a failed layout or grant erasure
        # rights. Reflect the already-computed current mode, without launching
        # evidence work from an action-state update or enabling English export.
        mode=self._display_mode_name()
        layout=self._layout_cache.get((*key,mode=='bilingual' and self.cover_checkbox.isChecked(),mode,'preview'))
        ready=bool(mode!='english' and layout and any(
            item.placement=='inline' and item.lines for item in layout.items)
            and all(item.placement in {'inline','preserved'} for item in layout.items))
        if ready and mode=='chinese':
            from .annotation_models import MaskDecision
            ready=all(item.mask_decision==MaskDecision.SAFE
                      for item in layout.items if item.placement=='inline')
        self.actions["export"].setEnabled(reviewed and idle and ready)
        self.cover_checkbox.setEnabled(mode=='bilingual')

    # 清当前页全部扩展布局键
    # 模式/遮挡/preview都不能漏
    # 同时清标准布局底图
    # 遮挡证据也需重新取得
    # 原始OCR和译文不在此删除
    def _invalidate_preview_page(self,key):
        for candidate in tuple(self._layout_cache):
            if candidate[:2]==key and len(candidate)>3:self._layout_cache.pop(candidate,None)
        self._preview_source_cache.pop(key,None)
        self._preview_mask_cache.pop(key,None)

    # 完成翻译只更新结果
    # 尊重用户当前英文选择
    # 不强行切换下拉模式
    # 批处理每页也遵守该规则
    def _show_chinese_preview(self):
        # Translation completion updates data, not the user's chosen language
        # mode. This also applies to cache hits and every page of a batch.
        self.actions['annotations'].setChecked(self._display_mode_name()!='english')
        self._refresh_annotation_preview()

    # 按语言模式同步标注显示
    # QSignalBlocker避免双重刷新
    # 切换复用既有翻译结果
    # 模式本身也保存到项目草稿
    def _display_mode_changed(self,index):
        with QSignalBlocker(self.actions['annotations']):self.actions['annotations'].setChecked(self._display_mode_name()!='english')
        self._refresh_annotation_preview()
        self.mark_project_dirty('display_mode')

    # 独立开关触发布局刷新
    # 不改变中文的邻近位置
    # 关闭后新布局不含mask
    # 用户选择进入项目dirty状态
    def _mask_option_changed(self,checked):
        self._refresh_annotation_preview()
        self.mark_project_dirty('mask_requested')

    # 先清OCR后自动翻译意图
    # OCR和翻译均执行软取消
    # 模型调用继续安全返回
    # 遮挡任务请求协作中断
    # 这里不清活动引用冒充空闲
    # 也不终止导出原子写盘
    def _cancel_pipeline(self):
        self._auto_translate_key=None
        if self._ocr_task is not None:self._ocr_session.cancel_active()
        if self._translation_task is not None:self._translation_session.cancel_active()
        if self._mask_task is not None:self._mask_task.requestInterruption()

    # 只验证当前页当前目标
    # 已有任务不重复提交
    # 输入标准预览源图
    # 原始OCR也作为保护障碍
    # expected_source_sha绑定原图字节
    # 任务自行打开独立PDF资源
    # 结果回调再检条目身份
    # 此过程不修改原PDF
    # 图像OCR框不提供擦除授权
    def _request_mask_evidence(self,key,selection):
        if self._mask_task is not None or self._closing or key!=(self._document.info.id,self.current_page):return
        from .preview_mask_evidence import MaskEvidenceTask
        task=MaskEvidenceTask(self._document.info.path,self._page_geometry(key[1]),
            selection.candidate_entries,self._preview_source_cache[key].width(),self,
            source_image=self._preview_source_cache[key],
            obstacles=tuple({r.id:r for r in (*self._ocr_cache.get(key,()),*selection.raw_regions)}.values()),
            expected_source_sha=self._source_identity_sha)
        self._mask_task=task
        task.result_ready.connect(self._mask_evidence_ready)
        task.finished.connect(self._mask_evidence_finished)
        task.start();self._update_annotation_actions()

    # 关闭或取消结果直接丢弃
    # 请求对象必须是当前任务
    # 文档页与目标元组都不变
    # 背景证据按页单独缓存
    # 只失效请求mask的布局变体
    # 无mask中文布局无需重排
    # 刷新使用同一中文邻近位置
    # 不能让旧证据污染新目标
    def _mask_evidence_ready(self,request,evidence):
        if self._closing or self._document is None or self._mask_task is None:return
        if self._mask_task.isInterruptionRequested():return
        key=(request.document_id,request.page_index);selection=self._target_cache.get(key)
        if (request!=self._mask_task.request or key!=(self._document.info.id,self.current_page)
            or selection is None or tuple(selection.candidate_entries)!=request.entries):return
        self._preview_mask_cache[key]=evidence
        for cached in tuple(self._layout_cache):
            if cached[:2]==key and len(cached)>3 and cached[2]:self._layout_cache.pop(cached,None)
        self._refresh_annotation_preview()

    # 先清任务引用再更新状态
    # 防止显示持续正在验证
    # deleteLater延迟安全回收
    # 关闭等待在此继续
    # 批次可能正等该任务让路
    # 完成后排队唤醒页队列
    # 旧页验证结束可补当前页需求
    # 取消本页不直接重启同任务
    def _mask_evidence_finished(self):
        task=self._mask_task;self._mask_task=None
        self._update_mask_status()
        if task is not None:task.deleteLater()
        if self._closing:self.close()
        else:
            self._update_annotation_actions()
            if self._batch_active:
                # The queue may have yielded while this evidence worker owned
                # the page. Its completion is an actual pipeline wake-up.
                QTimer.singleShot(0,self._advance_page_batch)
            if (task is not None and self._document is not None
                and (task.request.document_id,task.request.page_index)!=(self._document.info.id,self.current_page)
                and self._display_mode_name()=='bilingual' and self.cover_checkbox.isChecked()):
                self._refresh_annotation_preview()

    # 批次剩余页面立即清空
    # 已保存缓存结果保留
    # 管线仅屏蔽旧交付
    # UI提示等待后台安全结束
    # 任务未finished前仍显示忙碌
    def cancel_automatic_translation(self):
        self._batch_active=False
        self._batch_queue=[]
        self._cancel_pipeline()
        self.state_status.setText('已取消本次处理；正在等待后台安全结束')
        self._update_annotation_actions()

    # 画布点击按组定位列表
    # 全部OCR视图可能没有语义组行
    # 此时回候选分组视图
    # 不改OCR数据或结果身份
    # 选择后详情显示原文与译文
    def _inspect_group(self,group_id):
        if self._document is None:return
        key=(self._document.info.id,self.current_page)
        selection=self._target_cache.get(key)
        if (selection is not None and self.translation_panel.show_all_checkbox.isChecked()
            and not any(e.group_id==group_id for e in self.translation_panel._entries)):
            # Return to grouped results for this source/translation inspection;
            # diagnostic raw OCR rows have different identities by definition.
            self.translation_panel.show_all_checkbox.setChecked(False)
        self.translation_panel.select_group(group_id)

    # 对话框只返回输出意图
    # 取消不创建任务或文件
    # 确认后仍须完整资格检查
    # 范围和格式不会绕过审核
    def _choose_annotation_export(self) -> None:
        if self._document is None:
            return
        dialog = ExportDialog(
            self._document.info.path, self._document.info.page_count, self,
        )
        if dialog.exec() == dialog.DialogCode.Accepted:
            self.start_annotation_export(
                dialog.output_path, all_pages=dialog.all_pages,
                output_format=dialog.output_format,
            )

    # 先检查全部互斥任务空闲
    # 英文模式明确拒绝中文副本
    # PDF可选当前页或全页
    # 图片始终只有零号页
    # 导出前再核验源SHA
    # 源字节改变禁止旧坐标输出
    # 未处理页不能混成整份已审核
    # 确认为空候选的页可保留原页
    # 尺寸页保留但历史结果仍验身份
    # 每个有中文页取得正式共享布局
    # 没有任何布局则拒绝空导出
    # 请求携带人工审核结果字典
    # 也携带expected_source_sha256
    # 服务层还会独立执行门禁
    # 启动后禁用打开与识别
    # 单次ExportTask在线程内写盘
    # 返回True只表示任务已启动
    # 实际成功等回读验证结果
    def start_annotation_export(
        self, output_path: str | Path, *, all_pages: bool, output_format: str,
    ) -> bool:
        """用已缓存的共享布局启动一次非阻塞导出。"""
        if (self._document is None or self._export_task is not None
                or self._ocr_task is not None or self._translation_task is not None
                or self._closing or self._mask_task is not None):
            return False
        info = self._document.info
        if self._display_mode_name()=='english':
            self.state_status.setText('英文模式仅显示原图；导出中文副本前请选择中英对照')
            return False
        if info.file_type == "PDF":
            page_indices = (tuple(range(info.page_count)) if all_pages
                            else (self.current_page,))
        else:
            page_indices = (0,)
        mask_requested = self.cover_checkbox.isChecked()
        layouts: dict[int, AnnotationLayout] = {}
        try:
            from .project_store import source_sha256
            if source_sha256(info.path)!=self._source_identity_sha:
                raise ValueError('源图纸打开后发生变化，禁止使用旧坐标导出；请重新打开和识别')
            for page_index in page_indices:
                selection=self._target_cache.get((info.id,page_index))
                if (info.id, page_index) not in self._translation_cache:
                    if selection is not None and not selection.candidate_entries:continue
                    raise ValueError(f'第 {page_index+1} 页尚未处理，不能生成所选整份已审核图纸')
                from .source_text_policy import is_translatable_english
                if selection is not None and not any(is_translatable_english(entry.source_text)
                        for entry in selection.candidate_entries):
                    # Preserve a processed dimension-only page in a multi-page
                    # copy; still validate every historical result's identity.
                    self._export_results_for_page(page_index)
                    continue
                layouts[page_index] = self._layout_for_page(page_index, mask_requested)
            if not layouts:
                raise ValueError("所选范围内没有可导出的人工审核译文")
            request = ExportRequest(
                info.path, Path(output_path), info.id, layouts, page_indices,
                output_format,
                reviewed_results={r.group_id:r for p in page_indices
                    for r in self._export_results_for_page(p).values()},
                expected_source_sha256=self._source_identity_sha,
            )
        except (OSError, TypeError, ValueError) as exc:
            self.state_status.setText(f"无法开始导出：{exc}")
            return False
        task = ExportTask(request, self._export_service_factory, self)
        self._export_task = task
        task.result_ready.connect(self._export_ready)
        task.failed.connect(self._export_failed)
        task.finished.connect(self._export_finished)
        self.actions["open"].setEnabled(False)
        self.actions["recognize"].setEnabled(False)
        self._update_annotation_actions()
        self.state_status.setText("正在后台导出中文标注副本…")
        task.start()
        return True

    # 成功只接收当前导出请求
    # 关闭期间不覆盖用户状态
    # 服务返回前已执行输出回读
    # 提示保留安全告警数量
    # 不将文件存在当全部验收通过
    def _export_ready(self, request: ExportRequest, result: ExportResult) -> None:
        if self._closing or request != getattr(self._export_task, "request", None):
            return
        warning = f"；{len(result.warnings)} 条安全提示" if result.warnings else ""
        self.state_status.setText(f"导出完成并通过回读验证：{result.output_path}{warning}")

    # 只显示当前任务异常
    # 用户源图仍由服务保护
    # 失败不会伪造输出完成状态
    def _export_failed(self, request: ExportRequest, message: str) -> None:
        if not self._closing and request == getattr(self._export_task, "request", None):
            self.state_status.setText(f"导出失败：{message}")

    # 完成先清任务引用
    # Qt对象随后deleteLater
    # 延迟关闭在此继续
    # 否则恢复打开和识别动作
    # 导出资格仍按当前布局重算
    def _export_finished(self) -> None:
        task = self._export_task
        self._export_task = None
        if task is not None:
            task.deleteLater()
        if self._closing:
            self.close()
            return
        self.actions["open"].setEnabled(True)
        self.actions["recognize"].setEnabled(
            self._document is not None and self._ocr_task is None,
        )
        self._update_annotation_actions()

    # 这是遗留人工分类接口
    # 新日常面板不提供分类按钮
    # 复查区域先反查原始parent
    # 持久判定绑定原框指纹
    # 重建目标后旧译文必须失效
    # 知识追踪和全部模式布局同清
    # 手工分类不授予翻译审核来源
    # 原OCR仍保留可追溯
    # 修改进入项目dirty状态
    def _set_manual_decision(self, entry: TargetEntry, category) -> None:
        if self._document is None or entry.document_id != self._document.info.id:
            return
        source = self._document.info.path
        try:
            for region in entry.regions:
                key=(entry.document_id,entry.page_index)
                report=self._refinement_cache.get(key)
                parent=report.parent_by_region_id.get(region.id,region.id) if report else region.id
                original=next((r for r in self._ocr_cache[key] if r.id==parent),region)
                self._decision_store.set_override(source, original, category)
            key = (entry.document_id, entry.page_index)
            raw = self._ocr_cache[key]
            overrides = self._decision_store.overrides_for(source, raw)
        except (OSError, ValueError) as exc:
            self.state_status.setText(f"无法保存人工判定：{exc}")
            return
        self._target_cache[key] = self._select_page_targets(key,overrides)
        self._translation_cache.pop(key, None)
        self._translation_knowledge_cache.pop(key, None)
        # The cache now includes mode and evidence identity in addition to the
        # old cover flag. No generated caption may survive a target override.
        self._clear_product_layout(key)
        self.mark_project_dirty('target_override')
        self.actions["annotations"].setChecked(False)
        self._display_cached_regions()
        self._update_annotation_actions()
        self.state_status.setText("人工判定已保存")

    # 选中结果先核文档身份
    # 跨页条目先切到对应页
    # 高亮组内全部原始区域
    # 不会伪画字符级小框
    def _focus_region(self, entry: TargetEntry) -> None:
        if self._document is None or entry.document_id != self._document.info.id:
            return
        if entry.page_index != self.current_page:
            self.show_page(entry.page_index)
        self.drawing_view.focus_ocr_regions(entry.region_ids)

    # 显示视图矩阵的百分比
    # 源PDF point大小并未改变
    # 不是模型输入DPI
    def _update_zoom(self, percent: float) -> None:
        self.zoom_status.setText(f"{percent:.0f}%")

    # 只对当前PDF安排高清预览
    # 图片无需重复重采样
    # 防抖合并连续缩放事件
    # 关闭后不再排新渲染
    def _schedule_pdf_preview(self) -> None:
        if (not self._closing and self._document is not None
                and self._document.info.file_type == "PDF"):
            self._preview_timer.start()

    # 目标尺寸由倍率与DPR决定
    # 仅升级超过当前约10%的需求
    # 避免轻微缩放重复整页渲染
    # 已有预览任务就置pending
    # 同一窗口只跑一个高清任务
    # 请求固定当前文档ID与页码
    # Worker使用独立QtPdf文档
    # 不共享主线程LoadedDocument
    def _start_pdf_preview(self) -> None:
        if (self._closing or self._document is None
                or self._document.info.file_type != "PDF"):
            return
        target = self.drawing_view.pdf_render_size()
        current = self.drawing_view.page_raster_size
        if (target.isEmpty() or
                (target.width() <= current.width() * 1.1
                 and target.height() <= current.height() * 1.1)):
            return
        if self._preview_task is not None:
            self._preview_pending = True
            return
        info = self._document.info
        task = PdfPreviewTask(info.path, info.id, self.current_page, target, self)
        self._preview_task = task
        task.result_ready.connect(self._pdf_preview_ready)
        task.failed.connect(self._pdf_preview_failed)
        task.finished.connect(self._pdf_preview_finished)
        task.start()

    # 旧文档或旧页预览直接丢弃
    # 仅接受双维度不更小的图
    # 替换底图不改变场景尺寸
    # 因此OCR框与中文不发生漂移
    # 标准canonical底图不被升级图替代
    def _pdf_preview_ready(self, document_id: str, page_index: int, image) -> None:
        if (self._closing or self._document is None
                or self._document.info.id != document_id
                or self.current_page != page_index):
            return
        current = self.drawing_view.page_raster_size
        if image.width() >= current.width() and image.height() >= current.height():
            self.drawing_view.replace_page_image(image)

    # 只报告当前页高清失败
    # 已有低分辨率首帧仍可使用
    # 不清OCR或翻译缓存
    def _pdf_preview_failed(self, document_id: str, page_index: int,
                            message: str) -> None:
        if (not self._closing and self._document is not None
                and self._document.info.id == document_id
                and self.current_page == page_index):
            self.state_status.setText(f"高清预览失败：{message}")

    # 完成后安全释放任务引用
    # 关闭等待可继续
    # pending只合并一次后续需求
    # 再经防抖计算最新目标尺寸
    # 不用旧任务尺寸重复渲染
    def _pdf_preview_finished(self) -> None:
        task = self._preview_task
        self._preview_task = None
        if task is not None:
            task.deleteLater()
        if self._closing:
            self.close()
        elif self._preview_pending:
            self._preview_pending = False
            self._schedule_pdf_preview()

    def _show_about(self) -> None:
        QMessageBox.about(self, "关于软件", "MechanicalDrawingTranslator\n本地图纸预览、RapidOCR 与翻译目标筛选")

    # 关闭先检查所有后台所有者
    # 翻译prepare也算忙碌
    # 单次线程未finished不能销毁
    # 先设closing阻止新交付
    # 软取消OCR/翻译
    # 高清和遮挡请求协作中断
    # 导出仍等安全写完
    # 此轮close事件忽略
    # 各finished会再次调用close
    # 空闲后保护未保存修改
    # 草稿失败不能默认放弃
    # 隐藏调用者保守拒绝关闭
    # 可见用户才可明确Discard
    # 取消关闭恢复closing=False
    # 确认后等待草稿屏障
    # 常驻服务先在所属线程释放
    # 再quit/wait停止事件循环
    # 最后清场景和文档句柄
    # 不把活动QThread交给Qt强销毁
    # 完整源文件始终不被关闭流程改写
    def closeEvent(self, event) -> None:
        ocr_busy = self._ocr_task is not None
        translation_busy = self._translation_task is not None
        preparation_busy = self._translation_session.is_preparing
        preview_busy = self._preview_task is not None and not self._preview_task.isFinished()
        export_busy = self._export_task is not None and not self._export_task.isFinished()
        mask_busy = self._mask_task is not None and not self._mask_task.isFinished()
        if ocr_busy or translation_busy or preparation_busy or preview_busy or export_busy or mask_busy:
            self._closing = True
            self._preview_timer.stop()
            if ocr_busy:
                self._ocr_session.cancel_active()
            if translation_busy:
                self._translation_session.cancel_active()
            if preview_busy:
                self._preview_task.requestInterruption()
            if mask_busy:self._mask_task.requestInterruption()
            self.state_status.setText("正在等待后台任务结束…")
            event.ignore()
            return
        if self._project_dirty and not self.flush_recovery_draft():
            # A failed backup is not consent to lose live edits. Hidden API
            # callers fail closed; visible users may explicitly abandon this
            # live session while older recovery evidence remains untouched.
            discard=False
            if self.isVisible():
                discard=QMessageBox.question(self,'未保存的修改',
                    '恢复草稿无法写入。可取消关闭并另存项目；是否明确放弃本次未保存修改？',
                    QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel)==QMessageBox.StandardButton.Discard
            if not discard:
                self._closing=False
                self.state_status.setText('未关闭：请先保存项目、处理恢复草稿，或明确放弃本次未保存修改')
                event.ignore();return
            self.discard_unsaved_live_changes()
        self._closing = True
        self._ocr_session.shutdown()
        self._translation_session.shutdown()
        self.drawing_view.clear_page()
        if self._document:
            self._document.close()
            self._document = None
        super().closeEvent(event)
