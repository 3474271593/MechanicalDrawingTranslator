# mechanical_drawing_translator/panels.py
"""文件导航和翻译结果的可复用侧边面板。"""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFormLayout, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QListWidget, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,QScrollArea,QSizePolicy,
)

from .models import DocumentInfo, TextRegion
from .annotation_models import AnnotationLayout
from .translation_results import TranslationResult,preview_translation_available,translation_status_text
from .engineering_source_risk import warning_display_text
from .translation_targets import SelectionResult, TargetEntry
from .source_text_policy import is_translatable_english


class FilePanel(QWidget):
    """显示当前文件与页面列表，并发出零起始页码。"""

    page_selected = Signal(int)

    # 左侧只展示文件与导航
    # 业务页码信号采用零起始
    # 列表标签与SpinBox显示一始
    # 上一页下一页共用同一信号
    # SpinBox单独占整行
    # 适配190逻辑像素侧栏
    # 文件名允许换行
    # 没有文档时统一禁用导航
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setMinimumWidth(190)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(12)

        heading = QLabel("文件与页面")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        self.empty_label = QLabel("尚未打开图纸\n使用“文件 → 打开图纸”开始")
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)

        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        form = QFormLayout(info_frame)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.name_label = QLabel("—")
        self.name_label.setWordWrap(True)
        self.type_label = QLabel("—")
        self.count_label = QLabel("—")
        self.current_label = QLabel("—")
        for title, widget in (("图纸", self.name_label), ("格式", self.type_label),
                              ("总页数", self.count_label), ("当前页", self.current_label)):
            form.addRow(title, widget)
        layout.addWidget(info_frame)

        page_heading = QLabel("页面导航")
        page_heading.setObjectName("sectionHeading")
        layout.addWidget(page_heading)
        self.page_list = QListWidget()
        self.page_list.setObjectName("pageList")
        self.page_list.currentRowChanged.connect(self._selected_row_changed)
        layout.addWidget(self.page_list, 1)

        controls = QHBoxLayout()
        self.previous_button = QPushButton("上一页")
        self.next_button = QPushButton("下一页")
        self.page_spin = QSpinBox()
        self.page_spin.setPrefix("第 ")
        self.page_spin.setSuffix(" 页")
        self.page_spin.setMinimumWidth(90)
        # At the supported 190 logical-pixel sidebar width, a three-control
        # row cannot accommodate the spinbox plus two translated labels.
        # A separate full-width page field preserves every navigation action.
        controls.addWidget(self.previous_button, 1)
        controls.addWidget(self.next_button, 1)
        layout.addLayout(controls)
        layout.addWidget(self.page_spin)
        self.previous_button.clicked.connect(lambda: self.page_selected.emit(self.page_list.currentRow() - 1))
        self.next_button.clicked.connect(lambda: self.page_selected.emit(self.page_list.currentRow() + 1))
        self.page_spin.valueChanged.connect(lambda value: self.page_selected.emit(value - 1))
        self.set_document(None)

    # 列表索引本来就是零起始
    # 清空产生负索引时不发信号
    # 有效范围最终由窗口核验
    def _selected_row_changed(self, row: int) -> None:
        if row >= 0:
            self.page_selected.emit(row)

    # 一次性重建页列表
    # 填充时屏蔽列表与SpinBox信号
    # 否则初始选中会重复切页
    # 显示名只使用文件基本名
    # 空文档恢复占位符
    # 单页文档禁用页码跳转
    # 导航按钮按首尾范围更新
    def set_document(self, info: DocumentInfo | None) -> None:
        self.empty_label.setVisible(info is None)
        self.name_label.setText(info.name if info else "—")
        self.type_label.setText(info.file_type if info else "—")
        self.count_label.setText(str(info.page_count) if info else "—")
        self.current_label.setText("1" if info else "—")
        with QSignalBlocker(self.page_list), QSignalBlocker(self.page_spin):
            self.page_list.clear()
            if info:
                for index in range(info.page_count):
                    self.page_list.addItem(f"第 {index + 1} 页")
                self.page_list.setCurrentRow(0)
            self.page_spin.setRange(1, info.page_count if info else 1)
            self.page_spin.setValue(1)
        self.page_list.setEnabled(info is not None)
        self.page_spin.setEnabled(info is not None and info.page_count > 1)
        self._update_buttons(info.page_count if info else 0, 0)

    # 外部传入零起始页号
    # SpinBox显示时加一
    # 同步控件时避免反向再发信号
    # 当前页标签同步一始值
    # 不主动触发第二次渲染
    def set_current_page(self, page_index: int) -> None:
        count = self.page_list.count()
        if not 0 <= page_index < count:
            return
        with QSignalBlocker(self.page_list), QSignalBlocker(self.page_spin):
            self.page_list.setCurrentRow(page_index)
            self.page_spin.setValue(page_index + 1)
        self.current_label.setText(str(page_index + 1))
        self._update_buttons(count, page_index)

    # 第一页不能继续向前
    # 末页不能继续向后
    # 空文档和单页都禁用两按钮
    def _update_buttons(self, count: int, page_index: int) -> None:
        self.previous_button.setEnabled(count > 1 and page_index > 0)
        self.next_button.setEnabled(count > 1 and page_index < count - 1)


# 批量填表先暂停绘制
# 列宽计算临时改固定
# 避免逐单元格反复ResizeToContents
# 原始列模式逐列保存
# finally无条件恢复
# 异常也不能把表格永久冻住
# 只优化UI布局不裁掉数据
@contextmanager
def _batch_table_layout(table):
    """Defer per-cell automatic column sizing until the complete batch exists."""
    header = table.horizontalHeader()
    modes = [header.sectionResizeMode(i) for i in range(table.columnCount())]
    updates = table.updatesEnabled()
    table.setUpdatesEnabled(False)
    try:
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        yield
    finally:
        for index, mode in enumerate(modes):
            header.setSectionResizeMode(index, mode)
        table.setUpdatesEnabled(updates)


class TranslationPanel(QWidget):
    """Read-only source/translation inspection, without a daily review workflow."""

    region_selected = Signal(object)
    show_all_changed = Signal(bool)
    HEADERS = ("序号", "英文原文", "中文翻译", "所在页", "来源", "置信度",
               "原始坐标", "筛选结果", "处理状态")
    CATEGORY_LABELS = {
        "translatable": "需要翻译",
        "non_translatable": "无需翻译",
        "review_required": "自动尝试（疑难项）",
    }

    # 右侧表格默认只读
    # 中文编辑走项目菜单对话框
    # 可勾选全部OCR查看排除项
    # 默认只显示英文/中文/状态
    # 隐藏技术列仍保留数据
    # 整行选择用于组定位
    # 详情区域显示完整原文与证据
    # PlainText避免OCR被当HTML
    # 内容允许鼠标选择复制
    # 滚动详情容纳长说明
    # 表格不直接授予审核来源
    # 选择信号把TargetEntry交主窗口
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setMinimumWidth(280)
        self._entries: list[TargetEntry] = []
        self._translations: Mapping[str, TranslationResult] = {}
        self._knowledge_details = {}
        self._layout: AnnotationLayout | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(12)
        heading = QLabel("识别与翻译")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        self.show_all_checkbox = QCheckBox("显示全部 OCR 结果")
        self.show_all_checkbox.toggled.connect(self.show_all_changed.emit)
        layout.addWidget(self.show_all_checkbox)
        self.summary_label = QLabel("尚未识别")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.hint_label = QLabel("打开图纸后点击“自动翻译”，即可识别并显示中文。")
        self.hint_label.setObjectName("emptyState")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.setWordWrap(False)
        for column in (0,3,4,5,6,7):self.table.setColumnHidden(column,True)
        for column in (1,2):self.table.horizontalHeader().setSectionResizeMode(column,QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(32)
        layout.addWidget(self.table, 1)
        detail_heading = QLabel("原文与译文 · 点击图纸可定位")
        detail_heading.setObjectName("sectionHeading")
        layout.addWidget(detail_heading)
        self.detail_label = QLabel("选择一条识别结果后显示详细信息")
        self.detail_label.setObjectName("emptyState")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.detail_scroll=QScrollArea();self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setWidget(self.detail_label);self.detail_scroll.setMinimumHeight(140)
        self.detail_scroll.setMaximumHeight(300);layout.addWidget(self.detail_scroll)

    # 兼容早期单区域接口
    # 每个区域包装成疑难条目
    # 不在此做专业目标判断
    # 统一交set_entries显示
    def set_regions(self, regions: Iterable[TextRegion]) -> None:
        """保留第一阶段供外部调用的简单表格接口。"""
        entries = [TargetEntry((region,), "review_required", "尚未筛选")
                   for region in regions]
        self.set_entries(entries)

    # 条目顺序固定为输入顺序
    # 译文通过group_id匹配
    # 源文策略可覆盖历史显示
    # 尺寸历史译文不用于当前图面
    # 原始结果映射本身不改写
    # machine/manual_test/human分开标注
    # 布局待审核与译文状态并列
    # 坐标来自组的联合原框
    # 填表期间屏蔽选择回调
    # 每个单元格保留完整tooltip
    # summary原始数按raw_regions
    # 分类数按区域决策口径
    # 已布置数只算inline且有lines
    # 待布局不计已显示中文
    # 图框已识别需effective_bounds
    # 没有候选时提示查看全部OCR
    # 刷新后清除过期详情文字
    # 显示变化不触发后台推理
    def set_entries(
        self,
        entries: Iterable[TargetEntry],
        selection: SelectionResult | None = None,
        translations: Mapping[str, TranslationResult] | None = None,
        layout: AnnotationLayout | None = None,
        *, refinement=None, knowledge_details=None,
    ) -> None:
        # 保存条目对象而非解析表格文字。
        # 表格只是业务数据的展示投影。
        # 当前行号不会写入项目身份。
        # 共享原框的不同句段仍各有group_id。
        self._entries = list(entries)
        self._translations = translations or {}
        self._layout = layout
        self._refinement = refinement
        self._knowledge_details = knowledge_details or {}
        with QSignalBlocker(self.table), _batch_table_layout(self.table):
            self.table.clearSelection()
            self.table.setRowCount(len(self._entries))
            for row, entry in enumerate(self._entries):
                bounds = entry.bounds
                score = "—" if entry.confidence is None else f"{entry.confidence:.2f}"
                result = self._translations.get(entry.group_id)
                # 当前策略在显示时再次核验。
                # 旧项目可能曾翻译过尺寸字段。
                # 历史结果继续留在原缓存中。
                # 这里只选择本次展示的字面。
                excluded=not is_translatable_english(entry.source_text)
                translated = entry.source_text if excluded else result.translated_text if result else ""
                if excluded:
                    status='原文保留：非英文（尺寸/代号）'
                elif result is None:
                    status = "人工判定" if entry.reason == "人工判定" else "自动"
                elif result.provenance == "manual_test":
                    status = "人工测试译文（非机器翻译）"
                elif result.provenance == "machine":
                    status = translation_status_text(result)
                else:
                    status = "人工审核译文"
                # 布局结果按组ID关联当前行。
                # 不能用相同bounds猜译文对应关系。
                # 一个原框可能有多个语义跨度。
                # 待调整状态与机器翻译状态并列。
                layout_item=next((i for i in layout.items if i.group_id==entry.group_id),None) if layout else None
                if layout_item and layout_item.placement=='needs_layout_review':status+=' · 待调整布局'
                values = (str(row + 1), entry.source_text, translated,
                          str(entry.page_index + 1), entry.source, score,
                          f"{bounds.min_x:g}, {bounds.min_y:g} – {bounds.max_x:g}, {bounds.max_y:g}",
                          self.CATEGORY_LABELS[entry.category], status)
                for column, value in enumerate(values):
                    cell=QTableWidgetItem(value);cell.setToolTip(value);self.table.setItem(row,column,cell)
        if selection is None:
            self.summary_label.setText("尚未识别" if not self._entries else f"显示 {len(self._entries)} 条")
        else:
            counts = selection.counts
            self.summary_label.setText(
                f"原始 {len(selection.raw_regions)} · 需翻译 {counts['translatable']}"
                f" · 自动尝试 {counts['review_required']} · 排除 {counts['non_translatable']}"
            )
        if not self._entries and selection is not None and selection.raw_regions:
            hint = "本页没有翻译候选；可勾选“显示全部 OCR 结果”核对。"
        else:
            hint = "打开图纸后点击“自动翻译”，即可识别并显示中文。"
        self.hint_label.setText(hint)
        # 已布置只统计真正绘制的中文。
        # 只有译文字面却无lines不能计成功。
        # 待布局仍可在详情查看结果。
        # 图框证据存在不代表图框已确认。
        if layout:
            pending=sum(i.placement=='needs_layout_review' for i in layout.items)
            placed=sum(i.placement=='inline' and bool(i.lines) for i in layout.items)
            self.summary_label.setText(self.summary_label.text()+f' · 已布置 {placed} · 待调整 {pending}')
            if layout.frame_evidence:
                self.summary_label.setText(self.summary_label.text()+
                    (' · 图框已识别' if layout.effective_bounds is not None else ' · 图框未确认'))
        self.hint_label.setVisible(not self._entries)
        self.detail_label.setText("选择一条识别结果后显示详细信息")

    # 图上点击按组ID选行
    # 不靠英文全文模糊匹配
    # 选择后详情滚动回顶部
    # 没有当前组行则保持原选择
    def select_group(self,group_id) -> None:
        for row,entry in enumerate(self._entries):
            if entry.group_id==group_id:
                self.table.selectRow(row);self.detail_scroll.verticalScrollBar().setValue(0)
                return

    # 只有有效表格行才读条目
    # 历史结果另保留引用
    # 当前策略排除时只改显示副本
    # 不会覆盖旧批准或历史中文字面
    # 源坐标展示联合框与各原框
    # SourceSpan显示字符半开区间
    # 共享原框明确不是字符级几何
    # paragraph_id连接句与父段
    # semantic_units展示聚合前语义
    # 整段批准不扩散为逐句批准
    # 局部复查追溯到原parent
    # 候选方法/置信度/SHA可检查
    # 采用与回滚状态分开提示
    # machine标明自动预览来源
    # 历史排除译文只作为追溯说明
    # 警告用可读文本展示
    # mask结论不隐含人工批准
    # V2追踪含知识ID与源引用
    # shadow和candidate显示不同标签
    # 知识候选未采用也可保留证据
    # 最终发region_selected定位原框
    # 详情本身不修改任何业务对象
    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        selected = 0 <= row < len(self._entries)
        if selected:
            entry = self._entries[row]
            result = self._translations.get(entry.group_id)
            historical=result
            excluded=not is_translatable_english(entry.source_text)
            if excluded and result is not None:
                # Display policy does not rewrite the stored approval/history.
                result=replace(result,translated_text=entry.source_text,provenance='machine',
                    warnings=(*result.warnings,'source_text_preserved'))
            layout_item = next((item for item in self._layout.items
                                if item.group_id == entry.group_id), None) if self._layout else None
            bounds = entry.bounds
            score = "—" if entry.confidence is None else f"{entry.confidence:.2f}"
            # 联合矩形用于整组定位。
            # 逐原框坐标用于追溯碎片来源。
            # 两者均为源单位而非屏幕像素。
            # 不以原框数冒充独立语句数。
            coordinates = "; ".join(
                f"{r.bounds.min_x:g},{r.bounds.min_y:g}–{r.bounds.max_x:g},{r.bounds.max_y:g}"
                for r in entry.regions
            )
            details = (
                f"英文：{entry.source_text}\n中文：{result.translated_text if result else '尚未翻译'}"
                f"\n状态：{'原文保留：非英文（尺寸/代号）' if excluded else translation_status_text(result)}"
                f"\n\n技术信息\n页码：{entry.page_index + 1} · 来源：{entry.source}"
                f" · 置信度：{score}\n判定：{self.CATEGORY_LABELS[entry.category]}"
                f"（{entry.reason}）\n原始坐标：{bounds.min_x:g}, {bounds.min_y:g}"
                f" – {bounds.max_x:g}, {bounds.max_y:g}"
                f"\nGroup ID：{entry.group_id}\nRegion IDs：{', '.join(entry.region_ids)}"
                f"\n关联原始区域 {len(entry.regions)} 个：{coordinates}"
            )
            if entry.spans:
                offsets = "; ".join(
                    f"{s.region_id}[{s.start}:{s.end}]" for s in entry.spans
                )
                precision = ("共享原始区域范围（无逐字符坐标）"
                             if entry.coordinate_precision == "shared_region_bounds"
                             else "原始区域范围")
                details += (f"\n语义片段字符区间：{offsets}\n定位精度：{precision}"
                            f"\n段落关系：{entry.paragraph_id}"
                            "\n共享原始区域用于定位，不代表逐字符几何")
                if entry.boundary_notes:
                    details += "\n句界提示：" + "; ".join(entry.boundary_notes)
            # 展示聚合与语义输入是不同层次。
            # 保留原单元便于检查完整覆盖。
            # 整段中文不自动拆成各句答案。
            # 每个单元仍展示自己的跨度证据。
            if entry.semantic_units:
                details += (f'\n\n原语义单元：{len(entry.semantic_units)}条；当前中文按整段关联'
                            '\n整段参考不代表逐句独立人工批准。')
                for index,unit in enumerate(entry.semantic_units,1):
                    details += f'\n{index}. {unit.source_text}\n句ID：{unit.group_id}'
                    if unit.spans:
                        details += '\n来源跨度：'+'; '.join(f'{s.region_id}[{s.start}:{s.end}]' for s in unit.spans)
            # 原始OCR与局部采用结果同时可查。
            # 通过parent映射找到最初区域。
            # 候选图像SHA只标识实际图像输入。
            # 它不证明OCR或工程语义必然正确。
            refinement=getattr(self,"_refinement",None)
            if refinement is not None:
                if refinement.warnings:
                    details += "\n局部复查提示："+"; ".join(refinement.warnings)
                parents={refinement.parent_by_region_id.get(rid,rid) for rid in entry.region_ids}
                for record in refinement.records:
                    if record.region_id in parents:
                        details += f"\nOCR-3C父区域：{record.region_id}\n复查结论：{record.decision}"
                        if record.original_region is not None:
                            details += (f"\n原始OCR：{record.original_region.source_text}"
                                        f" · 置信度 {record.original_region.confidence}")
                        details += f"\n数字像素证据：{record.numeric_views}"
                        for candidate in record.candidates:
                            details += (f"\n图像候选[{candidate.method}]：{candidate.text}"
                                        f" · {candidate.confidence:.3f} · 图像SHA256 {candidate.image_sha256}")
                        if record.adopted is not None:
                            active=any(r.id!=record.region_id and
                                       refinement.parent_by_region_id.get(r.id)==record.region_id
                                       for r in entry.regions)
                            details += ("\n已采用局部图像结果；原始OCR保留，可回滚；定位引用原区域范围"
                                        if active else "\n局部图像结果已回滚；当前使用原始OCR")
            # 来源标签沿用结果数据契约。
            # 显示红字不能把machine提升为human。
            # manual_test只表明受控测试来源。
            # 真实审核应经明确菜单动作生成。
            if result is not None:
                provenance = {
                    "manual_test": "人工测试译文（非机器翻译）",
                    "human_reviewed": "人工审核译文",
                    "machine": "机器译文（自动预览，非人工审核来源）",
                }[result.provenance]
                if excluded and historical is not None:
                    details += (f'\n历史译文（当前策略排除，不用于图面）：{historical.translated_text}'
                                f'\n历史来源：{historical.provenance}；原审核记录保留')
                else:
                    details += f"\n译文：{result.translated_text}\n来源标记：{provenance}"
                if result.warnings:
                    details += f"\n翻译提示：{'；'.join(warning_display_text(item) for item in result.warnings)}"
            if layout_item is not None:
                details += f"\n遮挡结论：{layout_item.mask_decision.value}"
                if layout_item.warnings:
                    details += f"\n提示：{'；'.join(layout_item.warnings)}"
            # 影子知识追踪不代表已采用候选。
            # 来源引用与指纹供本地复查。
            # 无知识命中时也可能保留拒绝原因。
            # 不把知识记录数量当正确翻译数量。
            for trace in self._knowledge_details.get(entry.group_id, ()):
                label = '影子对照' if trace.mode == 'shadow' else '候选使用'
                details += (f'\nV2 {label}：{trace.decision} · {trace.scope}'
                            f'\n知识ID：{trace.knowledge_id or "无可用译文"}'
                            f'\n知识来源：{"; ".join(trace.source_refs) or "仅保留证据"}'
                            f'\n使用原因：{trace.reason}\n运行包：{trace.fingerprint}')
                if trace.candidate is not None:
                    details += f'\nV2候选译文：{trace.candidate}'
            # 详情作为纯文本一次更新。
            # 再发选择信号定位原图多边形。
            # 读取详情不会更改审核或遮挡资格。
            # 复制文字也不触发联网行为。
            self.detail_label.setText(details)
            self.region_selected.emit(entry)
