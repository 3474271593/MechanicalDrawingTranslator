# mechanical_drawing_translator/product_workflow.py
"""User project, explicit human review, manual layout and serial page workflow.

All mutations happen in the Qt main thread. Background OCR/translation retain
their existing worker sessions; a page queue advances only after completion.
"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from PySide6.QtCore import QTimer, QSignalBlocker
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from . import project_store
from .recovery_workflow import RecoveryWorkflow
from .recovery_store import project_payload_from_snapshot
from .annotation_models import MaskDecision
from .layout_identity import LAYOUT_ALGORITHM_VERSION
from .annotation_preview import BlankSpace, footprint
from .models import TextBounds
from .translation_results import validate_translation_results
from .production_logging import record_event


class ProductWorkflow(RecoveryWorkflow):
    # 先校验历史结果身份
    # 允许读取旧machine记录
    # 再按当前源文策略筛导出项
    # 尺寸历史译文不能误当中文注释
    # 原审核记录不会因此被删除
    # 真正英文待审核仍阻止导出
    # 此处筛选不提升provenance
    def _export_results_for_page(self, page_index):
        """Validate historical identity, then gate only currently eligible notes.

        Old projects may contain machine translations of dimensions that the
        current policy preserves. Keep those records intact without treating
        them as captions requiring review. Eligible pending text still blocks.
        """
        from .source_text_policy import is_translatable_english
        key=(self._document.info.id,page_index)
        selection=self._target_cache.get(key)
        results=self._translation_cache.get(key,{})
        if selection is None:return {}
        validated=validate_translation_results(selection.candidate_entries,results.values(),
            allowed_provenance=frozenset({'machine','manual_test','human_reviewed'}))
        eligible={entry.group_id for entry in selection.candidate_entries
                  if is_translatable_english(entry.source_text)}
        return {group:result for group,result in validated.items() if group in eligible}

    # 工作流混入主窗口使用
    # 所有状态变更在Qt主线程
    # 恢复草稿先初始化
    # manual_layouts按组与模式存储
    # machine_suggestions留首版建议
    # review_events记录显式修改
    # 批处理队列独立于单页请求
    # 正式项目路径初始为空
    # 结果身份与当前处理身份分开
    # 项目与审核菜单是真实入口
    # 编辑通过对话框提交
    # 审核按钮是显式人工动作
    # 恢复机器不等于取消历史记录
    # 位置调整保存源图坐标
    # 风险轮廓仅影响画布诊断
    # 草稿恢复/丢弃也可菜单操作
    # 图上拖动连接统一调整方法
    # 菜单动作不创建第二套模型
    def _initialize_product_workflow(self):
        self._initialize_recovery_workflow()
        self._manual_layouts = {}
        self._legacy_display_state = None
        self._review_events = []
        self._machine_suggestions = {}
        self._batch_queue = []
        self._batch_total = 0
        self._batch_results = {}
        self._batch_active = False
        self._project_path = None
        self._result_runtime_identity=self._translation_runtime_identity
        self._page_result_runtime={}
        for key, title, slot, shortcut in (
            ('save_project', '保存项目…', self.choose_save_project, 'Ctrl+S'),
            ('open_project', '打开项目…', self.choose_open_project, 'Ctrl+Shift+O'),
            ('translate_all', '翻译整份图纸', self.translate_all_pages, None),
            ('translate_range', '翻译指定页范围…', self.choose_page_range, None),
            ('edit_translation', '编辑当前译文…', self.choose_edit_translation, None),
            ('review_translation', '审核通过当前译文', lambda: self.review_translation(self._selected_group()), None),
            ('pending_translation', '当前译文标记待确认', lambda: self.review_translation(self._selected_group(), approved=False), None),
            ('restore_machine', '恢复机器建议', self.restore_machine_translation, None),
            ('adjust_annotation', '调整位置、字号及宽度…', self.choose_adjust_annotation, None),
            ('reset_annotation', '恢复当前标注自动布局', self.reset_annotation, None),
        ):
            action = QAction(title, self); action.triggered.connect(slot)
            if shortcut: action.setShortcut(shortcut)
            self.actions[key] = action
        menu = self.menuBar().addMenu('项目与审核')
        for key in ('open_project', 'save_project', 'translate_all', 'translate_range',
                    'edit_translation', 'review_translation', 'pending_translation', 'restore_machine',
                    'adjust_annotation', 'reset_annotation'): menu.addAction(self.actions[key])
        inspect=QAction('显示遮挡范围与风险轮廓',self);inspect.setCheckable(True)
        inspect.toggled.connect(self.drawing_view.set_diagnostics_visible)
        self.actions['inspect_masks']=inspect;menu.addAction(inspect)
        for key,title,slot in (('recover_draft','恢复未保存草稿',self.restore_recovery_draft),
                               ('discard_draft','丢弃未保存草稿',self.discard_recovery_draft)):
            action=QAction(title,self);action.triggered.connect(lambda checked=False,fn=slot:fn())
            self.actions[key]=action;menu.addAction(action)
        self.drawing_view.annotation_moved.connect(self._annotation_moved)

    # 行号只用于查当前条目
    # group_id才是业务身份
    # 表格排序或刷新后重新取
    # 没有有效选择返回None
    def _selected_group(self):
        row = self.translation_panel.table.currentRow()
        return self.translation_panel._entries[row].group_id if 0 <= row < len(self.translation_panel._entries) else None

    # Ctrl+S打开正式保存选择
    # 默认复用当前项目路径
    # 未保存项目由用户选目录
    # 文件类型为mdt.json
    # 取消不改变任何状态
    def choose_save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存翻译项目', str(self._project_path or ''), 'MDT 项目 (*.mdt.json)')
        if path: self.save_project(path)

    # 先保护当前未保存修改
    # 草稿写失败才询问放弃
    # 默认取消避免意外丢编辑
    # 放弃只针对当前会话
    # 旧草稿与正式项目仍保留
    # 替换失败恢复dirty保护
    # 成功后才提示新源恢复草稿
    def choose_open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开翻译项目', '', 'MDT 项目 (*.json)')
        abandoned_document=None
        if path and self._project_dirty and not self.flush_recovery_draft():
            choice=QMessageBox.warning(self,'当前修改尚未保存',
                '无法安全写入当前恢复草稿。放弃当前会话的未保存修改并打开项目？已有恢复草稿和正式项目仍会保留。',
                QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if choice!=QMessageBox.StandardButton.Discard:return
            abandoned_document=self._document.info.id
            self.discard_unsaved_live_changes()
        if path:
            if self.load_project(path):self.choose_recovery_draft()
            elif self._document is not None and self._document.info.id==abandoned_document:
                # Abandonment was conditional on replacing this session. An
                # invalid replacement leaves the old edits live and unsaved.
                self.mark_project_dirty('failed_project_replacement')

    # 快照捕获必须在主线程
    # 冻结模型可以共享引用
    # 可变容器必须独立复制
    # SelectionResult内部映射也复制
    # 译文各页字典单独复制
    # 审核事件深拷贝
    # 历史模式资料深拷贝
    # 不包含QImage或文档句柄
    # 源路径使用解析后绝对路径
    # 源SHA沿用打开时身份
    # 编码时再次检查实际字节
    # 保存两种运行身份
    # page_result_runtime按页追溯
    # 记录双模式产品版本二
    # 旧chinese数据置legacy字段
    # 布局版本供重建资格判断
    # geometry覆盖全部实际页面
    # 后台编码只消费这个快照
    def _project_snapshot(self):
        """Capture containers on the GUI thread; immutable model values may share.

        SelectionResult is frozen but its count/decision mappings are mutable,
        so copy those too. No QImage, document handle or widget crosses to the
        encoder thread. Review events are small mutable JSON dictionaries.
        """
        import copy
        if self._document is None: raise ValueError("没有已打开图纸")
        source=self._document.info.path.resolve()
        targets={key:replace(value,decisions=dict(value.decisions),counts=dict(value.counts))
            for key,value in self._target_cache.items()}
        payload = {'schema_version': 1, 'kind': 'mdt_project',
            'source': {'path': str(source), 'sha256': self._source_identity_sha,
                       'format': self._document.info.file_type, 'page_count': self._document.info.page_count},
            'document': self._document.info,
            'runtime_identity': self._result_runtime_identity,
            'processing_runtime_identity':self._translation_runtime_identity,
            'page_result_runtime':dict(self._page_result_runtime),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'current_page': self.current_page,
            'mask_requested': self.cover_checkbox.isChecked(),
            'display_mode': self._display_mode_name(),
            'product_display_version': 2,
            'legacy_display_state': copy.deepcopy(self._legacy_display_state),
            'layout_algorithm': LAYOUT_ALGORITHM_VERSION,
            'ocr': dict(self._ocr_cache), 'targets': targets,
            'translations': {key:dict(value) for key,value in self._translation_cache.items()},
            'layouts': dict(self._layout_cache), 'manual_layouts': dict(self._manual_layouts),
            'machine_suggestions': dict(self._machine_suggestions),
            'review_events': copy.deepcopy(self._review_events), 'page_status': dict(self._batch_results),
            'geometry': tuple(self._page_geometry(i) for i in range(self._document.info.page_count))}
        return payload

    # 正式保存复用草稿编码契约
    # 源SHA和PDF盒再次核验
    # 返回数据不含任何活动控件
    def _project_payload(self):
        return project_payload_from_snapshot(self._project_snapshot())

    # 没有图纸不能保存空项目
    # 保存前核验源图未变化
    # 禁止目标覆盖源图
    # 覆盖已打开正式项目需验base_sha
    # 外部较新审核不得被静默覆盖
    # 冲突提示用户另存
    # payload编码失败保留原项目
    # 写盘由同目录原子替换完成
    # 旧文件另有bak
    # 成功后更新恢复基准SHA
    # 日志只记保存事件与页数
    # False返回值不会伪装成功
    def save_project(self, path):
        if self._document is None: return False
        try:
            source = self._document.info.path.resolve()
            source_sha=project_store.source_sha256(source)
            if source_sha != self._source_identity_sha: raise ValueError('原图纸打开后发生变化；请保留项目并重新识别')
            if Path(path).resolve() == source: raise ValueError('项目文件不能覆盖原图纸')
            if self._project_path and Path(path).resolve()==self._project_path.resolve():
                current_sha=project_store.source_sha256(path) if Path(path).is_file() else None
                if current_sha!=self._project_base_sha:
                    raise ValueError('正式项目已被更新；禁止覆盖较新的审核，请另存项目')
            payload = self._project_payload()
            # Geometry is explicitly recorded as plain metadata; its dataclass is
            # not consumed on restore, which checks actual source dimensions.
            project_store.write_project(path, payload)
            self.state_status.setText('项目已安全保存')
            self._project_saved_for_recovery(path)
            record_event('project_saved',page_count=self._document.info.page_count)
            return True
        except (OSError, ValueError, TypeError) as exc:
            self.state_status.setText(f'项目保存失败：{exc}'); return False

    # 在OCR/翻译/导出忙时拒绝
    # 先保护当前dirty状态
    # loading抑制恢复过程中自动写草稿
    # 先读纯数据并核验源SHA
    # 再还原白名单业务类型
    # 完整校验早于替换当前文档
    # 结果和目标逐项校验来源
    # 重新打开源只为当前文档句柄
    # 恢复原UUID保持缓存键稳定
    # 已保存生成布局不直接信任
    # 当前_layout_cache从空开始
    # 人工布局与机器建议独立恢复
    # 审核事件保持原状态
    # 正式文件SHA成为并发保护基准
    # 旧结果运行身份单独保留
    # 配置不匹配只提示不重推理
    # 当前处理身份用于后续新请求
    # 旧chinese迁移成bilingual
    # legacy副本保留历史几何
    # 旧SAFE标志不能继承新资格
    # QSignalBlocker避免恢复控件触发重算
    # 恢复页码后显示已有缓存
    # 安全遮挡只恢复产品v2明确开关
    # 历史中文模式强制关闭遮挡
    # 不重新加载OCR或翻译模型
    # 异常给出状态与可选弹窗
    # finally总退出loading状态
    def load_project(self, path, show_error=True, *, _payload=None, _formal_path=None):
        if self._ocr_task or self._translation_task or self._export_task: return False
        if _payload is None and self._project_dirty and not self.flush_recovery_draft():return False
        self._loading_project=True
        try:
            payload = project_store.read_project(path) if _payload is None else _payload
            if project_store.source_sha256(payload['source']['path'])!=payload['source']['sha256']:
                raise ValueError('原图纸已改变，禁止恢复旧坐标')
            info = project_store.decode(payload['document'])
            caches = {key: project_store.decode(payload[key]) for key in ('ocr', 'targets', 'translations', 'layouts', 'manual_layouts', 'machine_suggestions')}
            self._validate_project_payload(payload, info, caches)
            for key, results in caches['translations'].items():
                selection = caches['targets'].get(key)
                if selection is None: raise ValueError('项目缺少对应英文区域')
                validate_translation_results(selection.candidate_entries, results.values(), allowed_provenance=frozenset({'machine','manual_test','human_reviewed'}))
            if not self.open_document(payload['source']['path'], show_error=False): raise ValueError('原图纸无法打开')
            if self._document.info.page_count != info.page_count: raise ValueError('图纸页数不匹配')
            self._document.info = info
            self._ocr_cache = caches['ocr']; self._target_cache = caches['targets']
            self._translation_cache = caches['translations']; self._layout_cache = {}
            self._manual_layouts = caches['manual_layouts']; self._machine_suggestions = caches['machine_suggestions']
            self._review_events = payload.get('review_events', [])
            self._batch_results = payload.get('page_status', {})
            self._project_path = Path(path) if _payload is None else (Path(_formal_path) if _formal_path else None)
            self._project_base_sha=project_store.source_sha256(path) if _payload is None else None
            self._project_dirty=False
            self._recovery_pending_decision=(_payload is None and self.recovery_draft_status()!='missing')
            self._result_runtime_identity=payload['runtime_identity']
            self._page_result_runtime=payload.get('page_result_runtime',{})
            if payload.get('runtime_identity') != self._translation_runtime_identity:
                self.state_status.setText('模型/知识版本发生变化；已恢复已有结果，重新翻译需人工确认')
            self._translation_cache_namespace = self._translation_runtime_identity
            restored_mode=payload.get('display_mode','bilingual')
            # Preserve retired geometry exactly as encoded for recovery. Never
            # load its cached SAFE flags into the current drawing pipeline.
            import copy
            self._legacy_display_state=copy.deepcopy(payload.get('legacy_display_state'))
            if restored_mode=='chinese':
                self._legacy_display_state={name:copy.deepcopy(payload.get(name)) for name in
                    ('display_mode','mask_requested','layout_algorithm','layouts','manual_layouts')}
                restored_mode='bilingual'
            with QSignalBlocker(self.display_mode_selector):
                self.display_mode_selector.setCurrentIndex(('english','bilingual').index(restored_mode))
            self.file_panel.set_document(info)
            target_page = payload.get('current_page', 0)
            if target_page: self.show_page(target_page)
            self._display_cached_regions(); self._update_annotation_actions()
            with QSignalBlocker(self.cover_checkbox):
                self.cover_checkbox.setChecked(payload.get('product_display_version')==2 and
                    payload.get('display_mode')!='chinese' and payload.get('mask_requested') is True)
            self.actions['annotations'].setChecked(True); self._refresh_annotation_preview()
            mismatch=payload.get('runtime_identity')!=self._translation_runtime_identity
            self.state_status.setText('项目已恢复；未重复识别或翻译'+('；模型/知识版本不同，旧结果来源身份已保留' if mismatch else ''))
            if payload.get('display_mode')=='chinese':
                self.state_status.setText('旧中文模式已迁移为中英对照；历史布局与人工调整保留，安全遮挡默认关闭')
            if mismatch:
                self.preview_hint.setText('当前模型/知识版本与已保存结果不同；保留旧译文及来源，重新翻译需另行处理。')
                self.preview_hint.show()
            record_event('project_restored',page_count=info.page_count)
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            self.state_status.setText(f'项目恢复失败：{exc}')
            if show_error: QMessageBox.warning(self, '无法恢复项目', str(exc))
            return False
        finally:
            self._loading_project=False

    # 导入数据不能只信schema
    # 显示模式必须在兼容白名单
    # DocumentInfo与source路径一致
    # 格式与页数也必须一致
    # 临时打开真实源检查尺寸
    # 每页geometry须完整且有序
    # 尺寸允许0.01的数值误差
    # 临时句柄finally释放
    # 缓存键绑定文档和有效页号
    # 目标内所有区域须同页同文档
    # OCR边界和角度必须有限
    # 越页坐标安全拒绝
    # 生成布局另验页面身份
    # 人工布局必须找到英文锚点
    # 兼容旧字符串与新元组键
    # 当前页索引也验证范围
    # 审核来源还经结果校验器复核
    # 旧数据可追溯不等于可直接绘制
    def _validate_project_payload(self, payload, info, caches):
        """Validate completely before replacing live UI or trusting saved approval.

        Source bytes, metadata, source spans, group identities and actual page
        geometry must agree. Imported nonfinite/foreign coordinates fail closed.
        """
        import math
        if payload.get('display_mode','bilingual') not in {'chinese','bilingual','english'}:
            raise ValueError('项目显示模式无效')
        from .document_service import DocumentLoader
        source=Path(payload['source']['path']).resolve()
        if info.path.resolve()!=source: raise ValueError('项目源文件身份不一致')
        if info.page_count!=payload['source']['page_count'] or info.file_type!=payload['source']['format']:
            raise ValueError('项目源文件元数据不一致')
        temporary=DocumentLoader.open(source)
        try:
            if temporary.info.page_count!=info.page_count or temporary.info.file_type!=info.file_type:
                raise ValueError('源图纸页数或格式改变')
            geometry=[project_store.decode(g) for g in payload['geometry']]
            if len(geometry)!=info.page_count: raise ValueError('项目页面几何缺失')
            for index,g in enumerate(geometry):
                actual=temporary.source_size(index)
                if (g.document_id!=info.id or g.page_index!=index or
                    abs(g.width-actual.width())>.01 or abs(g.height-actual.height())>.01):
                    raise ValueError('项目页面几何不匹配')
        finally: temporary.close()
        groups={}
        for cache_name in ('ocr','targets','translations','layouts'):
            for key,value in caches[cache_name].items():
                if (not isinstance(key,tuple) or len(key)<2 or key[0]!=info.id
                    or not isinstance(key[1],int) or not 0<=key[1]<info.page_count):
                    raise ValueError('项目缓存文档或页码不一致')
                if cache_name=='targets':
                    for entry in (*value.all_entries,*value.candidate_entries):
                        if not entry.regions or entry.document_id!=info.id or entry.page_index!=key[1]:
                            raise ValueError('项目英文目标身份不一致')
                        groups[entry.group_id]=entry
                    regions=(*value.raw_regions,*(r for e in (*value.all_entries,*value.candidate_entries) for r in e.regions))
                elif cache_name=='ocr': regions=value
                else: regions=()
                for region in regions:
                    if region.document_id!=info.id or region.page_index!=key[1]: raise ValueError('项目OCR身份不一致')
                    g=geometry[key[1]]; b=region.bounds
                    if (not all(math.isfinite(n) for n in (b.min_x,b.min_y,b.max_x,b.max_y,region.rotation_degrees))
                        or b.min_x<-.1 or b.min_y<-.1 or b.max_x>g.width+.1 or b.max_y>g.height+.1):
                        raise ValueError('项目OCR坐标无效')
        for key,layout in caches['layouts'].items():
            g=geometry[key[1]]
            if layout.document_id!=info.id or layout.page_index!=key[1] or layout.page_width!=g.width or layout.page_height!=g.height:
                raise ValueError('项目标注页面身份不一致')
            for item in layout.items: self._validate_saved_item(item,groups,g)
        for group,item in caches['manual_layouts'].items():
            if isinstance(group,tuple):
                if len(group)!=2 or group[1] not in {'chinese','bilingual','english'}:
                    raise ValueError('项目人工位置显示模式无效')
                group=group[0]
            entry=groups.get(group)
            if entry is None or group!=item.group_id: raise ValueError('项目人工标注缺少英文锚点')
            self._validate_saved_item(item,groups,geometry[entry.page_index])
        page=payload.get('current_page',0)
        if not isinstance(page,int) or not 0<=page<info.page_count: raise ValueError('项目当前页无效')

    # 标注必须匹配原英文与Region IDs
    # 未知group不能挂靠任意页面
    # 未绘制项只有原始源矩形
    # 不能再次旋转它造出页外墨迹
    # 真正绘制文字检查旋转footprint
    # 字号限定6到18
    # 坐标和角度都须为有限数
    # 页面边缘仅容忍微小浮点误差
    # text_origin由专用几何规则核验
    # 人工批准不能豁免几何门禁
    @staticmethod
    def _validate_saved_item(item,groups,geometry):
        import math
        entry=groups.get(item.group_id)
        if entry is None or item.source_text!=entry.source_text or item.region_ids!=entry.region_ids:
            raise ValueError('项目标注与原英文不一致')
        # A pending/unavailable item retains its original source rectangle and
        # orientation but has no painted Chinese footprint. Rotating that raw
        # source box again invents ink outside the page and blocks valid files.
        unpainted=(not item.lines and item.placement in
                   {'needs_layout_review','detail','unavailable','preserved'})
        b=item.text_bounds if unpainted else footprint(item.text_bounds,item.rotation_degrees)
        if (not math.isfinite(item.font_size) or not 6<=item.font_size<=18
            or not all(math.isfinite(v) for v in (b.min_x,b.min_y,b.max_x,b.max_y,item.rotation_degrees))
            or b.min_x<-.01 or b.min_y<-.01 or b.max_x>geometry.width+.01 or b.max_y>geometry.height+.01):
            raise ValueError('项目标注坐标或字号无效')
        from .glyph_text_geometry import validate_text_origin
        validate_text_origin(item)

    # 空中文修改被拒绝
    # 按当前文档页组定位结果
    # 首版机器建议只保存一次
    # 修改后重新标machine
    # 编辑不是人工审核批准
    # 旧translation_unavailable不适用新字面
    # 工程风险警告继续保留
    # 审核事件记录改前改后文本
    # 这是项目内容而不是操作日志
    # 清所有当前页生成布局
    # 重新预览及更新导出按钮
    # 标dirty并排恢复草稿
    def edit_translation(self, group_id, text):
        if self._document is None or not text.strip(): return False
        key = (self._document.info.id, self.current_page)
        result = self._translation_cache.get(key, {}).get(group_id)
        if result is None: return False
        self._machine_suggestions.setdefault(group_id, result)
        # The old machine availability verdict no longer describes user text.
        # Keep engineering evidence and the immutable original suggestion; only
        # explicit review may promote this new text to human_reviewed.
        self._translation_cache[key][group_id] = replace(result, translated_text=text.strip(), provenance='machine',
            warnings=tuple(w for w in result.warnings if w!='translation_unavailable'))
        self._review_events.append({'action':'edit', 'group_id':group_id, 'source_text':result.source_text,
            'previous_text':result.translated_text, 'text':text.strip(), 'at':datetime.now(timezone.utc).isoformat()})
        self._clear_product_layout(key); self._refresh_annotation_preview(); self._update_annotation_actions()
        self.mark_project_dirty('translation_edit')
        record_event('translation_edited',page_index=self.current_page)
        return True

    # 只编辑当前选择的组
    # 多行输入支持完整说明句
    # 对话框提示修改后需重新审核
    # 取消不触发字面更改
    def choose_edit_translation(self):
        group = self._selected_group()
        if not group or self._document is None: return
        result = self._translation_cache.get((self._document.info.id,self.current_page),{}).get(group)
        if result:
            text, ok = QInputDialog.getMultiLineText(self, '编辑中文（修改后需重新审核）', result.source_text, result.translated_text)
            if ok: self.edit_translation(group, text)

    # 批准由显式菜单动作触发
    # approved才标human_reviewed
    # 待确认回到machine
    # 保留首次机器建议供回滚
    # 审核事件附UTC时间
    # 审核只代表这版译文来源
    # 不能证明背景可安全擦除
    # 也不能解决待布局几何
    # 更新布局与正式导出门禁
    # 所有修改进入项目dirty状态
    def review_translation(self, group_id, approved=True):
        if self._document is None: return False
        key = (self._document.info.id, self.current_page)
        result = self._translation_cache.get(key, {}).get(group_id)
        if result is None: return False
        self._machine_suggestions.setdefault(group_id, result)
        self._translation_cache[key][group_id] = replace(result, provenance='human_reviewed' if approved else 'machine')
        self._review_events.append({'action':'approve' if approved else 'pending', 'group_id':group_id,
            'source_text':result.source_text,'text':result.translated_text,'at':datetime.now(timezone.utc).isoformat()})
        self._clear_product_layout(key); self._refresh_annotation_preview(); self._update_annotation_actions()
        self.mark_project_dirty('review')
        record_event('translation_reviewed',page_index=self.current_page,status='approved' if approved else 'pending')
        return True

    # 恢复完整原始机器结果
    # 不是只复制建议中文字面
    # 旧不可用与风险警告一并恢复
    # 来源重新为machine
    # 不沿用当前人工审核标志
    # 审核事件保留前一版内容
    # 刷新预览并重验导出资格
    # 该动作不删除既往审核历史
    def restore_machine_translation(self):
        group = self._selected_group(); result = self._machine_suggestions.get(group)
        if result is None or self._document is None:return False
        key=(self._document.info.id,self.current_page)
        current=self._translation_cache.get(key,{}).get(group)
        if current is None:return False
        # Restore the entire original verdict, including unavailable and risk
        # warnings. Reusing edit_translation would erase that original verdict
        # and retain warnings belonging to a different revision.
        self._translation_cache[key][group]=replace(result,provenance='machine')
        self._review_events.append({'action':'restore_machine','group_id':group,
            'source_text':result.source_text,'previous_text':current.translated_text,
            'text':result.translated_text,'at':datetime.now(timezone.utc).isoformat()})
        self._clear_product_layout(key);self._refresh_annotation_preview();self._update_annotation_actions()
        self.mark_project_dirty('restore_machine')
        record_event('machine_suggestion_restored',page_index=self.current_page)
        return True

    # 按文档和页清全部布局变体
    # 包含模式/遮挡/预览键
    # 不能只清旧三元缓存键
    # 遍历键副本允许安全删除
    def _clear_product_layout(self, key):
        for cache_key in tuple(self._layout_cache):
            if cache_key[:2] == key: self._layout_cache.pop(cache_key)

    # 历史中文分支仅供兼容调用
    # 保存位置不能授予擦除权
    # 中文编辑仍须当前证据复验
    # 复验失败清除lines和mask
    # 保留needs_layout_review供用户查
    # 双语人工位置重新按完整译文排版
    # 旧换行不能覆盖新译文字面
    # 宽度与字号都重新约束
    # 旋转footprint限制在页面内
    # 人工条目主动清遮挡多边形
    # 不能把移动后的中文当原英文位置
    # 最后用实际源图检查空白
    # 还检查与其他中文重叠
    # 风险位置保留数据但不正式导出
    # 手动锁定不是安全免检
    # 返回新不可变布局供共享绘制
    def _apply_manual_layouts(self, layout, source_image):
        if layout.display_mode=='chinese':
            from .safe_annotation_edit import edit_chinese_item
            page=self._page_geometry(layout.page_index);items=[]
            for original in layout.items:
                manual=self._manual_for_group(original.group_id,'chinese')
                if manual is None:items.append(original);continue
                try:
                    edited=edit_chinese_item(self._layout_engine,original,page,source_image,
                        x=manual.text_bounds.min_x,y=manual.text_bounds.min_y,
                        width=manual.text_bounds.max_x-manual.text_bounds.min_x,
                        font_size=manual.font_size,obstacles=self._chinese_edit_obstacles(layout,original))
                    items.append(edited)
                except (ValueError,RuntimeError) as exc:
                    # Saved polygons never grant erase authority. Current evidence
                    # must validate again after reopen, edits or mode changes.
                    items.append(replace(original,placement='needs_layout_review',lines=(),
                        mask_polygons=(),mask_decision=MaskDecision.NOT_REQUESTED,
                        warnings=(*original.warnings,f'中文人工调整待重新验证：{exc}')))
            return replace(layout,items=tuple(items))
        items=[]
        for original in layout.items:
            manual=self._manual_for_group(original.group_id,layout.display_mode)
            if manual is None: items.append(original); continue
            width=manual.text_bounds.max_x-manual.text_bounds.min_x
            try:
                lines,size,w,h=self._layout_engine._fit_text(original.translated_text,width,layout.page_height,manual.font_size)
            except (ValueError,RuntimeError) as exc:
                items.append(replace(original,placement='needs_layout_review',lines=(),
                    warnings=(*original.warnings,f'人工文本框无法容纳完整译文：{exc}')))
                continue
            b=manual.text_bounds
            bounds=self._layout_engine._clamped_bounds(b.min_x,b.min_y,w,h,self._page_geometry(layout.page_index))
            bounds=self._layout_engine._clamp_rotated_footprint(bounds,original.rotation_degrees,self._page_geometry(layout.page_index))
            items.append(replace(original,text_bounds=bounds,font_size=size,lines=lines,
                text_origin=None,mask_polygons=(),mask_decision=MaskDecision.NOT_REQUESTED,fallback_text_bounds=None))
        blank = BlankSpace(source_image, self._page_geometry(layout.page_index))
        checked=[]
        for item in items:
            if not item.lines:
                checked.append(item);continue
            if self._manual_for_group(item.group_id,layout.display_mode) is not None:
                box = footprint(item.text_bounds, item.rotation_degrees)
                other = [footprint(o.text_bounds,o.rotation_degrees) for o in items if o.group_id != item.group_id and o.lines and o.placement not in {'needs_layout_review','detail','unavailable','preserved'}]
                safe = blank.clear(box) and not any(self._layout_engine._overlap_area(box,b)>0 for b in other)
                item = replace(item, placement='inline' if safe else 'needs_layout_review',
                    warnings=tuple(w for w in item.warnings if not w.startswith('人工位置')) + (() if safe else ('人工位置可能遮挡工程信息，需重新调整；禁止正式导出',)))
            checked.append(item)
        return replace(layout, items=tuple(checked))

    # 人工位置按显示模式区分
    # 英文模式不应用中文位置
    # 旧chinese不继承双语拖动
    # 双语兼容旧裸group键
    # 查找不修改历史保存数据
    def _manual_for_group(self,group_id,mode=None):
        mode=mode or self._display_mode_name()
        # In-place Chinese replacement cannot inherit a moved bilingual caption.
        if mode=='english':return None
        if mode=='chinese':return self._manual_layouts.get((group_id,mode))
        return self._manual_layouts.get((group_id,mode),self._manual_layouts.get(group_id))

    # 障碍含原始与有效OCR区域
    # 自身原文区域按ID排除
    # 其他已布置中文也列为障碍
    # footprint包含阅读旋转
    # 用于历史中文编辑的保守核验
    def _chinese_edit_obstacles(self,layout,item):
        key=(self._document.info.id,layout.page_index)
        selection=self._target_cache.get(key)
        regions=(*self._ocr_cache.get(key,()),*(selection.raw_regions if selection else ()))
        source=[r.bounds for r in regions if r.id not in item.region_ids]
        captions=[footprint(other.text_bounds,other.rotation_degrees) for other in layout.items
            if other.group_id!=item.group_id and other.lines and other.placement=='inline']
        return (*source,*captions)

    # 调整参数均为源图坐标
    # 英文模式没有中文拖动目标
    # 当前正式UI仅双语走主分支
    # 历史中文先撤旧人工位置
    # 再生自动锚点避免连续走偏
    # 中文安全编辑单独复验
    # 该分支修改几何要求重新审核
    # 双语重排完整文字而不裁字
    # 字号限制到引擎允许范围
    # 宽度限定在页面范围
    # 旋转后包围框仍不可出页
    # 保存模式专属人工布局
    # 清除旧mask与fallback位置
    # 不因人工拖动获得遮挡权限
    # 重新预览时再检背景和碰撞
    # 成功后标dirty供项目恢复
    # 状态说明风险位置会阻止导出
    def adjust_annotation(self, group_id, *, x, y, font_size, width):
        if self._document is None: return False
        mode=self._display_mode_name()
        if mode=='english':
            self.state_status.setText('请切换中文或中英对照模式调整中文标注')
            return False
        key=(self._document.info.id,self.current_page)
        if mode=='chinese':
            # Regenerate the automatic anchor without the old manual edit so
            # repeated two-unit moves cannot walk Chinese away from its source.
            previous=self._manual_layouts.pop((group_id,mode),None)
            self._clear_product_layout(key)
            try:layout=self._layout_for_page(self.current_page,True,for_preview=True)
            finally:
                if previous is not None:self._manual_layouts[(group_id,mode)]=previous
            item=next((i for i in layout.items if i.group_id==group_id),None)
            if item is None:return False
            from .safe_annotation_edit import edit_chinese_item
            try:
                edited=edit_chinese_item(self._layout_engine,item,self._page_geometry(self.current_page),
                    self._preview_source_cache[key],x=x,y=y,font_size=font_size,width=width,
                    obstacles=self._chinese_edit_obstacles(layout,item))
            except (ValueError,RuntimeError) as exc:
                self._clear_product_layout(key);self.state_status.setText(f'无法调整中文：{exc}');return False
            self._manual_layouts[(group_id,mode)]=edited
            # Changed visual geometry requires explicit review again.
            result=self._translation_cache[key][group_id]
            self._translation_cache[key][group_id]=replace(result,provenance='machine')
            self._clear_product_layout(key);self._refresh_annotation_preview();self._update_annotation_actions()
            self.mark_project_dirty('chinese_layout')
            self.state_status.setText('中文调整已重新验证；保持源锚点和遮挡范围，需重新审核')
            return True
        layout = self._layout_for_page(self.current_page, False, for_preview=True)
        item = next((i for i in layout.items if i.group_id == group_id),None)
        if item is None: return False
        page = self._page_geometry(self.current_page)
        size = min(self._layout_engine.max_font_size, max(self._layout_engine.min_font_size, font_size))
        try:
            lines,size,w,h = self._layout_engine._fit_text(item.translated_text,max(size,min(width,page.width)),page.height,size)
        except (ValueError,RuntimeError) as exc:
            self.state_status.setText(f'无法调整标注：{exc}');return False
        bounds = self._layout_engine._clamp_rotated_footprint(self._layout_engine._clamped_bounds(x,y,w,h,page), item.rotation_degrees,page)
        self._manual_layouts[(group_id,self._display_mode_name())] = replace(item, text_bounds=bounds, font_size=size, lines=lines,
            text_origin=None,mask_polygons=(),mask_decision=MaskDecision.NOT_REQUESTED,fallback_text_bounds=None)
        self._clear_product_layout((page.document_id,page.page_index)); self._refresh_annotation_preview()
        self.mark_project_dirty('layout')
        self.state_status.setText('人工位置已锁定；保存项目可恢复。风险位置将阻止正式导出。')
        return True

    # 画布只报告组ID和新源位置
    # 字号与宽度取当前共享布局
    # 统一复用adjust_annotation验证
    # 不能由鼠标事件直接改绘制对象
    def _annotation_moved(self, group, x, y):
        layout=self._layout_for_page(self.current_page,False,for_preview=True)
        item=next((i for i in layout.items if i.group_id==group),None)
        if item: self.adjust_annotation(group,x=x,y=y,font_size=item.font_size,width=item.text_bounds.max_x-item.text_bounds.min_x)

    # 对话框输入X/Y/字号/宽度
    # 显示值来自当前源坐标布局
    # 逗号分隔四个浮点值
    # 非法数值由调整路径拒绝
    # 取消不改变锁定位置
    def choose_adjust_annotation(self):
        group=self._selected_group()
        if not group or self._document is None:return
        layout=self._layout_for_page(self.current_page,False,for_preview=True)
        item=next((i for i in layout.items if i.group_id==group),None)
        if item is None:return
        value,ok=QInputDialog.getText(self,'调整标注（源图坐标）','X, Y, 字号, 宽度',text=f'{item.text_bounds.min_x:.2f}, {item.text_bounds.min_y:.2f}, {item.font_size:.2f}, {item.text_bounds.max_x-item.text_bounds.min_x:.2f}')
        if ok:
            try:
                x,y,size,width=map(float,value.split(',')); self.adjust_annotation(group,x=x,y=y,font_size=size,width=width)
            except (ValueError,RuntimeError) as exc: self.state_status.setText(f'调整失败：{exc}')

    # 仅恢复当前选组自动布局
    # 按模式移除人工锁定
    # 双语兼容清理旧裸组键
    # 不删除中文编辑或审核记录
    # 下一次布局重新运行安全检查
    def reset_annotation(self):
        group=self._selected_group()
        if self._document is None:return
        self._manual_layouts.pop((group,self._display_mode_name()),None)
        if self._display_mode_name()=='bilingual':self._manual_layouts.pop(group,None)
        self._clear_product_layout((self._document.info.id,self.current_page));self._refresh_annotation_preview()
        self.mark_project_dirty('reset_layout')

    # 全部页面按零起始构造
    # 仍交统一串行队列执行
    # 不同时创建多个模型Worker
    def translate_all_pages(self):
        if self._document:self.start_page_batch(tuple(range(self._document.info.page_count)))

    # 用户页号采用一始显示
    # 起止范围包含结束页
    # 转换为零起始range后提交
    # 不在对话框中直接推理
    # 格式错误显示可读状态
    def choose_page_range(self):
        if self._document is None:return
        value,ok=QInputDialog.getText(self,'页范围','输入起止页，如 1-6',text=f'1-{self._document.info.page_count}')
        if ok:
            try:
                start,end=map(int,value.split('-')); self.start_page_batch(tuple(range(start-1,end)))
            except ValueError as exc:self.state_status.setText(f'页范围无效：{exc}')

    # 任一耗时任务忙时拒绝新批次
    # 批次不能为空或越界
    # dict去重并保留用户顺序
    # 总数以去重队列为准
    # 记录每页完成/空/失败
    # 开始后立即标项目dirty
    # 零延迟定时让出事件循环
    # 不在UI槽内阻塞循环推理
    def start_page_batch(self, pages):
        if self._document is None or self._ocr_task or self._translation_task or self._batch_active or self._mask_task or self._export_task:
            self.state_status.setText('后台任务尚未完成，请稍后开始多页处理');return False
        if not pages or any(p<0 or p>=self._document.info.page_count for p in pages):return False
        self._batch_queue=list(dict.fromkeys(pages));self._batch_total=len(self._batch_queue);self._batch_active=True
        self._batch_results={};self.mark_project_dirty('batch_started')
        QTimer.singleShot(0,self._advance_page_batch);return True

    # 队列推进只在主线程
    # 关闭或取消后不再取页
    # OCR/翻译/遮挡未完先让出
    # 结束显示实际已处理页数
    # batch_switch隔离用户手动切页
    # 自动切页不清剩余队列
    # 已有翻译缓存直接记完成
    # 不会无故重新运行OCR
    # 否则复用自动翻译链路
    # 无任务无结果按empty记账
    # 下一次推进用singleShot排队
    # 失败项不会凭空标成功
    # 页面结果与队列状态可保存
    def _advance_page_batch(self):
        if not self._batch_active or self._closing or self._document is None:return
        if self._ocr_task or self._translation_task or self._mask_task:return
        if not self._batch_queue:
            self._batch_active=False; self.state_status.setText(f'多页处理结束：{len(self._batch_results)}/{self._batch_total} 页；失败项保留状态');return
        page=self._batch_queue.pop(0)
        # show_page cancels outstanding single-page intent; no worker is active
        # here, and this queue retains ownership of its remaining page indices.
        self._batch_switch=True
        try:self.show_page(page)
        finally:self._batch_switch=False
        self._batch_active=True
        key=(self._document.info.id,page)
        if key in self._translation_cache:
            self._batch_results[str(page)]='complete';self.mark_project_dirty('batch_progress')
            QTimer.singleShot(0,self._advance_page_batch);return
        self.translate_current_page()
        if self._ocr_task is None and self._translation_task is None:
            self._batch_results[str(page)]='empty';self.mark_project_dirty('batch_progress')
            QTimer.singleShot(0,self._advance_page_batch)
        self.state_status.setText(f'多页进度 {len(self._batch_results)}/{self._batch_total}；正在处理第 {page+1} 页')

    # 只有活动批次接收完成通知
    # 用当前页目标判断是否为空
    # 失败优先于empty状态
    # 完成页号以字符串保存JSON
    # 标dirty保留中途进度
    # 推进先排队不递归调用
    # 若遮挡仍忙下一步会等待
    def _batch_page_finished(self, page, failed=False):
        if not self._batch_active or self._document is None:return
        key=(self._document.info.id,page)
        selection=self._target_cache.get(key)
        empty=selection is not None and not selection.candidate_entries
        self._batch_results[str(page)]='failed' if failed else 'empty' if empty else 'complete'
        self.mark_project_dirty('batch_progress')
        QTimer.singleShot(0,self._advance_page_batch)
