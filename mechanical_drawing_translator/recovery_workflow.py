# mechanical_drawing_translator/recovery_workflow.py
"""Qt lifecycle hooks for explicit local recovery, separate from formal saves."""
from pathlib import Path
from PySide6.QtCore import QTimer,QThread,Qt
from PySide6.QtWidgets import QMessageBox
from . import project_store
from .recovery_store import RecoveryDraftStore,project_payload_from_snapshot


class _RecoveryWriteTask(QThread):
    """One detached snapshot, with no UI/document objects in the worker."""
    # 任务保存独立数据快照
    # 不持有QWidget或源句柄
    # generation标识快照修订
    # document_id隔离切图
    # formal_path和base_sha固定
    # 不能后台读取可变窗口状态
    def __init__(self,store,snapshot,formal_path,base_sha,generation,parent):
        super().__init__(parent)
        self.store=store;self.snapshot=snapshot;self.formal_path=formal_path
        self.base_sha=base_sha;self.generation=generation;self.error=None
        self.document_id=snapshot['document'].id

    # 后台编码后写恢复副本
    # 失败文本留在任务对象
    # 不得让异常逃出QThread.run
    # 不把失败标成正式已保存
    # 完成后主线程读取error
    def run(self):
        try:
            payload=project_payload_from_snapshot(self.snapshot)
            self.store.write(payload,formal_path=self.formal_path,base_formal_sha=self.base_sha)
        except Exception as exc:
            # A background serialization/I/O error stays visible and dirty;
            # it must never escape QThread.run or bless a formal save.
            self.error=str(exc)


class RecoveryWorkflow:
    # dirty跟踪内存未保存修改
    # base_sha跟踪正式项目版本
    # loading抑制恢复期间脏标记
    # pending保留旧草稿选择权
    # generation每次修订递增
    # 500毫秒单次定时合并连改
    # 定时器在Qt主线程运行
    # 后台写任务同一时刻仅一个
    def _initialize_recovery_workflow(self):
        self._recovery_store=RecoveryDraftStore()
        self._project_dirty=False;self._project_base_sha=None
        self._loading_project=False;self._recovery_pending_decision=False
        self._recovery_task=None;self._recovery_generation=0
        self._recovery_timer=QTimer(self);self._recovery_timer.setSingleShot(True)
        self._recovery_timer.setInterval(500)
        self._recovery_timer.timeout.connect(self._start_recovery_draft)

    # 先停止延迟写
    # 再等待旧目录写任务结束
    # 之后更换本地存储位置
    # 测试可注入隔离目录
    # 新目录旧草稿仍要显式决定
    def configure_recovery_directory(self,directory):
        """Inject a local directory before processing, notably for isolated tests."""
        self._recovery_timer.stop();self._wait_recovery_write()
        self._recovery_store=RecoveryDraftStore(directory)
        self._recovery_pending_decision=self.recovery_draft_status()!='missing'

    # 普通打开新源后重置
    # 项目载入自行恢复正式身份
    # 先等待旧写任务结束
    # 递增generation使旧快照过期
    # 清除旧正式项目路径
    # 检测新源是否已有恢复草稿
    # 不自动覆盖发现的旧证据
    def reset_project_recovery_state(self):
        """Call after a new source opens; project/draft restore owns its own state."""
        if self._loading_project:return
        self._wait_recovery_write();self._recovery_generation+=1
        self._recovery_timer.stop();self._project_dirty=False;self._project_base_sha=None
        self._project_path=None
        self._recovery_pending_decision=self.recovery_draft_status()!='missing'

    # 无文档时没有可保存状态
    # 恢复过程中不标新修改
    # 每次修改递增generation
    # 已有待决定草稿不自动写
    # 避免新处理抹掉崩溃证据
    # 允许写时重启防抖计时器
    def mark_project_dirty(self,reason='change'):
        if self._document is None or self._loading_project:return
        self._project_dirty=True
        self._recovery_generation+=1
        # Existing crash evidence must await the explicit restore/discard choice.
        # Processing a new page must never overwrite it before that choice.
        if not self._recovery_pending_decision:self._recovery_timer.start()

    # 仅用于明确放弃当前会话
    # 调用方随后关闭或替换源
    # 这里不回滚内存中的显示
    # 先挡住待写和在途快照
    # 只清当前dirty状态
    # 旧崩溃草稿仍在磁盘
    # 若替换失败需重新标dirty
    def discard_unsaved_live_changes(self):
        """Abandon this session only after an explicit close/switch decision.

        This lifecycle operation does not revert the displayed in-memory data;
        the caller must immediately close or replace the source. Older crash
        evidence and formal reviewed files are deliberately left on disk.
        """
        self._recovery_timer.stop();self._wait_recovery_write();self._project_dirty=False

    # 只收集当前任务对象
    # 错误保持可见并保留dirty
    # 写入期间可能又有新修改
    # 同文档且generation更新
    # 才重启下一轮防抖写
    # 切图后不替旧文档重排任务
    # deleteLater在主线程回收
    def _collect_recovery_write(self,task,*,schedule):
        if task is None or self._recovery_task is not task:return
        self._recovery_task=None
        if task.error:self.state_status.setText(f'恢复草稿写入失败；请另存正式项目：{task.error}')
        if (schedule and self._project_dirty and not self._recovery_pending_decision
            and self._document is not None and self._document.info.id==task.document_id
            and self._recovery_generation>task.generation):self._recovery_timer.start()
        task.deleteLater()

    # 生命周期屏障先等在途写
    # 否则删除草稿后旧线程可复活它
    # 同步收集时禁止再次排写
    # 切源/正式保存/丢弃均复用
    def _wait_recovery_write(self):
        """Lifecycle barrier: an older worker cannot resurrect a deleted draft."""
        task=self._recovery_task
        if task is not None:
            task.wait();self._collect_recovery_write(task,schedule=False)

    # finished信号回主线程
    # sender用于核对任务身份
    # 异步完成允许续写新修订
    def _recovery_write_finished(self):
        self._collect_recovery_write(self.sender(),schedule=True)

    # 只在有脏数据且空闲时启动
    # 旧草稿待决定则不覆盖
    # 快照必须在GUI线程捕获
    # 只把脱离控件的数据交后台
    # 任务绑定当前generation
    # 完成信号显式排队回主线程
    # 快照失败提示另存正式项目
    def _start_recovery_draft(self):
        if (self._document is None or not self._project_dirty or self._loading_project
            or self._recovery_pending_decision or self._recovery_task is not None):return
        try:
            snapshot=self._project_snapshot()
            task=_RecoveryWriteTask(self._recovery_store,snapshot,self._project_path,
                self._project_base_sha,self._recovery_generation,self)
            self._recovery_task=task
            task.finished.connect(self._recovery_write_finished,Qt.ConnectionType.QueuedConnection)
            task.start()
        except (OSError,ValueError,TypeError,KeyError) as exc:
            self.state_status.setText(f'恢复草稿快照失败；请另存正式项目：{exc}')

    # 关闭或切源前同步落草稿
    # 先停防抖并等旧任务
    # 无修改可以立即成功
    # 旧草稿未决定则拒绝覆盖
    # 同步写仍不等于正式保存
    # 失败返回False让UI保护修改
    # 不得把等待时间当用户同意
    def flush_recovery_draft(self):
        self._recovery_timer.stop()
        self._wait_recovery_write()
        if self._document is None or not self._project_dirty:return True
        if self._recovery_pending_decision:
            self.state_status.setText('已有恢复草稿；请先恢复或丢弃，再保存新的恢复草稿');return False
        try:
            self._recovery_store.write(self._project_payload(),formal_path=self._project_path,
                base_formal_sha=self._project_base_sha)
            return True
        except (OSError,ValueError,TypeError,KeyError) as exc:
            self.state_status.setText(f'恢复草稿写入失败；请另存正式项目：{exc}');return False

    # 没有源时不存在恢复对象
    # 状态由实际本地信封核验
    # 不会因查询触发恢复
    def recovery_draft_status(self):
        if self._document is None:return 'missing'
        return self._recovery_store.status(self._document.info.path)

    # 后台任务忙时拒绝恢复
    # 先等待草稿写入结束
    # read先执行源与正式SHA检查
    # 再走普通load_project验证
    # 恢复只替换内存状态
    # 正式项目保持原字节
    # 恢复后dirty仍为True
    # 待用户另存正式审核结果
    # 旧machine来源不会自动批准
    # 异常通过状态栏和可选弹窗显示
    def restore_recovery_draft(self,show_error=True):
        if self._document is None:return False
        if self._ocr_task or self._translation_task or self._export_task or self._mask_task:return False
        self._recovery_timer.stop();self._wait_recovery_write()
        try:
            envelope=self._recovery_store.read(self._document.info.path)
            if not self.load_project(None,show_error=show_error,_payload=envelope['payload'],
                _formal_path=envelope.get('formal_path')):return False
            self._project_base_sha=envelope.get('base_formal_sha256')
            self._project_dirty=True;self._recovery_pending_decision=False
            self.state_status.setText('恢复草稿已载入；正式项目未改写，审核和机器来源保持原状态')
            return True
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.state_status.setText(f'恢复草稿拒绝载入：{exc}')
            if show_error:QMessageBox.warning(self,'无法恢复草稿',str(exc))
            return False

    # 先等待在途写完再删除
    # 防止旧Worker重建草稿
    # 只清pending决策状态
    # 仍有新修改则排新草稿
    # 正式项目内容不受影响
    def discard_recovery_draft(self):
        if self._document is None:return False
        self._recovery_timer.stop();self._wait_recovery_write()
        try:
            discarded=self._recovery_store.discard(self._document.info.path)
            self._recovery_pending_decision=False
            if self._project_dirty:self._recovery_timer.start()
            self.state_status.setText('已丢弃恢复草稿；正式项目保持不变')
            return discarded
        except OSError as exc:
            self.state_status.setText(f'恢复草稿丢弃失败：{exc}');return False

    # 只有明确UI打开时提示
    # 可用草稿提供恢复按钮
    # 冲突草稿只能丢弃或稍后
    # 稍后处理继续保留旧证据
    # 没有静默合并正式审核
    # 恢复不是磁盘正式保存
    def choose_recovery_draft(self):
        """Explicit UI choice; conflicts can be inspected/discarded, never merged."""
        status=self.recovery_draft_status()
        if status=='missing':return
        self._recovery_pending_decision=True
        box=QMessageBox(self);box.setWindowTitle('发现本地恢复草稿')
        box.setText('发现未保存的本地修改。恢复仅载入内存，不覆盖正式项目。' if status=='available'
            else f'恢复草稿存在冲突（{status}），禁止复用旧坐标或覆盖较新的正式审核。')
        restore=(box.addButton('恢复草稿',QMessageBox.ButtonRole.AcceptRole) if status=='available' else None)
        discard=box.addButton('丢弃草稿',QMessageBox.ButtonRole.DestructiveRole)
        box.addButton('稍后处理',QMessageBox.ButtonRole.RejectRole);box.exec()
        if restore is not None and box.clickedButton()==restore:self.restore_recovery_draft()
        elif box.clickedButton()==discard:self.discard_recovery_draft()

    # 正式保存完成后刷新base_sha
    # 先等旧草稿任务安全结束
    # 清当前未保存修改标志
    # 旧崩溃草稿未决定仍保留
    # 保存当前会话不替用户做选择
    # 删除失败留到下次冲突核查
    # 不能因此撤销已成功的正式保存
    def _project_saved_for_recovery(self,path):
        self._recovery_timer.stop();self._wait_recovery_write();self._project_path=Path(path)
        self._project_base_sha=project_store.source_sha256(path)
        self._project_dirty=False
        # Saving the current live session is not a restore/discard decision for
        # an older crash draft encountered when this source was opened.
        if self._recovery_pending_decision:return
        try:self._recovery_store.discard(self._document.info.path)
        except OSError:self.state_status.setText('正式项目已保存；旧恢复草稿未能删除，后续恢复会执行冲突检查')
