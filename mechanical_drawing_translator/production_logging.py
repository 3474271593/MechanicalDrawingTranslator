# mechanical_drawing_translator/production_logging.py
"""Local metadata journal; a capacity budget never shortens the retention floor.

One UTF-8 JSONL file per calendar day makes retention inspectable without parsing
user content. Storage errors are contained so a full/read-only disk cannot crash
OCR or prevent a user from saving their project to another drive.
"""
from __future__ import annotations

import json
import os
import threading
from time import perf_counter
from datetime import date, datetime, timezone
from pathlib import Path, PureWindowsPath


# 日志存用户数据目录
# 不写源码或冻结包目录
# 继承当前用户目录权限
# 不宣称独立ACL或加密
def default_log_directory() -> Path:
    return Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'MechanicalDrawingTranslator' / 'logs'


# 只保留白名单元数据
# 不记录图纸OCR或中文全文
# 文件基本名仍可能敏感
# 按已确认本地保留要求保存
# 容量预算不能缩短180天
# 锁允许Worker信号安全记账
class ProductionLog:
    """Append metadata and a sanitized basename, never full paths or content.

    The journal inherits the current user's local application-data permissions.
    Filenames remain local for incident correlation; drawing text is excluded.
    A basename may still be sensitive, so it is retained only under the same
    explicit 180-day floor as other required operation metadata.
    """
    FIELDS = frozenset({'page_index', 'page_count', 'region_count', 'result_count',
                        'elapsed_ms', 'status', 'error_type', 'model_version',
                        'knowledge_version', 'runtime_identity', 'completed',
                        'failed', 'cancelled', 'bytes', 'retention_days', 'filename'})

    # 保留期不能低于180天
    # 容量必须为正
    # today可注入以核验过期边界
    # RLock允许event内调用cleanup
    # last_error仅保留异常类型
    # 不存可能包含路径的完整异常
    def __init__(self, directory=None, *, retention_days=180,
                 capacity_bytes=512 * 1024 * 1024, today=date.today):
        if retention_days < 180:
            raise ValueError('production logs must be retained for at least 180 days')
        if capacity_bytes <= 0:
            raise ValueError('capacity must be positive')
        self.directory = Path(directory) if directory is not None else default_log_directory()
        self.retention_days = retention_days
        self.capacity_bytes = capacity_bytes
        self.today = today
        self.last_error = ''
        self._lock = threading.RLock()
        self._last_cleanup = None

    # 只识别events日期日志
    # 未知文件不碰
    # 未来日期也不删除
    # 严格超过保留天数才清理
    # 容量超标保留近期日志
    # 报告超标供外部处理
    # 不为限容量破坏保留底线
    # 目录/删除错误局部吸收
    # last_cleanup跟踪本地日期
    def cleanup(self):
        """Delete only recognized journals older than the configured floor.

        Over-budget recent logs remain intact. The caller can show/report the
        capacity condition instead of silently violating contractual retention.
        Unknown files and future-dated journals are never deleted.
        """
        report = {'removed': 0, 'bytes': 0, 'over_capacity': False}
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                for path in self.directory.glob('events-*.jsonl'):
                    try:
                        day = date.fromisoformat(path.stem.removeprefix('events-'))
                    except ValueError:
                        continue
                    if (self.today() - day).days > self.retention_days:
                        path.unlink()
                        report['removed'] += 1
                    else:
                        report['bytes'] += path.stat().st_size
                report['over_capacity'] = report['bytes'] > self.capacity_bytes
                self._last_cleanup = self.today()
            except OSError as exc:
                self.last_error = type(exc).__name__
                report['error_type'] = self.last_error
        return report

    # 每日首次事件尝试清理
    # 事件时间戳使用UTC
    # 日志文件按本地日期分组
    # 字段只取基础标量类型
    # filename移除Windows父路径
    # 控制字符变下划线
    # 长度上限255避免异常扩张
    # JSONL保证一事件一行
    # allow_nan拒绝非有限值
    # 磁盘满或只读返回False
    # 不能让日志错误阻止保存图纸
    def event(self, name: str, **metadata) -> bool:
        """Return a status instead of raising on an unavailable journal drive."""
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                if self._last_cleanup != self.today():
                    self.cleanup()
                row = {'timestamp': datetime.now(timezone.utc).isoformat(),
                       'event': name, **{key: value for key, value in metadata.items()
                                        if key in self.FIELDS and isinstance(value, (str, int, float, bool, type(None)))}}
                if 'filename' in row:
                    # Windows separators must be stripped even in cross-platform
                    # tests. Control characters cannot create forged log lines.
                    basename = PureWindowsPath(str(row['filename'])).name
                    row['filename'] = ''.join(c if c.isprintable() else '_' for c in basename)[:255]
                path = self.directory / f'events-{self.today().isoformat()}.jsonl'
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                return True
            except (OSError, TypeError, ValueError) as exc:
                self.last_error = type(exc).__name__
                return False


_journal = None


# 按进程懒建单一日志器
# 环境容量非法回默认预算
# 容量至少一字节
# 仍不影响180天保留门槛
def journal() -> ProductionLog:
    global _journal
    if _journal is None:
        try:
            capacity = max(1, int(os.environ.get('MDT_LOG_CAPACITY_BYTES', 512 * 1024 * 1024)))
        except ValueError:
            capacity = 512 * 1024 * 1024
        _journal = ProductionLog(capacity_bytes=capacity)
    return _journal


def record_event(name: str, **metadata) -> bool:
    return journal().event(name, **metadata)


# 只观察已有信号
# 不接管Worker生命周期
# starts键含任务种类和ID
# OCR与翻译ID不会互相覆盖
# 记录容器数量不序列化结果
# 取消任务仍有finished计时
# 该计时包含线程排队等待
# 不能当纯模型推理性能
# 兼容公共与私有会话属性
# 错误消息不写入日志正文
def attach_window_logging(window):
    """Observe existing worker signals without taking ownership of the workers."""
    # QObject signals deliver metadata on the UI thread; full payloads are never
    # serialized. Connections survive file changes and naturally die with Qt.
    starts = {}
    # 优先从不可变请求取路径
    # OCR请求通常带path
    # 翻译请求通常带pdf_path
    # 最终只由日志器保存基本名
    def filename(request):
        path = getattr(request, 'path', None) or getattr(request, 'pdf_path', None)
        if path is None:
            document = getattr(window, '_document', None)
            path = getattr(getattr(document, 'info', None), 'path', '')
        return str(path or '')
    # 在提交信号处开始计时
    # 记录页号而非OCR内容
    # 键绑定当前会话请求序号
    def started(request, prefix):
        starts[(prefix, request.request_id)] = perf_counter()
        record_event(prefix + '_started', page_index=request.page_index, filename=filename(request))
    # 从配对开始事件计算耗时
    # 缺开始记录就保存空值
    # 不编造缺失的性能数据
    def finished(request, prefix):
        start = starts.pop((prefix, request.request_id), None)
        record_event(prefix + '_finished', page_index=request.page_index, filename=filename(request),
                     elapsed_ms=round((perf_counter()-start)*1000, 3) if start is not None else None)
    # 仅汇总区域数或结果数
    # 不展开文本或图像
    # complete事件不代表专业验收
    def completed(request, result, prefix):
        # Count only container sizes; never serialize recognition payloads.
        counts = {}
        if prefix == 'ocr' and hasattr(result, 'pages'):
            counts['region_count'] = sum(len(page.regions) for page in result.pages)
        elif prefix == 'translation' and isinstance(result, (list, tuple, dict)):
            counts['result_count'] = len(result)
        record_event(prefix + '_complete', page_index=request.page_index,
                     filename=filename(request), status='complete', **counts)
    for attribute, prefix in (('_ocr_session', 'ocr'), ('_translation_session', 'translation'),
                              ('ocr_session', 'ocr'), ('translation_session', 'translation')):
        session = getattr(window, attribute, None)
        if session is None:
            continue
        if hasattr(session, '_execute'):
            session._execute.connect(lambda request, p=prefix: started(request, p))
        if hasattr(session, 'finished'):
            session.finished.connect(lambda request, p=prefix: finished(request, p))
        if hasattr(session, 'result_ready'):
            session.result_ready.connect(lambda request, result, p=prefix: completed(request, result, p))
        if hasattr(session, 'failed'):
            session.failed.connect(lambda request, message, p=prefix:
                record_event(p + '_failed', page_index=request.page_index, status='failed'))
