# mechanical_drawing_translator/project_store.py
# 项目文件只保存数据
# 类型标签使用封闭白名单
# 不执行pickle或动态导入
# 源图SHA先于坐标恢复核验
# 项目文件与原图分开存放
# 写盘使用同目录原子替换
# 业务迁移由ProductWorkflow负责
"""Versioned local projects: safe data-only decoding and atomic replacement.

The source identity is checked before changing the live document. Dataclass
tags are an allowlist, never import paths or executable pickle payloads.
"""
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .models import DocumentInfo, TextBounds, TextRegion
from .semantic_grouping import SourceSpan
from .translation_targets import TargetEntry, SelectionResult, Classification
from .translation_results import TranslationResult
from .annotation_models import AnnotationLayout, AnnotationItem, MaskDecision, PageGeometry

_TYPES = {c.__name__: c for c in (DocumentInfo, TextBounds, TextRegion, SourceSpan,
    TargetEntry, SelectionResult, Classification, TranslationResult, AnnotationLayout, AnnotationItem, PageGeometry)}


# 每次实际读取源字节
# 一MiB分块限制临时内存
# 不以文件名或mtime代替身份
# 同名内容变化也必须发现
# 文件访问错误由调用方处理
def source_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


# Path保留为显式数据标签
# tuple保留不可变容器语义
# mapping支持元组缓存键
# 字典不是直接转JSON对象
# dataclass记录类型和字段
# MaskDecision使用枚举值
# 递归展开嵌套业务模型
# 不序列化Qt句柄或控件
def encode(value):
    if isinstance(value, MaskDecision): return {'enum': value.value}
    if isinstance(value, Path): return {'path': str(value)}
    if is_dataclass(value):
        return {'type': type(value).__name__, 'fields': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, tuple): return {'tuple': [encode(v) for v in value]}
    if isinstance(value, dict): return {'mapping': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, list): return [encode(v) for v in value]
    return value


# 非有限浮点立即拒绝
# 只识别既定数据标签
# 未知类型不能动态导入
# dataclass构造仍执行自身校验
# tuple键可还原页缓存身份
# 还原模型不等于批准可信
# 源几何与来源在上层再核验
# 这里不做历史模式迁移
def decode(value):
    if isinstance(value, float) and not math.isfinite(value): raise ValueError('项目包含非有限数值')
    if isinstance(value, list): return [decode(v) for v in value]
    if not isinstance(value, dict): return value
    if 'enum' in value: return MaskDecision(value['enum'])
    if 'path' in value: return Path(value['path'])
    if 'tuple' in value: return tuple(decode(v) for v in value['tuple'])
    if 'mapping' in value: return {decode(k): decode(v) for k, v in value['mapping']}
    if value.get('type') not in _TYPES: raise ValueError('项目包含未知数据类型')
    cls = _TYPES[value['type']]
    return cls(**{k: decode(v) for k, v in value['fields'].items()})


# 临时文件与目标同目录
# 保证replace处于同卷
# UTF-8保留中文内容
# allow_nan关闭非法JSON数值
# flush后fsync提交文件内容
# 旧项目另留bak副本
# 随后原子替换正式目标
# 任何失败清理临时文件
# 不会用原图路径作默认目标
# 源保护还需调用方检查
# 备份存在不等于多版本管理
def write_project(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        if path.exists(): shutil.copy2(path, path.with_suffix(path.suffix + '.bak'))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


# JSON非法数值也要拒绝
# schema与kind必须同时匹配
# 版本不兼容保留旧文件
# 加载前核验实际源SHA
# 源改变禁止复用旧坐标
# 校验通过仅返回数据
# 不直接替换窗口中的文档
def read_project(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('项目包含非法数值：'+value)))
    if payload.get('schema_version') != 1 or payload.get('kind') != 'mdt_project':
        raise ValueError('项目版本不兼容；请保留原文件并使用对应版本软件')
    source = Path(payload['source']['path'])
    if source_sha256(source) != payload['source']['sha256']:
        raise ValueError('原图纸已发生变化，禁止复用旧坐标；请重新识别并另存项目')
    return payload
