# mechanical_drawing_translator/ort_graph_cache.py
"""Opt-in, local-machine ORT graph serialization with source-model fallback.

Only ORT's existing ORT_ENABLE_ALL transformations are persisted. This is not a
weight conversion, inference-result cache, or model replacement. The caller must
keep the original model available and include all effective session settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path


# 流式读取模型内容哈希
# 不缓存OCR输入或输出
# 模型存在仍是必要条件
def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 图优化可能依赖CPU指令
# 身份包含主机和操作系统
# ORT版本与构建信息也绑定
# 原模型名称和SHA均记录
# Session有效配置完整进入键
# 不跨主机盲目复用图文件
# 环境变更会自然生成新键
def _identity(ort, source: Path, configuration: dict) -> dict:
    # Layout optimizations can depend on the CPU instruction set. Be conservative:
    # do not reuse artifacts across hosts, OS versions, CPU models or ORT builds.
    return {
        "schema": 1,
        "source_name": source.name,
        "source_sha256": _sha256(source),
        "ort_version": ort.__version__,
        "ort_build": ort.get_build_info(),
        "host": platform.node(),
        "system": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "cpu_count": os.cpu_count(),
        "configuration": configuration,
    }


# 只有显式缓存目录才进此路
# 原模型始终保持可用
# 缓存不是权重转换
# 也不是图纸结果缓存
# 命中先验manifest身份
# 再验优化图内容SHA
# 通过后关闭重复图优化
# ORT仍可能拒绝合法manifest图
# 此时撤销提交标记并回原模型
# 每请求原模型回退只启动一次
# 未命中写唯一临时图
# 首请求承担构图和序列化成本
# 模型SHA建前建后复核
# 期间模型变化不发布缓存
# 先替换图文件
# 后替换manifest作提交标记
# 并发交错由哈希检查拒绝
# 发布失败仍返回可用Session
# 不为写盘失败重复建模型
# 回退恢复原图优化参数
# 原模型本身失败继续抛出
# finally清理残留临时文件
# 诊断事件只记模型与成本
# 不改变正常OCR文本契约
def cached_cpu_session(ort, source: Path, options, cache_root: Path,
                       configuration: dict, events: list[dict] | None = None):
    """Return one session; cache failures retry the unchanged source configuration.

    The first request pays optimization, serialization and hash costs in full and
    returns that same source session. On a validated hit, only graph transforms
    are disabled. No OCR inputs, outputs, metadata or weights are rewritten.
    """
    started = time.perf_counter()
    temporary_paths: list[Path] = []
    original_retry_started = False

    # 缓存状态供性能诊断
    # 耗时从本次调用起算
    # 不写入任何图纸字面
    def record(status: str, reason: str = "") -> None:
        if events is not None:
            events.append({
                "model": source.name,
                "status": status,
                "reason": reason,
                "seconds": time.perf_counter() - started,
            })

    # 清除优化图输出路径
    # 恢复完整原图优化级别
    # 标记已尝试原模型回退
    # 防止外层再次重复初始化
    def original_session():
        nonlocal original_retry_started
        original_retry_started = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.optimized_model_filepath = ""
        return ort.InferenceSession(str(source), sess_options=options,
                                    providers=["CPUExecutionProvider"])

    try:
        identity = _identity(ort, source, configuration)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        cache_root.mkdir(parents=True, exist_ok=True)
        graph = cache_root / f"{key}.onnx"
        manifest_path = cache_root / f"{key}.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (manifest["identity"] == identity
                     and graph.is_file()
                     and manifest["optimized_sha256"] == _sha256(graph))
        except (OSError, ValueError, TypeError, KeyError):
            valid = False
        if valid:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            try:
                session = ort.InferenceSession(str(graph), sess_options=options,
                                               providers=["CPUExecutionProvider"])
            except Exception as error:
                # A graph accepted by the manifest can still be rejected by ORT.
                # Do not rebuild repeatedly during this request or mask a source
                # error. Invalidate metadata so the next request can rebuild.
                try:
                    manifest_path.unlink(missing_ok=True)
                except OSError:
                    pass
                session = original_session()
                record("fallback", f"cached_session:{type(error).__name__}")
                return session
            record("hit")
            return session

        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{key}-", suffix=".onnx", dir=cache_root)
        os.close(descriptor)
        temporary_graph = Path(temporary_name)
        temporary_paths.append(temporary_graph)
        options.optimized_model_filepath = str(temporary_graph)
        try:
            session = ort.InferenceSession(str(source), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        except Exception as error:
            session = original_session()
            record("fallback", f"serialization:{type(error).__name__}")
            return session

        # The session is already usable. Publication failures must not make us
        # pay another model initialization or turn a successful OCR into failure.
        try:
            if _sha256(source) != identity["source_sha256"]:
                record("uncached", "source_changed_during_build")
                return session
            manifest = {"identity": identity, "optimized_sha256": _sha256(temporary_graph)}
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{key}-", suffix=".json", dir=cache_root)
            temporary_manifest = Path(temporary_name)
            temporary_paths.append(temporary_manifest)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True)
            # Metadata is the commit marker. Unique temp files plus replacement
            # keep interrupted/concurrent writers from exposing partial graphs;
            # an interleaved graph/manifest mismatch is rejected by the hash.
            os.replace(temporary_graph, graph)
            os.replace(temporary_manifest, manifest_path)
        except Exception as error:
            record("uncached", f"publication:{type(error).__name__}")
            return session
        record("built")
        return session
    except Exception as error:
        if original_retry_started:
            raise
        session = original_session()
        record("fallback", f"cache_setup:{type(error).__name__}")
        return session
    finally:
        for path in temporary_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
