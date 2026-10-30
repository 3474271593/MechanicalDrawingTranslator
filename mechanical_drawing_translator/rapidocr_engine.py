# mechanical_drawing_translator/rapidocr_engine.py
"""使用安装包内的 RapidOCR ONNX 权重执行纯 CPU 离线识别。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .ocr_service import DetectedText, OcrError, english_candidate
from .resource_paths import package_resource


DEFAULT_CPU_THREADS = 6


# 只注册CPU执行提供器
# 关闭线程自旋减少空闲占用
# intra_op负责算子内部并行
# inter_op固定单线程
# 图优化使用ORT_ENABLE_ALL
# arena由候选配置决定
# memory pattern可单独关闭
# 缓存分支复用同一配置
# 有效参数进入缓存身份
# 无缓存直接读原ONNX
# 这里不使用GPU回退
# 缺少本地资源直接报错
def _create_cpu_session(model_path: Path, threads: int, *, use_arena: bool = False,
                        use_memory_pattern: bool = True,
                        optimized_graph_cache_dir: str | Path | None = None,
                        cache_events: list[dict] | None = None):
    """Create one CPU-only ORT session with the measured production settings."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.enable_cpu_mem_arena = use_arena
    options.enable_mem_pattern = use_memory_pattern
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if optimized_graph_cache_dir is not None:
        from .ort_graph_cache import cached_cpu_session

        configuration = {
            "providers": ["CPUExecutionProvider"],
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "execution_mode": "ORT_SEQUENTIAL",
            "intra_op_num_threads": threads,
            "inter_op_num_threads": 1,
            "enable_cpu_mem_arena": use_arena,
            "enable_mem_pattern": use_memory_pattern,
            "intra_op_allow_spinning": "0",
            "inter_op_allow_spinning": "0",
        }
        return cached_cpu_session(ort, Path(model_path), options,
                                  Path(optimized_graph_cache_dir), configuration, cache_events)
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


class RapidOcrCpuEngine:
    name = "rapidocr-onnxruntime-cpu"
    MODEL_FILES = (
        "PP-OCRv6_det_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "PP-OCRv6_rec_small.onnx",
    )

    # 检测/方向/识别共三个模型
    # 默认定位受控模型目录
    # 显式model_dir覆盖包路径
    # 先检查文件完整再建Session
    # 缺模型不自动联网下载
    # 常规CPU线程默认六个
    # performance或compact选择八个
    # 三Session并行初始化
    # 配置导入与构图过程重叠
    # 首次请求承担全部初始化成本
    # 不能把预备加载排出冷计时
    # compact关闭动态形状pattern
    # CPU arena仍继续复用
    # RapidOCR禁用CUDA和DML
    # 最大边长参数保持4000
    # 自建Session经配置注入
    # OmegaConf需临时允许对象
    # 只在验证后设置session
    # 窗口复用该引擎跨页识别
    # 参数开关不证明性能已达标
    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        threads: int = DEFAULT_CPU_THREADS,
        performance_mode: bool = False,
        compact_memory: bool = False,
        optimized_graph_cache_dir: str | Path | None = None,
    ):
        import rapidocr

        # 默认权重归属项目，防止安装包模型变化或未来冻结漏带；显式目录仍可注入。
        # 与旧wheel相同的三份权重按SHA迁移，不改变识别参数或推理会话。
        root = Path(model_dir) if model_dir is not None else package_resource('models', 'ocr')
        missing = [name for name in self.MODEL_FILES if not (root / name).is_file()]
        if missing:
            raise OcrError(f"本地 RapidOCR 模型缺失：{', '.join(missing)}；请恢复项目 OCR 资源或显式指定模型目录")
        if threads < 1:
            raise ValueError("CPU 线程数必须大于 0")
        self.model_dir = root
        self.optimized_graph_cache_events: list[dict] = []
        cache_options = ({
            "optimized_graph_cache_dir": optimized_graph_cache_dir,
            "cache_events": self.optimized_graph_cache_events,
        } if optimized_graph_cache_dir is not None else {})
        if performance_mode or compact_memory:
            # Opt-in measured CPU profile. Initialization stays inside the first
            # worker request; it is not moved before the cold-start boundary.
            # Import ORT once before concurrent native constructors, and overlap
            # their graph setup with RapidOCR's Python/config module import.
            from concurrent.futures import ThreadPoolExecutor
            import onnxruntime  # noqa: F401

            threads = 8
            with ThreadPoolExecutor(max_workers=3) as pool:
                # Dynamic full-page and bounded-ROI shapes otherwise retain a
                # large memory-pattern high water mark. Keep CPU arena reuse,
                # all graph/input settings and the stage2 initialization overlap.
                memory_options = ({"use_memory_pattern": False} if compact_memory else {})
                futures = [pool.submit(_create_cpu_session, root / name, threads,
                                       use_arena=True, **memory_options, **cache_options)
                           for name in self.MODEL_FILES]
                rapid_class = rapidocr.RapidOCR
                sessions = [future.result() for future in futures]
        else:
            sessions = [_create_cpu_session(root / name, threads, **cache_options)
                        for name in self.MODEL_FILES]
            rapid_class = rapidocr.RapidOCR
        self._ocr = rapid_class(params={
            "Det.model_path": str(root / self.MODEL_FILES[0]),
            "Cls.model_path": str(root / self.MODEL_FILES[1]),
            "Rec.model_path": str(root / self.MODEL_FILES[2]),
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_dml": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Global.max_side_len": 4000,
            "Global.log_level": "error",
        })
        # RapidOCR accepts custom sessions, but its public params parser rejects
        # non-primitive OmegaConf values. Inject them after normal validation.
        from omegaconf import flag_override

        for section_name, session in zip(("Det", "Cls", "Rec"), sessions):
            section = getattr(self._ocr.cfg, section_name)
            with flag_override(section, "allow_objects", True):
                section.session = session

    # 整页启用检测方向和识别
    # 推理后核验实际提供器
    # 缺任一输出返回空结果
    # 文本/置信度/框按顺序配对
    # 引擎层先做英文候选过滤
    # 该过滤仍使用默认策略
    # 混框能否保留须看实际前后门
    # 多边形转成普通float
    # 竖向框另判阅读方向
    # 结果坐标仍是输入图像像素
    def recognize(self, image: Image.Image) -> list[DetectedText]:
        output = self._ocr(image, use_det=True, use_cls=True, use_rec=True)
        self._verify_cpu()
        if output.boxes is None or output.txts is None or output.scores is None:
            return []
        detections = []
        for box, text, score in zip(output.boxes, output.txts, output.scores):
            if not english_candidate(text):
                continue
            polygon = tuple((float(x), float(y)) for x, y in box)
            angle = self._vertical_orientation(image, polygon, text)
            detections.append(DetectedText(text, float(score), polygon, angle))
        return detections

    # 只识别有界且已扶正ROI
    # 默认关闭文字检测
    # 同时关闭方向分类器
    # 保存原开关避免污染整页
    # finally恢复三项状态
    # 无检测框时使用整张ROI框
    # 局部方向固定为零
    # 位置解释由调用者负责
    # 本方法不做原页坐标映射
    def recognize_local(self, image: Image.Image, *, detect: bool = False) -> list[DetectedText]:
        """Read one bounded upright ROI without changing production direction handling."""
        previous=(self._ocr.use_det,self._ocr.use_cls,self._ocr.use_rec)
        try:
            output=self._ocr(image,use_det=detect,use_cls=False,use_rec=True)
            self._verify_cpu()
            if output.txts is None or output.scores is None:
                return []
            boxes=getattr(output,"boxes",None)
            if boxes is None:
                boxes=[((0,0),(image.width,0),(image.width,image.height),(0,image.height))]
            return [DetectedText(text,float(score),tuple((float(x),float(y)) for x,y in box),0.0)
                    for text,score,box in zip(output.txts,output.scores,boxes)
                    if english_candidate(text)]
        finally:
            self._ocr.use_det,self._ocr.use_cls,self._ocr.use_rec=previous

    # 仅对明显竖长框加判
    # 长度不足两字符不处理
    # 高度需大于宽度1.7倍
    # 边缘补白后切出ROI
    # 分别旋转90和270度识别
    # 比较两方向置信度
    # 最优低于0.5则不下结论
    # 返回的是源阅读角度
    # 原始四边形并未旋转
    # 额外识别属于真实耗时
    def _vertical_orientation(self, image, polygon, text) -> float | None:
        """轴对齐竖排框通过两种旋转后的识别置信度判定阅读方向。"""
        if len(text.strip()) < 2:
            return None
        xs, ys = zip(*polygon)
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if height <= width * 1.7:
            return None
        pad = max(4, round(width * 0.1))
        crop = image.crop((max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
                           min(image.width, int(max(xs)) + pad),
                           min(image.height, int(max(ys)) + pad)))
        options = []
        for pil_angle, source_angle in ((90, 90.0), (270, -90.0)):
            result = self._ocr(crop.rotate(pil_angle, expand=True),
                               use_det=False, use_cls=False, use_rec=True)
            score = float(result.scores[0]) if result.scores else 0.0
            options.append((score, source_angle))
        best_score, best_angle = max(options)
        return best_angle if best_score >= 0.5 else None

    # 检查实际建立的三个子会话
    # 不能仅相信use_cuda参数
    # 任何非CPU提供器都拒绝
    # 纯CPU约束贯穿每次识别
    def _verify_cpu(self) -> None:
        for part in (self._ocr.text_det, self._ocr.text_cls, self._ocr.text_rec):
            if part is not None and part.session.session.get_providers() != ["CPUExecutionProvider"]:
                raise OcrError("RapidOCR 使用了非 CPU ONNX Runtime 执行器")
