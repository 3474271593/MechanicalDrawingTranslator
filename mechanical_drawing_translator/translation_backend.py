# mechanical_drawing_translator/translation_backend.py
"""Local-only CTranslate2 and SentencePiece translation backend."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


# 后端接口只负责本地模型计算，
# 术语选择与工程审核由服务执行。
# 配置、资源、推理错误分型抛出，
# 便于区分缺依赖、坏模型与不完整批次。
class TranslationConfigurationError(RuntimeError):
    """The local runtime is unavailable or misconfigured."""


class TranslationModelError(RuntimeError):
    """The local model resource is missing, invalid, or modified."""


class TranslationInferenceError(RuntimeError):
    """The loaded backend could not produce a complete ordered batch."""


@dataclass(frozen=True)
class ModelFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    package_version: str
    compute_type: str
    files: tuple[ModelFile, ...]


@dataclass(frozen=True)
# 每项结果记录真实输入输出token数，
# 达到解码上限时保留截断风险标记。
# 生成文本不自带正确或审核资格，
# 服务必须继续验证完整工程信息。
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    hit_output_limit: bool


@dataclass(frozen=True)
# 注意力只提供源目标定位证据，
# 字符跨度来自真实分词器proto。
# 没有安全跨度时保留原假设，
# 不手写偏移猜测字符对应。
# 这些跨度也不是图纸坐标。
class AlignedGeneration:
    """Same greedy hypothesis plus optional source/target character alignment.

    Empty alignment means the raw hypothesis remains usable by the original
    validation path. Attention locates tokens; it does not certify meaning.
    """
    generation: Generation
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    source_spans: tuple[tuple[int, int], ...]
    target_spans: tuple[tuple[int, int], ...]
    attention: tuple[tuple[float, ...], ...]


# Protocol维持模型实现可替换，
# 输入批次与输出顺序保持一致。
# diagnostics应返回可观察运行配置，
# 不暴露可被外部改写的内部状态。
class TranslationBackend(Protocol):
    def load(self) -> None: ...
    # 计数依赖真实SentencePiece分片，
    # 不是按英文单词或字符估计。
    # 首次计数也会触发加载与校验，
    # 性能计时不能把这部分藏到边界外。
    def count_input_tokens(self, text: str) -> int: ...
    def translate_batch(self, texts: Sequence[str]) -> tuple[Generation, ...]: ...
    @property
    def diagnostics(self) -> Mapping[str, Any]: ...


def _sha256(path: Path) -> str:
    # 分块哈希避免一次读入大权重，
    # 校验成本与模型初始化分开理解。
    # 文件校验不请求网络资源，
    # 不存在下载失败后的远端回退。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_manifest(model_root: Path) -> ModelManifest:
    """Validate the complete local model manifest before importing runtimes."""
    root = model_root.resolve()
    manifest_path = root / "model-manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationModelError(f"cannot read model manifest: {manifest_path}: {exc}") from exc
    # 清单schema与键集合精确校验，
    # 版本及模型身份必须非空。
    # 至少列出一个真实资源文件，
    # 不能把空清单当作离线模型已就绪。
    required = {"schema_version", "model_id", "package_version", "compute_type", "files"}
    if set(payload) != required or payload.get("schema_version") != 1:
        raise TranslationModelError("unsupported or invalid model manifest schema")
    if not all(isinstance(payload.get(field), str) and payload[field].strip()
               for field in ("model_id", "package_version", "compute_type")):
        raise TranslationModelError("model manifest identity fields must be non-empty strings")
    if not isinstance(payload["files"], list) or not payload["files"]:
        raise TranslationModelError("model manifest files must be a non-empty list")

    records: list[ModelFile] = []
    seen: set[str] = set()
    # 清单中每条只允许路径、大小、摘要，
    # 同一路径不应被重复登记。
    # 绝对路径与父目录跳转先拒绝，
    # resolve后再验证仍位于模型根下。
    # 符号链接也不能逃逸受控资源根。
    for raw in payload["files"]:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise TranslationModelError("invalid model manifest file record")
        relative = raw["path"]
        if not isinstance(relative, str) or not relative or relative in seen:
            raise TranslationModelError(f"invalid or duplicate model file path: {relative}")
        seen.add(relative)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise TranslationModelError(f"model file escapes resource root: {relative}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TranslationModelError(f"model file escapes resource root: {relative}") from exc
        if not path.is_file():
            raise TranslationModelError(f"model file is missing: {relative}")
        # 布尔值虽是int子类仍不得当字节数，
        # 负数大小与错误长度摘要同样拒绝。
        # 真实大小与SHA256都必须一致，
        # 不凭文件存在认定权重正确。
        if isinstance(raw["bytes"], bool) or not isinstance(raw["bytes"], int) or raw["bytes"] < 0:
            raise TranslationModelError(f"invalid byte count for model file: {relative}")
        expected_hash = raw["sha256"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise TranslationModelError(f"invalid SHA-256 for model file: {relative}")
        if path.stat().st_size != raw["bytes"] or _sha256(path).lower() != expected_hash.lower():
            raise TranslationModelError(f"model file hash/size mismatch: {relative}")
        records.append(ModelFile(relative, raw["bytes"], expected_hash.lower()))
    # 运行至少依赖配置、权重、词表，
    # 以及SentencePiece模型四类资源。
    # 清单缺任一必需项立即报错，
    # 不等初始化时再模糊提示失败。
    required_paths = {
        "model/config.json", "model/model.bin", "model/shared_vocabulary.json",
        "sentencepiece.model",
    }
    if not required_paths.issubset(seen):
        missing = sorted(required_paths - seen)
        raise TranslationModelError(f"model manifest omits required files: {missing}")
    return ModelManifest(
        payload["model_id"], payload["package_version"], payload["compute_type"],
        tuple(records),
    )


class CTranslate2Backend:
    """Reusable CPU translator that never downloads or resolves remote assets."""

    def __init__(
        self,
        model_root: Path,
        *,
        intra_threads: int = 6,
        inter_threads: int = 1,
        batch_size: int = 16,
        beam_size: int = 1,
        compute_type: str = "int8_float32",
        max_output_tokens: int = 512,
    ) -> None:
        # 线程、批大小、beam和上限都要正整数，
        # 不接受True冒充线程数1。
        # 构造时仅保存配置不加载大模型，
        # 后台首次使用才支付真实冷启动成本。
        values = (intra_threads, inter_threads, batch_size, beam_size, max_output_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in values):
            raise TranslationConfigurationError("translation backend numeric settings must be positive")
        self.model_root = Path(model_root)
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.requested_compute_type = compute_type
        self.max_output_tokens = max_output_tokens
        self._translator: Any = None
        self._processor: Any = None
        self._diagnostics: dict[str, Any] = {"loaded": False}

    @property
    # 诊断返回字典副本防止外部修改，
    # loaded只说明运行时已构造。
    # 模型ID、线程和实际compute_type，
    # 用于解释本次推理配置和计时条件。
    # 它们不表示专业准确率或性能已验收。
    def diagnostics(self) -> Mapping[str, Any]:
        return dict(self._diagnostics)

    def load(self) -> None:
        if self._translator is not None:
            return
        # 同一后端实例只加载一次，
        # 先核对本地清单再导入推理依赖。
        # 配置诊断记录实际库版本，
        # 不把提示中的要求版本当实测版本。
        manifest = verify_model_manifest(self.model_root)
        try:
            import ctranslate2
            import sentencepiece as spm
        except (ImportError, ModuleNotFoundError) as exc:
            raise TranslationConfigurationError(
                "ctranslate2==4.8.3 and sentencepiece==0.2.1 are required for local translation"
            ) from exc
        started = time.perf_counter()
        try:
            # SentencePiece与CT2使用同一模型根，
            # CT2固定CPU避免隐含独显依赖。
            # compute_type使用请求配置，
            # 诊断再记录引擎实际采用类型。
            # 任何初始化异常都归为模型加载错误。
            # SentencePiece原生文件接口在Windows中文路径下可能误报不存在。
            # 使用Python的Unicode文件接口读取已通过清单校验的原始字节，
            # 再交给同一分词器解析，不复制模型、不改变分词或翻译参数。
            processor = spm.SentencePieceProcessor(
                model_proto=(self.model_root / "sentencepiece.model").read_bytes()
            )
            translator = ctranslate2.Translator(
                str(self.model_root / "model"),
                device="cpu",
                compute_type=self.requested_compute_type,
                inter_threads=self.inter_threads,
                intra_threads=self.intra_threads,
            )
        except Exception as exc:
            raise TranslationModelError(f"could not load local translation model: {exc}") from exc
        self._processor = processor
        self._translator = translator
        self._diagnostics = {
            "loaded": True,
            "model_id": manifest.model_id,
            "package_version": manifest.package_version,
            "compute_type": translator.compute_type,
            "ctranslate2_version": ctranslate2.__version__,
            "sentencepiece_version": spm.__version__,
            "intra_threads": self.intra_threads,
            "inter_threads": self.inter_threads,
            "batch_size": self.batch_size,
            "beam_size": self.beam_size,
            "load_seconds": time.perf_counter() - started,
        }

    def count_input_tokens(self, text: str) -> int:
        self.load()
        return len(self._processor.encode(text, out_type=str))

    def translate_batch(self, texts: Sequence[str]) -> tuple[Generation, ...]:
        if not texts:
            return ()
        self.load()
        # 整个输入批次先按原顺序分词，
        # max_batch_size是内部推理分批。
        # batch_type按样本数而非token数，
        # 不保证每条说明的计算量相同。
        # 默认beam为1但接口保留显式配置。
        token_batches = [self._processor.encode(text, out_type=str) for text in texts]
        try:
            results = self._translator.translate_batch(
                token_batches,
                max_batch_size=self.batch_size,
                batch_type="examples",
                beam_size=self.beam_size,
                max_decoding_length=self.max_output_tokens,
                return_scores=False,
            )
        except Exception as exc:
            raise TranslationInferenceError(f"local translation inference failed: {exc}") from exc
        # 返回数量不同即无法可靠对齐，
        # 整批抛错而非错配到其它英文。
        # 空hypothesis解码成空文本，
        # 交由上层明确走空结果回退。
        if len(results) != len(texts):
            raise TranslationInferenceError(
                f"translation result count {len(results)} does not match input count {len(texts)}"
            )
        generations: list[Generation] = []
        for source_tokens, result in zip(token_batches, results):
            if not getattr(result, "hypotheses", None) or not result.hypotheses[0]:
                output_tokens: list[str] = []
            else:
                output_tokens = list(result.hypotheses[0])
            # token计数来自真实生成序列，
            # 长度达到上限只表示疑似截断。
            # 不自动补句号或延长模型输出，
            # 保持原假设便于安全校验和诊断。
            generations.append(Generation(
                self._processor.decode(output_tokens),
                len(source_tokens),
                len(output_tokens),
                len(output_tokens) >= self.max_output_tokens,
            ))
        return tuple(generations)

    def translate_aligned_batch(self, texts: Sequence[str]) -> tuple[AlignedGeneration, ...]:
        """Return alignment in the existing greedy call, without a new model.

        CT2 4.8.3 removes implicit source BOS/EOS columns itself. Never add
        hand-written offsets. SentencePiece immutable protos provide offsets
        for the actual pieces, including multi-character Chinese pieces.
        """
        if not texts:
            return ()
        self.load()
        # 对齐接口仍使用同一条生成假设，
        # 仅增加return_attention请求。
        # 不另跑模型以制造第二套译文，
        # 批次顺序与数量不变量仍相同。
        tokens = [self._processor.encode(s, out_type=str) for s in texts]
        try:
            results = self._translator.translate_batch(tokens,
                max_batch_size=self.batch_size, batch_type='examples',
                beam_size=self.beam_size, max_decoding_length=self.max_output_tokens,
                return_scores=False, return_attention=True)
        except Exception as exc:
            raise TranslationInferenceError(f'local aligned inference failed: {exc}') from exc
        if len(results) != len(texts):
            raise TranslationInferenceError('aligned result count does not match input count')
        aligned = []
        for text, source, result in zip(texts, tokens, results):
            target = list(result.hypotheses[0]) if result.hypotheses and result.hypotheses[0] else []
            value = Generation(self._processor.decode(target), len(source), len(target),
                               len(target) >= self.max_output_tokens)
            # 只有原文、译文与pieces逐项一致，
            # 才接纳proto提供的字符跨度。
            # 分片数量相同仍不足以猜测偏移，
            # BOS/EOS列由CT2协议负责处理。
            source_spans = target_spans = attention = ()
            try:
                sp = self._processor.encode(text, out_type='immutable_proto')
                tp = self._processor.decode(target, out_type='immutable_proto')
                if (sp.text == text and tp.text == value.text
                        and [p.piece for p in sp.pieces] == source
                        and [p.piece for p in tp.pieces] == target):
                    source_spans = tuple((p.begin, p.end) for p in sp.pieces)
                    target_spans = tuple((p.begin, p.end) for p in tp.pieces)
                    attention = tuple(tuple(float(v) for v in row) for row in result.attention[0])
            except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
                # Keep the original hypothesis, without unsafe guessed spans.
                source_spans = target_spans = attention = ()
            aligned.append(AlignedGeneration(value, tuple(source), tuple(target),
                                             source_spans, target_spans, attention))
        self._diagnostics['aligned_batches'] = self._diagnostics.get('aligned_batches', 0) + 1
        return tuple(aligned)

    # 多候选是显式有界实验接口，
    # 最多四beam且候选数不大于beam。
    # 不改变原translate_batch贪心API，
    # 也不以模型似然替代工程正确性。
    # 每个候选都要通过完整源目标守卫。
    def translate_candidates(self, texts: Sequence[str], *, beam_size: int = 4,
                             num_hypotheses: int = 4) -> tuple[tuple[Generation, ...], ...]:
        """Bounded alternatives from the same model, never a terminology guarantee.

        The original greedy API is untouched. Callers must validate the complete
        source/target of every hypothesis; likelihood alone is not engineering truth.
        """
        if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 4
               for v in (beam_size, num_hypotheses)) or num_hypotheses > beam_size:
            raise ValueError('candidate budget must be 1 <= hypotheses <= beam <= 4')
        if not texts:
            return ()
        self.load()
        token_batches = [self._processor.encode(text, out_type=str) for text in texts]
        try:
            results = self._translator.translate_batch(
                token_batches, max_batch_size=self.batch_size, batch_type='examples',
                beam_size=beam_size, num_hypotheses=num_hypotheses,
                max_decoding_length=self.max_output_tokens, return_scores=False,
            )
        except Exception as exc:
            raise TranslationInferenceError(f'local candidate inference failed: {exc}') from exc
        if len(results) != len(texts):
            raise TranslationInferenceError('candidate result count does not match input count')
        # 候选按输入分组保持一一关联，
        # 空候选集或超预算候选立即拒绝。
        # 每条候选分别记录截断风险，
        # 调用次数仅进入诊断计数。
        batches=[]
        for source_tokens,result in zip(token_batches,results):
            hypotheses=getattr(result,'hypotheses',None)
            if not hypotheses or len(hypotheses)>num_hypotheses:
                raise TranslationInferenceError('missing or excessive candidate hypotheses')
            batches.append(tuple(Generation(self._processor.decode(tokens),len(source_tokens),
                len(tokens),len(tokens)>=self.max_output_tokens) for tokens in hypotheses))
        self._diagnostics['candidate_batches']=self._diagnostics.get('candidate_batches',0)+1
        return tuple(batches)
