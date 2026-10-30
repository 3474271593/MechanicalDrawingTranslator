# mechanical_drawing_translator/translation_glossary.py
"""Load and apply the versioned, locally maintained mechanical glossary."""

from __future__ import annotations

import json
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from .resource_paths import package_resource
from typing import Any


# 参数类型是有限工程语法，
# 模型型号、尺寸、范围和角度分开。
# 参数原字面直接捕获到目标模板，
# 不会数值换算或猜测缺失单位。
# term仅负责validate_only，
# 不会在任意英文句中做全局替换。
_PARAMETER_PATTERNS = {
    "angle": r"±\s*(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*°",
    "number": r"[+-]?(?:\d+\.\d+|\.\d+|\d+)",
    "range": r"\d+(?:\.\d+)?\s*[-~–—]\s*\d+(?:\.\d+)?",
    "model": r"(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9]+(?:-[A-Z0-9]+)+",
    "measurement": r"[+-]?(?:\d+\.\d+|\.\d+|\d+)\s*(?:mm|cm|µm|um|[\"″])",
}
_ENTRY_FIELDS = {
    "exact": frozenset({"id", "kind", "source", "target", "source_reference"}),
    "parameterized_sentence": frozenset({
        "id", "kind", "source", "target", "parameters", "source_reference",
    }),
    "term": frozenset({
        "id", "kind", "source", "targets", "inline_policy", "source_reference",
    }),
}


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).upper().split())


# 比较模板形状时忽略槽位名称，
# 相同字面结构改个参数名仍冲突。
# 避免两模板按加载次序争抢同一输入，
# 无法区分的资源应在加载时拒绝。
def _parameterized_shape(value: str) -> tuple[tuple[str, bool], ...]:
    """Normalize template literals while deliberately ignoring placeholder names."""
    return tuple(
        (_normalize(literal), field is not None)
        for literal, field, _, _ in string.Formatter().parse(value)
    )


def default_glossary_path() -> Path:
    return package_resource('translation', 'mechanical_glossary.json')


def default_v1_glossary_path() -> Path:
    return (
        package_resource('translation', 'mechanical_glossary_v1_safe.json')
    )


@dataclass(frozen=True)
class GlossaryResolution:
    translated_text: str
    route: str
    entry_id: str


@dataclass(frozen=True)
# 该视图只暴露已加载的完整词条，
# 不可变元组避免调用方原地改义。
# targets可以明确多个可接受同义词，
# 不能默认首项就是唯一专业意义。
# 后续组合器需按各自歧义策略判断，
# 不能由本视图名称推断整句新授权。
class ReviewedPhrase:
    entry_id: str
    source: str
    targets: tuple[str, ...]
    kind: str
    origin: str


@dataclass(frozen=True)
class _Entry:
    identifier: str
    kind: str
    source: str
    target: str | None
    targets: tuple[str, ...]
    parameters: dict[str, str]
    pattern: re.Pattern[str] | None


def _compile_parameterized(source: str, target: str, parameters: dict[str, str]) -> re.Pattern[str]:
    # 格式器只用于解析已声明槽位，
    # 拒绝format_spec与conversion指令。
    # 模板字面用re.escape保护，
    # 空格可兼容多个空白字符。
    # 每个源槽位必须恰好声明一次，
    # 目标也必须恰好保留同一组槽位。
    formatter = string.Formatter()
    source_fields: list[str] = []
    pieces = ["^"]
    for literal, field, format_spec, conversion in formatter.parse(source):
        if format_spec or conversion:
            raise ValueError("parameter templates do not support formatting directives")
        escaped = re.escape(literal)
        pieces.append(re.sub(r"(?:\\ )+", r"\\s+", escaped))
        if field is not None:
            source_fields.append(field)
            kind = parameters.get(field)
            if kind not in _PARAMETER_PATTERNS:
                raise ValueError(f"unsupported parameter type: {kind}")
            pieces.append(f"(?P<{field}>{_PARAMETER_PATTERNS[kind]})")
    pieces.append("$")
    if set(source_fields) != set(parameters) or len(source_fields) != len(set(source_fields)):
        raise ValueError("parameter source fields must be unique and exactly declared")
    # 源与目标的参数出现次数也核对，
    # 只比set会漏掉目标重复插入数字。
    # 每个槽位只能在目标出现一次，
    # 防止合法值被模型外规则重复添加。
    # 未声明槽位、重复槽位都提前拒绝，
    # 运行时才不会产生误导性参数译文。
    target_fields = [field for _, field, _, _ in formatter.parse(target) if field is not None]
    if sorted(target_fields) != sorted(source_fields) or len(target_fields) != len(set(target_fields)):
        raise ValueError("parameter target fields must each appear exactly once")
    return re.compile("".join(pieces), re.I)


class Glossary:
    def __init__(self, version: str, entries: tuple[_Entry, ...]) -> None:
        self.version = version
        self._entries = entries
        # 对外暴露不可变的已审核短语视图，
        # 参数模板不伪装成完整已审核句子。
        # mgt前缀词条保留V1来源，
        # 其它基础条目标记D6来源。
        # 完整规则与术语分开保存，
        # 术语按长度逆序保护长术语优先。
        self._reviewed_phrase_view = tuple(ReviewedPhrase(
            entry_id=entry.identifier, source=entry.source,
            targets=(entry.target or '',) if entry.kind == 'exact' else entry.targets,
            kind=entry.kind, origin='v1' if entry.identifier.startswith('mgt-') else 'd6',
        ) for entry in entries if entry.kind != 'parameterized_sentence')
        self._full_entries = tuple(entry for entry in entries if entry.kind != "term")
        self._term_entries = tuple(
            sorted((entry for entry in entries if entry.kind == "term"),
                   key=lambda item: len(_normalize(item.source)), reverse=True)
        )

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load translation glossary: {exc}") from exc
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: dict) -> "Glossary":
        """Validate an in-memory compiled resource using the same strict schema."""
        # 顶层字段与schema严格一致，
        # 版本必须是非空字符串。
        # 加载失败不返回部分词库，
        # 避免用户误以为全部资源已生效。
        # 每种entry类型也有专门字段集合。
        if set(payload) != {"schema_version", "glossary_version", "entries"}:
            raise ValueError("translation glossary has invalid top-level fields")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported translation glossary schema_version")
        if not isinstance(payload["glossary_version"], str) or not payload["glossary_version"].strip():
            raise ValueError("translation glossary version must be non-empty")
        if not isinstance(payload["entries"], list):
            raise ValueError("translation glossary entries must be a list")

        entries: list[_Entry] = []
        ids: set[str] = set()
        full_sources: set[str] = set()
        # ID在整个词库中唯一，
        # 英文、来源引用都要非空。
        # 完整表达按归一源串检查冲突，
        # 不能以不同大小写重复授予含义。
        # 术语与整句有不同目标字段契约。
        for raw in payload["entries"]:
            if not isinstance(raw, dict):
                raise ValueError("translation glossary entry must be an object")
            kind = raw.get("kind")
            if kind not in _ENTRY_FIELDS or set(raw) != _ENTRY_FIELDS[kind]:
                raise ValueError(f"translation glossary entry has invalid fields for kind: {kind}")
            identifier = raw["id"]
            source = raw["source"]
            reference = raw["source_reference"]
            if not all(isinstance(value, str) and value.strip()
                       for value in (identifier, source, reference)):
                raise ValueError("translation glossary strings must be non-empty")
            if identifier in ids:
                raise ValueError(f"duplicate glossary id: {identifier}")
            ids.add(identifier)

            target: str | None = None
            targets: tuple[str, ...] = ()
            parameters: dict[str, str] = {}
            pattern: re.Pattern[str] | None = None
            if kind in {"exact", "parameterized_sentence"}:
                normalized_source = _normalize(source)
                if normalized_source in full_sources:
                    raise ValueError(f"ambiguous full glossary source: {source}")
                full_sources.add(normalized_source)
                target = raw["target"]
                if not isinstance(target, str) or not target.strip():
                    raise ValueError("glossary target must be non-empty")
                # 参数映射仅接受字符串键和值，
                # 编译时再检查类型与槽位完整性。
                # 不能让未声明变量在format阶段，
                # 临时读取任意对象或丢失源参数。
                if kind == "parameterized_sentence":
                    if not isinstance(raw["parameters"], dict) or not all(
                        isinstance(key, str) and isinstance(value, str)
                        for key, value in raw["parameters"].items()
                    ):
                        raise ValueError("glossary parameters must be a string mapping")
                    parameters = dict(raw["parameters"])
                    pattern = _compile_parameterized(source, target, parameters)
            else:
                # 术语只登记可接受中文集合，
                # inline_policy必须明确validate_only。
                # 不允许资源偷偷改成句内自动替换，
                # 多种中文同义译法由targets完整声明。
                if raw["inline_policy"] != "validate_only":
                    raise ValueError("term inline_policy must be validate_only")
                if not isinstance(raw["targets"], list) or not raw["targets"] or not all(
                    isinstance(value, str) and value.strip() for value in raw["targets"]
                ):
                    raise ValueError("term targets must be a non-empty string list")
                targets = tuple(raw["targets"])
            entries.append(_Entry(identifier, kind, source, target, targets, parameters, pattern))
        _validate_full_entry_ambiguity(entries)
        return cls(payload["glossary_version"], tuple(entries))

    @classmethod
    def load_many(cls, paths: tuple[Path, ...]) -> "Glossary":
        """Load ordered glossary layers without allowing silent overrides."""
        if not paths:
            raise ValueError("at least one translation glossary path is required")
        layers = tuple(cls.load(path) for path in paths)
        return cls.combine(layers)

    @classmethod
    def combine(cls, layers: tuple["Glossary", ...]) -> "Glossary":
        if not layers:
            raise ValueError("at least one glossary is required")
        merged: list[_Entry] = []
        identifiers: set[str] = set()
        sources: dict[tuple[str, str], _Entry] = {}
        full_sources: dict[str, _Entry] = {}
        # 合并保持给定资源层序，
        # 但不允许后层静默覆盖前层。
        # 跨层ID重复一律拒绝，
        # 整句归一同源仅可接受同一准确译文。
        # 术语同源也必须targets集合顺序一致，
        # 不同意义不能靠优先级掩盖。
        for layer in layers:
            for entry in layer._entries:
                if entry.identifier in identifiers:
                    raise ValueError(f"duplicate glossary id across layers: {entry.identifier}")
                identifiers.add(entry.identifier)
                normalized_source = _normalize(entry.source)
                if entry.kind != "term":
                    previous = full_sources.get(normalized_source)
                    if previous is not None:
                        if entry.kind == previous.kind == "exact" \
                                and (entry.target or "").strip() == (previous.target or "").strip():
                            continue
                        raise ValueError(
                            "conflicting glossary source across layers: "
                            f"{previous.source} / {entry.source}"
                        )
                    full_sources[normalized_source] = entry
                else:
                    key = (entry.kind, normalized_source)
                    previous = sources.get(key)
                    if previous is not None:
                        if entry.targets == previous.targets:
                            continue
                        raise ValueError(
                            "conflicting glossary source across layers: "
                            f"{previous.source} / {entry.source}"
                        )
                    sources[key] = entry
                merged.append(entry)
        # 逐项合并完成还需重新跨层查歧义，
        # 单层合法不代表组合后也互不重叠。
        # 不同层的模板可能命中另一层exact，
        # 因此合并末尾不能省略此检查。
        # 合成version按资源顺序连接，
        # 便于区分使用了哪组离线知识。
        _validate_full_entry_ambiguity(merged)
        return cls("+".join(layer.version for layer in layers), tuple(merged))

    # 只解析完整exact或参数化句，
    # validate_only术语不会在此返回译文。
    # exact按整串NFKC与空白归一，
    # 参数匹配捕获原英文的实际值。
    # 未命中明确返回None供服务继续路由。
    def resolve(self, source: str) -> GlossaryResolution | None:
        normalized = _normalize(source)
        for entry in self._full_entries:
            if entry.kind == "exact" and normalized == _normalize(entry.source):
                return GlossaryResolution(entry.target or "", "exact_rule", entry.identifier)
            if entry.kind == "parameterized_sentence" and entry.pattern is not None:
                match = entry.pattern.fullmatch(source.strip())
                if match is not None:
                    return GlossaryResolution(
                        (entry.target or "").format(**match.groupdict()),
                        "parameterized_rule",
                        entry.identifier,
                    )
        return None

    def reviewed_phrases(self) -> tuple[ReviewedPhrase, ...]:
        """Expose immutable reviewed text for the conservative D7-B matcher."""
        return self._reviewed_phrase_view

    def validate_terms(
        self, source: str, translated: str, *,
        extra_targets: dict[str, tuple[str, ...]] | None = None,
    ) -> tuple[str, ...]:
        normalized_source = _normalize(source)
        normalized_target = _normalize(translated)
        # 术语验证以长术语优先占据源跨度，
        # 重叠短术语不重复要求另一种译法。
        # 边界拒绝嵌在型号中的伪英文词，
        # 目标允许已声明的受控同义表达。
        # 这层只验证术语存在，
        # 不能证明对象绑定或句序正确。
        occupied: list[tuple[int, int]] = []
        warnings: list[str] = []
        for entry in self._term_entries:
            term = _normalize(entry.source)
            pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(term)}(?![A-Z0-9])")
            matches = [match for match in pattern.finditer(normalized_source)
                       if not any(match.start() < end and start < match.end()
                                  for start, end in occupied)]
            if not matches:
                continue
            occupied.extend((match.start(), match.end()) for match in matches)
            targets = (*entry.targets, *(extra_targets or {}).get(entry.identifier, ()))
            if not any(_normalize(target) in normalized_target for target in targets):
                warnings.append(f"required_term_missing:{entry.identifier}")
        return tuple(warnings)


# 相同模板形状不能重复，
# exact也不能落进某参数模板范围。
# 加载前消除无法裁决的重叠输入，
# 防止运行时凭遍历先后随机选译。
# 该检查不授予未审核的新表达。
def _validate_full_entry_ambiguity(entries: list[_Entry]) -> None:
    exact_entries = tuple(entry for entry in entries if entry.kind == "exact")
    parameterized_entries = tuple(
        entry for entry in entries if entry.kind == "parameterized_sentence"
    )
    parameterized_shapes: dict[tuple[tuple[str, bool], ...], str] = {}
    for entry in parameterized_entries:
        shape = _parameterized_shape(entry.source)
        if shape in parameterized_shapes:
            raise ValueError(
                "ambiguous parameterized glossary sources: "
                f"{parameterized_shapes[shape]} / {entry.source}"
            )
        parameterized_shapes[shape] = entry.source
    for exact in exact_entries:
        for parameterized in parameterized_entries:
            if parameterized.pattern is not None \
                    and parameterized.pattern.fullmatch(exact.source.strip()) is not None:
                raise ValueError(
                    "ambiguous exact and parameterized glossary sources: "
                    f"{exact.source} / {parameterized.source}"
                )
