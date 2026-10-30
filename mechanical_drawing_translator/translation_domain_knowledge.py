# mechanical_drawing_translator/translation_domain_knowledge.py
"""Strict, offline D7-C phrase reuse and parameterized domain knowledge."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from .resource_paths import package_resource

from .engineering_text import validate_engineering_semantics, validate_immutable_literals
from .translation_glossary import Glossary, ReviewedPhrase


# 装配数量模板完整消费英文，
# 件数与组件组数是不同参数槽。
# 两者只接正整数并保留原数字，
# 不推算总零件数替代原要求。
# 单复数稍后依数值明确核验。
_ASSEMBLY_PATTERN = re.compile(
    r"^\s*(?P<pieces>[1-9]\d*)\s+(?P<piece_word>PIECE|PIECES)\s+"
    r"IN\s+(?:THE\s+)?ASSEMBLY\s*,\s*"
    r"(?P<assemblies>[1-9]\d*)\s+"
    r"(?P<assembly_word>ASSEMBLY|ASSEMBLIES)\s+REQUIRED\s*\.?\s*$",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).upper().split())


def default_d7c_knowledge_path() -> Path:
    return (
        package_resource('translation', 'mechanical_domain_knowledge_d7c.json')
    )


def default_v1_isolation_path() -> Path:
    return (
        package_resource('translation', 'mechanical_glossary_v1_isolation.json')
    )


@dataclass(frozen=True)
# 覆盖只绑定原safe V1条目ID，
# 来源引用保留范围与复用理由。
# 不会改写V1资源中的原中文，
# 也不把覆盖译法推广到任意相似句。
class ReuseOverride:
    identifier: str
    source_entry_id: str
    source: str
    target: str
    source_reference: str


@dataclass(frozen=True)
class KnowledgeResolution:
    translated_text: str
    entry_id: str
    origin: str
    match_kind: str


class DomainKnowledge:
    def __init__(
        self,
        *,
        version: str,
        reuse_overrides: dict[str, ReuseOverride],
        reviewed_phrases: tuple[ReviewedPhrase, ...],
        # 模板仅支持已实现assembly_quantity类型，
        # 未知类型即加载错误。
        # 同种模板只允许一个来源ID，
        # 避免同一语法受到多份冲突规则控制。
        template_ids: dict[str, str],
        isolated_entry_ids: frozenset[str],
    ) -> None:
        self.version = version
        self.reuse_overrides = dict(reuse_overrides)
        self.reviewed_phrases = reviewed_phrases
        self._template_ids = dict(template_ids)
        self.isolated_entry_ids = isolated_entry_ids

    @classmethod
    def load(
        cls, path: Path, *, safe_glossary: Glossary, isolation_path: Path,
    ) -> "DomainKnowledge":
        # 领域知识与隔离清单同时读取，
        # schema、版本与列表类型逐层校验。
        # 无法完整加载时拒绝整个对象，
        # 不把坏资源静默当作没有增强。
        # 本层只处理本地JSON，
        # 没有网络查询或答案回填。
        payload = _read_json(path, "D7-C knowledge")
        isolation = _read_json(isolation_path, "V1 isolation manifest")
        if set(payload) != {
            "schema_version", "knowledge_version", "reuse_overrides",
            "reviewed_phrases", "templates",
        }:
            raise ValueError("D7-C knowledge has invalid top-level fields")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported D7-C knowledge schema_version")
        version = payload["knowledge_version"]
        if not isinstance(version, str) or not version.strip():
            raise ValueError("D7-C knowledge_version must be non-empty")
        for field in ("reuse_overrides", "reviewed_phrases", "templates"):
            if not isinstance(payload[field], list):
                raise ValueError(f"D7-C {field} must be a list")

        # 隔离同时按ID与归一源串建立集合，
        # 防止用新ID重新接纳同一危险英文。
        # 安全词表按当前传入视图建索引，
        # 不存在从全量V1直接借未审核证据。
        isolated_ids, isolated_sources = _load_isolation(isolation)
        safe_by_id = {
            phrase.entry_id: phrase for phrase in safe_glossary.reviewed_phrases()
        }
        identifiers: set[str] = set()
        overrides: dict[str, ReuseOverride] = {}
        for raw in payload["reuse_overrides"]:
            required = {
                "id", "source_entry_id", "source", "target", "source_reference",
            }
            _validate_object(raw, required, "reuse override")
            _validate_strings(raw, required, "reuse override")
            _claim_identifier(raw["id"], identifiers)
            # 覆盖必须引用真实safe V1，
            # 隔离项优先明确拒绝。
            # origin与源英文也须一致，
            # 不能只因ID相同便授予另一句中文。
            # 译文经字面、语义与术语三层验证。
            source_entry_id = raw["source_entry_id"]
            if source_entry_id in isolated_ids:
                raise ValueError(f"reuse override references isolated entry: {source_entry_id}")
            safe_phrase = safe_by_id.get(source_entry_id)
            if safe_phrase is None:
                raise ValueError(f"reuse override does not reference safe V1: {source_entry_id}")
            if safe_phrase.origin != "v1" or _normalize(safe_phrase.source) != _normalize(raw["source"]):
                raise ValueError("reuse override source does not match safe V1 entry")
            _validate_reviewed_translation(
                raw["source"], raw["target"], safe_glossary,
            )
            overrides[source_entry_id] = ReuseOverride(
                raw["id"], source_entry_id, raw["source"], raw["target"],
                raw["source_reference"],
            )

        # 新增reviewed短语不能重复源串，
        # 也不能重用隔离清单的危险原句。
        # 每一项都单独验证工程信息，
        # 不以所在文件名替代内容安全检查。
        # 来源统一标d7c保留与V1区别。
        phrases: list[ReviewedPhrase] = []
        seen_sources: set[str] = set()
        for raw in payload["reviewed_phrases"]:
            required = {"id", "source", "target", "source_reference"}
            _validate_object(raw, required, "reviewed phrase")
            _validate_strings(raw, required, "reviewed phrase")
            _claim_identifier(raw["id"], identifiers)
            source_key = _normalize(raw["source"])
            if source_key in seen_sources:
                raise ValueError(f"duplicate reviewed phrase source: {raw['source']}")
            if source_key in isolated_sources:
                raise ValueError(f"reviewed phrase duplicates isolated source: {raw['source']}")
            seen_sources.add(source_key)
            _validate_reviewed_translation(raw["source"], raw["target"], safe_glossary)
            phrases.append(ReviewedPhrase(
                raw["id"], raw["source"], (raw["target"],), "exact", "d7c",
            ))

        template_ids: dict[str, str] = {}
        for raw in payload["templates"]:
            required = {"id", "type", "source_reference"}
            _validate_object(raw, required, "template")
            _validate_strings(raw, required, "template")
            _claim_identifier(raw["id"], identifiers)
            if raw["type"] != "assembly_quantity":
                raise ValueError(f"unsupported template type: {raw['type']}")
            if raw["type"] in template_ids:
                raise ValueError(f"duplicate template type: {raw['type']}")
            template_ids[raw["type"]] = raw["id"]
        return cls(
            version=version,
            reuse_overrides=overrides,
            reviewed_phrases=tuple(phrases),
            template_ids=template_ids,
            isolated_entry_ids=frozenset(isolated_ids),
        )

    def resolve_template(self, source: str) -> KnowledgeResolution | None:
        identifier = self._template_ids.get("assembly_quantity")
        if identifier is None:
            return None
        match = _ASSEMBLY_PATTERN.fullmatch(source)
        if match is None:
            return None
        # 数值为1时必须使用单数形态，
        # 其它正整数必须用明确复数。
        # 不替源文本自动纠正PIECE/PIECES，
        # 语法证据不足就拒绝该模板。
        # 生成后再对完整原串做字面量保护。
        pieces = match.group("pieces")
        assemblies = match.group("assemblies")
        if (int(pieces) == 1) != (match.group("piece_word").upper() == "PIECE"):
            return None
        if (int(assemblies) == 1) != (
            match.group("assembly_word").upper() == "ASSEMBLY"
        ):
            return None
        translated = f"每个组件 {pieces} 件，需要 {assemblies} 组组件。"
        if not validate_immutable_literals(source, translated).is_valid:
            return None
        return KnowledgeResolution(
            translated, identifier, "d7c", "assembly_quantity_template",
        )


# JSON顶层必须是真实对象映射，
# 列表或标量不能冒充规则资源。
# 读文件失败与JSON损坏保留原异常链，
# 标签说明究竟哪份本地知识失败。
# 这里没有容错加载部分记录，
# 坏知识不会自动降低到不校验模式。
def _read_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


# 隔离项必须有非空warning_codes，
# 不能只登记一个失去原因的ID。
# 重复隔离ID同样拒绝，
# 返回源串与ID供两种准入检查。
def _load_isolation(payload: dict) -> tuple[set[str], set[str]]:
    if set(payload) != {"schema_version", "isolation_version", "entries"}:
        raise ValueError("V1 isolation manifest has invalid top-level fields")
    if payload["schema_version"] != 1 or not isinstance(payload["entries"], list):
        raise ValueError("invalid V1 isolation manifest schema")
    identifiers: set[str] = set()
    sources: set[str] = set()
    for raw in payload["entries"]:
        _validate_object(raw, {"id", "source", "warning_codes"}, "isolated entry")
        if not isinstance(raw["id"], str) or not raw["id"].strip() \
                or not isinstance(raw["source"], str) or not raw["source"].strip() \
                or not isinstance(raw["warning_codes"], list) \
                or not raw["warning_codes"] \
                or not all(isinstance(item, str) and item for item in raw["warning_codes"]):
            raise ValueError("invalid isolated entry values")
        if raw["id"] in identifiers:
            raise ValueError(f"duplicate isolated entry id: {raw['id']}")
        identifiers.add(raw["id"])
        sources.add(_normalize(raw["source"]))
    return identifiers, sources


def _validate_object(raw: object, required: set[str], label: str) -> None:
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"invalid {label} fields")


def _validate_strings(raw: dict, fields: set[str], label: str) -> None:
    if not all(isinstance(raw[field], str) and raw[field].strip() for field in fields):
        raise ValueError(f"invalid {label} string values")


# 同一D7C资源内各章节共用ID命名空间，
# 覆盖、完整短语与模板均不可重复。
# 唯一标识用于解释实际采用来源，
# 不是给重复条目重新编号后继续接纳。
# 新增ID也不能绕开按源串隔离，
# 两类检查承担不同安全职责。
def _claim_identifier(identifier: str, identifiers: set[str]) -> None:
    if identifier in identifiers:
        raise ValueError(f"duplicate D7-C identifier: {identifier}")
    identifiers.add(identifier)


# 人工审阅来源不免除工程检查，
# 数值、条件、否定或术语问题均拒绝。
# 加载错误包含具体警告供资源排查，
# 不在运行时自动“修正”该知识条目。
def _validate_reviewed_translation(source: str, target: str, glossary: Glossary) -> None:
    literal = validate_immutable_literals(source, target)
    semantic = validate_engineering_semantics(source, target)
    if not literal.is_valid or not semantic.is_valid:
        raise ValueError(
            "D7-C reviewed translation failed engineering validation: "
            f"{(*literal.warning_codes, *semantic.warning_codes)}"
        )
    term_warnings = glossary.validate_terms(source, target)
    if term_warnings:
        raise ValueError(f"D7-C reviewed translation failed term validation: {term_warnings}")
