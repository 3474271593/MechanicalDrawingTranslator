# mechanical_drawing_translator/translation_generalization.py
"""Strict D7-D contextual aliases and typed machining templates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from .resource_paths import package_resource

from .engineering_text import validate_immutable_literals
from .translation_glossary import Glossary


# 泛化限定装配后研磨英寸槽，
# 以及完整除非说明的全倒角字段。
# 尺寸、角度槽直接复制原字面，
# 不是任意机械句子的填空模板。
_GRIND_SLOT = re.compile(
    r'^GRIND\s+(?P<size>(?:\d+\.\d+|\.\d+|\d+)\s*["″])\s+'
    r'SLOT\s+AFTER\s+ASSEMBLY\s*\.?$', re.I,
)
_CHAMFER = re.compile(
    r"^ALL\s+CHAMFERS\s+(?P<size>\d+(?:\.\d+)?)\s*[X×]\s*"
    r"(?P<angle>\d+(?:\.\d+)?\s*°)\s+UNLESS\s+OTHERWISE\s+SPECIFIED\s*\.?$",
    re.I,
)


def default_d7d_knowledge_path() -> Path:
    return package_resource('translation', 'mechanical_generalization_d7d.json')


@dataclass(frozen=True)
class GeneralizationResolution:
    translated_text: str
    entry_id: str
    match_kind: str


@dataclass(frozen=True)
# 每条泛化规则记录来源文件集合，
# 来源隔离测试可据此排除重合资源。
# 记录source_files不是盲测证明，
# 仍需独立核对样本家族与接触史。
class _SourcedRule:
    identifier: str
    source_files: frozenset[str]


class GeneralizationKnowledge:
    def __init__(self, aliases: dict[str, tuple[tuple[str, _SourcedRule], ...]],
                 templates: dict[str, _SourcedRule]) -> None:
        self._aliases = aliases
        self._templates = templates

    @classmethod
    def load(cls, path: Path, *, glossary: Glossary) -> "GeneralizationKnowledge":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"schema_version", "knowledge_version", "term_aliases", "templates"}:
            raise ValueError("D7-D knowledge has invalid top-level fields")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported D7-D knowledge schema")
        # 别名必须绑定当前已知term ID，
        # 不能借别名文件新建整句词义。
        # 证据entry_ids和source_files都要存在，
        # 规则ID在别名与模板间也保持唯一。
        term_ids = {item.entry_id for item in glossary.reviewed_phrases() if item.kind == "term"}
        aliases: dict[str, list[tuple[str, _SourcedRule]]] = {}
        identifiers: set[str] = set()
        for raw in payload["term_aliases"]:
            if set(raw) != {"id", "term_id", "targets", "evidence_entry_ids", "source_files"}:
                raise ValueError("invalid D7-D alias fields")
            if raw["term_id"] not in term_ids or not raw["targets"] or not raw["source_files"]:
                raise ValueError("D7-D alias lacks a known term or evidence")
            if not raw["evidence_entry_ids"]:
                raise ValueError("D7-D alias lacks evidence entry IDs")
            rule = _SourcedRule(raw["id"], frozenset(raw["source_files"]))
            if rule.identifier in identifiers:
                raise ValueError("duplicate D7-D knowledge id")
            identifiers.add(rule.identifier)
            aliases.setdefault(raw["term_id"], []).extend((target, rule) for target in raw["targets"])
        templates = {}
        # 只接收源码明确实现的两个模板族，
        # 未知type在加载时拒绝。
        # 每模板保留自己的来源标识，
        # 无需读取源图或模型生成新答案。
        supported = {"grind_slot_after_assembly", "chamfer_unless_specified"}
        for raw in payload["templates"]:
            if set(raw) != {"id", "type", "source_files"} or raw["type"] not in supported:
                raise ValueError("invalid D7-D template")
            rule = _SourcedRule(raw["id"], frozenset(raw["source_files"]))
            if rule.identifier in identifiers or not rule.source_files:
                raise ValueError("duplicate or unsourced D7-D template")
            identifiers.add(rule.identifier)
            templates[raw["type"]] = rule
        return cls({key: tuple(value) for key, value in aliases.items()}, templates)

    # 排除集合与任一来源重合便去掉规则，
    # 同时过滤别名与模板两条路径。
    # 返回新知识对象不改原实例，
    # 便于同轮有来源/无来源A/B比较。
    def without_sources(self, excluded: set[str]) -> "GeneralizationKnowledge":
        aliases = {
            key: tuple((target, rule) for target, rule in values
                       if rule.source_files.isdisjoint(excluded))
            for key, values in self._aliases.items()
        }
        templates = {key: rule for key, rule in self._templates.items()
                     if rule.source_files.isdisjoint(excluded)}
        return GeneralizationKnowledge(aliases, templates)

    # 别名只返回可接受中文补充项，
    # 由服务传给术语验证器使用。
    # 本接口不会替换原英文，
    # 也不把别名升格为全局精确规则。
    def term_targets(self, term_id: str) -> tuple[str, ...]:
        return tuple(target for target, _ in self._aliases.get(term_id, ()))

    def resolve_template(self, source: str) -> GeneralizationResolution | None:
        normalized = " ".join(source.split())
        # 先要求完整英文形状匹配，
        # 且该模板在当前来源视图仍存在。
        # 英寸符号转中文单位属展示等价，
        # 数值和原小数精度仍须保护。
        # 译文最终带真实模板ID返回。
        match = _GRIND_SLOT.fullmatch(normalized)
        if match is not None and "grind_slot_after_assembly" in self._templates:
            size = re.sub(r'["″]\s*$', "", match.group("size")).strip()
            translated = f"组装后研磨 {size} 英寸槽"
            if validate_immutable_literals(source, translated).is_valid:
                rule = self._templates["grind_slot_after_assembly"]
                return GeneralizationResolution(translated, rule.identifier,
                                                "grind_slot_after_assembly")
        # ALL与UNLESS是模板不可缺的范围，
        # 中文同时保留全称与例外条件。
        # X与×只归一尺寸连接形式，
        # 不得交换尺寸和角度两个参数。
        # 字面量守卫失败直接不返回候选。
        match = _CHAMFER.fullmatch(normalized)
        if match is not None and "chamfer_unless_specified" in self._templates:
            translated = f"除非另有说明，所有倒角 {match.group('size')} × {match.group('angle').replace(' ', '')}"
            if validate_immutable_literals(source, translated).is_valid:
                rule = self._templates["chamfer_unless_specified"]
                return GeneralizationResolution(translated, rule.identifier,
                                                "chamfer_unless_specified")
        return None
