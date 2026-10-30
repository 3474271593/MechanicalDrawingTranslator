# mechanical_drawing_translator/translation_professional.py
"""Source-aware mechanical terminology composition and output quality gates."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from .resource_paths import package_resource

from .translation_glossary import Glossary


def default_d7e_knowledge_path() -> Path:
    return (
        package_resource('translation', 'mechanical_professional_d7e.json')
    )


# 源归一仅处理大小写、连字符和句点，
# 不会补词或进行OCR拼写修复。
# 目标键清理空白与兼容字形，
# 只为受控同义词比较服务。
def _normalize_source(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).upper()
    normalized = re.sub(r"[‐‑‒–—]", "-", normalized)
    return " ".join(normalized.strip().rstrip(".").split())


def _normalize_target(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


@dataclass(frozen=True)
# 专业组合保存参与的全部term ID，
# source_files汇总其来源范围。
# 这类结果仍是机器组合，
# 不能由词条证据推成整句已人工批准。
class ProfessionalResolution:
    translated_text: str
    entry_id: str
    match_kind: str
    term_ids: tuple[str, ...]
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class _Term:
    identifier: str
    source: str
    preferred_target: str
    accepted_targets: tuple[str, ...]
    evidence_entry_ids: tuple[str, ...]
    source_files: frozenset[str]
    source_pages: tuple[str, ...]


class ProfessionalKnowledge:
    """Compose only fully-accounted expressions from reviewed terminology."""

    def __init__(self, terms: tuple[_Term, ...]) -> None:
        self._terms = terms
        # 粗糙度MIN/MAX组合默认关闭，
        # 服务根据显式quality配置开启。
        # 相同英文可保留多个词义记录，
        # 不在建立索引时静默覆盖其中一项。
        self.enable_roughness_limits = False
        index: dict[str, list[_Term]] = {}
        for term in terms:
            index.setdefault(_normalize_source(term.source), []).append(term)
        self._index = {key: tuple(value) for key, value in index.items()}

    @classmethod
    def load(cls, path: Path, *, glossary: Glossary) -> "ProfessionalKnowledge":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"schema_version", "knowledge_version", "terms"}:
            raise ValueError("D7-E knowledge has invalid top-level fields")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported D7-E knowledge schema")

        # 资源必须引用当前词表中的安全证据，
        # 未知evidence ID不能被加载。
        # preferred_target必须在accepted集合中，
        # 来源文件和页码均不可空。
        # 每条专业term保持独立且唯一的ID。
        reviewed = {item.entry_id: item for item in glossary.reviewed_phrases()}
        terms: list[_Term] = []
        identifiers: set[str] = set()
        resource_fields = {
            "id", "source", "preferred_target", "accepted_targets",
            "evidence_entry_ids", "source_files", "source_pages",
        }
        for raw in payload["terms"]:
            if set(raw) != resource_fields:
                raise ValueError("invalid D7-E term fields")
            identifier = raw["id"]
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError("duplicate or invalid D7-E term id")
            identifiers.add(identifier)
            evidence_ids = tuple(raw["evidence_entry_ids"])
            if not evidence_ids or any(item not in reviewed for item in evidence_ids):
                raise ValueError("D7-E term lacks safe reviewed evidence")
            accepted = tuple(raw["accepted_targets"])
            if not accepted or raw["preferred_target"] not in accepted:
                raise ValueError("D7-E preferred target must be accepted")
            if not raw["source_files"] or not raw["source_pages"]:
                raise ValueError("D7-E term lacks source provenance")
            terms.append(_Term(
                identifier, raw["source"], raw["preferred_target"], accepted,
                evidence_ids, frozenset(raw["source_files"]),
                tuple(raw["source_pages"]),
            ))

        # D6 validate-only terms are reviewed global baseline knowledge.
        # D6 validate-only术语可作为全局基线，
        # 但它们的已有同义集合原样保留。
        # MIN/MAX是明确的结构安全词汇，
        # 不据此开放其它模糊材料或缩写。
        for phrase in glossary.reviewed_phrases():
            if phrase.kind != "term":
                continue
            terms.append(_Term(
                phrase.entry_id, phrase.source, phrase.targets[0], phrase.targets,
                (phrase.entry_id,), frozenset(), (),
            ))
        # Unambiguous engineering limit markers are structural safety vocabulary.
        terms.extend((
            _Term("engineering-min", "MIN", "最小", ("最小", "MIN"),
                  ("engineering_semantic_marker",), frozenset(), ()),
            _Term("engineering-max", "MAX", "最大", ("最大", "MAX"),
                  ("engineering_semantic_marker",), frozenset(), ()),
        ))
        return cls(tuple(terms))

    # 排除参与开发的来源依赖专业项，
    # 空来源的全局基础术语仍保留。
    # 返回独立实例支持来源隔离比较，
    # 不改写原加载资源。
    def without_sources(self, excluded: set[str]) -> "ProfessionalKnowledge":
        return ProfessionalKnowledge(tuple(
            term for term in self._terms
            if not term.source_files or term.source_files.isdisjoint(excluded)
        ))

    def _term(self, value: str) -> _Term | None:
        candidates = self._index.get(_normalize_source(value), ())
        if not candidates:
            return None
        # 同归一英文只有唯一首选中文才解析，
        # 多义时返回None交其它路线或回退。
        # 不能按资源顺序选专业意义，
        # 也不能自动合并不同来源的译法。
        targets = {item.preferred_target for item in candidates}
        return candidates[0] if len(targets) == 1 else None

    @staticmethod
    def _resolution(
        translated: str, match_kind: str, terms: tuple[_Term, ...],
    ) -> ProfessionalResolution:
        return ProfessionalResolution(
            translated, "d7e-professional-composition", match_kind,
            tuple(term.identifier for term in terms),
            tuple(sorted({source for term in terms for source in term.source_files})),
        )

    def resolve(self, source: str) -> ProfessionalResolution | None:
        normalized = _normalize_source(source)
        # 粗糙度语法绑定Ra数值与限值方向，
        # MIN与MINIMUM归同一最小角色，
        # MAX同理但不能与MIN互换。
        # 不解析负值或其它复杂粗糙度规范，
        # 超范围继续保持未知边界。
        if self.enable_roughness_limits:
            match = re.fullmatch(r"RA\s*([+]?(?:\d+\.\d+|\.\d+|\d+))\s+(MINIMUM|MAXIMUM|MIN|MAX)", normalized)
            if match:
                term = self._term('MIN' if match[2] in {'MIN', 'MINIMUM'} else 'MAX')
                if term is not None:
                    return self._resolution('Ra' + match[1] + term.preferred_target,
                                            'roughness_limit', (term,))
        exact = self._term(normalized)
        if exact is not None:
            return self._resolution(exact.preferred_target, "reviewed_term", (exact,))

        # 方向短语必须以已知术语为整体对象，
        # TOP/BOTTOM只加有限顶部底部表达。
        # FAR SIDE同样保留反面限定，
        # 不能把方向词翻成另一个物理对象。
        match = re.fullmatch(r"(TOP|BOTTOM) OF (.+)", normalized)
        if match is not None:
            term = self._term(match.group(2))
            if term is not None:
                prefix = "顶部" if match.group(1) == "TOP" else "底部"
                return self._resolution(
                    prefix + term.preferred_target, "direction_of_term", (term,),
                )

        match = re.fullmatch(r"(.+?) \(FAR SIDE\)", normalized)
        if match is not None:
            term = self._term(match.group(1))
            if term is not None:
                return self._resolution(
                    term.preferred_target + "（反面）", "term_far_side", (term,),
                )

        # TO连接两端都要是完整已知术语，
        # 不接受一端未知而模型补齐。
        # 三项列表加PER也要求四槽全部已知，
        # 任何剩余词都使fullmatch失败。
        # 每个槽位的term与来源进入结果。
        if " TO " in normalized:
            left, right = normalized.split(" TO ", 1)
            terms = (self._term(left), self._term(right))
            if all(terms):
                known = (terms[0], terms[1])
                return self._resolution(
                    known[0].preferred_target + "到" + known[1].preferred_target,
                    "term_to_term", known,
                )

        match = re.fullmatch(
            r"(.+?),\s*(.+?),?\s+AND\s+(.+?)\s+PER\s+(.+)", normalized,
        )
        if match is not None:
            resolved = tuple(self._term(value) for value in match.groups())
            if all(resolved):
                terms = (resolved[0], resolved[1], resolved[2], resolved[3])
                translated = (
                    f"{terms[0].preferred_target}，{terms[1].preferred_target}和"
                    f"{terms[2].preferred_target}按{terms[3].preferred_target}"
                )
                return self._resolution(translated, "term_list_per_term", terms)
        return None

    def term_targets(self, identifier: str) -> tuple[str, ...]:
        for term in self._terms:
            if term.identifier == identifier:
                return term.accepted_targets
        return ()

    # 此守卫与组合器职责分开，
    # 模型原始输出也需要限值与术语检查。
    # SentencePiece残留▁明确视为异常，
    # MIN/MAX不能仅因数值保留而省略。
    # D7E术语按英文边界定位，
    # 目标需有某个已声明可接受译法。
    def validate_output(self, source: str, translated: str) -> tuple[str, ...]:
        source_upper = _normalize_source(source)
        target = _normalize_target(translated)
        warnings: list[str] = []
        if "▁" in translated:
            warnings.append("model_output_tokenizer_artifact")
        if re.search(r"\bMIN(?:IMUM)?\b", source_upper) and not re.search(
            r"最小|\bMIN\b", unicodedata.normalize("NFKC", translated), re.I,
        ):
            warnings.append("engineering_minimum_missing")
        if re.search(r"\bMAX(?:IMUM)?\b", source_upper) and not re.search(
            r"最大|\bMAX\b", unicodedata.normalize("NFKC", translated), re.I,
        ):
            warnings.append("engineering_maximum_missing")
        # 只追加专业d7e项的缺失警告，
        # 基础D6项由Glossary验证保持原口径。
        # 多来源重复发现的同警告去重，
        # 不把空警告说成专家整句正确。
        for term in self._terms:
            if not term.identifier.startswith("d7e-"):
                continue
            pattern = rf"(?<![A-Z0-9]){re.escape(_normalize_source(term.source))}(?![A-Z0-9])"
            if re.search(pattern, source_upper) and not any(
                _normalize_target(item) in target for item in term.accepted_targets
            ):
                warnings.append(f"professional_term_missing:{term.identifier}")
        return tuple(dict.fromkeys(warnings))
