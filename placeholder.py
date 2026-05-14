from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from models import ExtractedNode, ExtractionResult, PlaceholderReplacement

COMPLEX_ENTITY_PATTERN = re.compile(r"[:()\[\]{}\"'\u201c\u201d\u2018\u2019,\-\u2013\u2014]")
LONG_NAMED_ENTITY_TYPES = {
    "album",
    "book",
    "film",
    "movie",
    "organization",
    "organisation",
    "company",
    "work",
    "song",
    "series",
    "play",
    "novel",
}


@dataclass(frozen=True)
class SpanReplacement:
    start: int
    end: int
    placeholder: str
    text: str


@dataclass
class _MaskCandidate:
    node: ExtractedNode
    start: int
    end: int
    original_text: str
    mask: str = ""
    masked_start: int = -1
    masked_end: int = -1


def selective_entity_masking(
    original_question: str,
    extracted_nodes: ExtractionResult | dict[str, Any],
) -> PlaceholderReplacement:
    """Mask only complex named entities and preserve type variables in the sentence.

    The returned ``anchor_extraction`` is aligned to the masked question: masked
    entities use EntityA/EntityB anchor ids, while all type-variable anchors stay
    unmasked with shifted character spans.
    """

    extraction = _coerce_extraction(original_question, extracted_nodes)
    candidates = _select_mask_candidates(original_question, extraction)
    _assign_entity_masks(candidates, extraction)
    _compute_masked_spans(candidates)

    masked_question = original_question
    for candidate in sorted(candidates, key=lambda item: item.start, reverse=True):
        masked_question = (
            masked_question[: candidate.start] + candidate.mask + masked_question[candidate.end :]
        )

    anchor_extraction = _build_anchor_extraction(
        original_question=original_question,
        masked_question=masked_question,
        extraction=extraction,
        candidates=candidates,
    )
    mask_mapping = _build_mask_mapping(candidates)
    preserved_type_variables = _build_preserved_type_variables(
        original_nodes=extraction.type_variables,
        masked_nodes=anchor_extraction.type_variables,
    )

    mapping = {node.placeholder: node.text for node in anchor_extraction.nodes}
    replacements = [
        {
            "start": candidate.start,
            "end": candidate.end,
            "masked_start": candidate.masked_start,
            "masked_end": candidate.masked_end,
            "placeholder": candidate.mask,
            "original_placeholder": candidate.node.placeholder,
            "text": candidate.original_text,
            "semantic_type": candidate.node.semantic_type,
            "kind": candidate.node.kind,
        }
        for candidate in sorted(candidates, key=lambda item: item.start)
    ]
    return PlaceholderReplacement(
        question=masked_question,
        mapping=mapping,
        replacements=replacements,
        mask_mapping=mask_mapping,
        preserved_type_variables=preserved_type_variables,
        anchor_extraction=anchor_extraction,
    )


def replace_with_placeholders(question: str, extraction: ExtractionResult) -> PlaceholderReplacement:
    spans = _collect_spans(question, extraction.nodes)
    filtered = _remove_overlaps(spans)

    replaced = question
    applied: list[dict[str, object]] = []
    for span in sorted(filtered, key=lambda item: item.start, reverse=True):
        replaced = replaced[: span.start] + span.placeholder + replaced[span.end :]
        applied.append(
            {
                "start": span.start,
                "end": span.end,
                "placeholder": span.placeholder,
                "text": span.text,
            }
        )

    mapping = {node.placeholder: node.text for node in extraction.nodes}
    return PlaceholderReplacement(
        question=replaced,
        mapping=mapping,
        replacements=list(reversed(applied)),
    )


def _collect_spans(question: str, nodes: list[ExtractedNode]) -> list[SpanReplacement]:
    spans: list[SpanReplacement] = []
    text_to_nodes: dict[str, list[ExtractedNode]] = {}
    for node in nodes:
        text_to_nodes.setdefault(node.text.lower(), []).append(node)

    for node in nodes:
        if node.start is not None and node.end is not None and 0 <= node.start < node.end <= len(question):
            spans.append(
                SpanReplacement(
                    start=node.start,
                    end=node.end,
                    placeholder=node.placeholder,
                    text=question[node.start : node.end],
                )
            )

    occupied = {(span.start, span.end) for span in spans}
    for lowered_text, same_text_nodes in text_to_nodes.items():
        # If the same surface form maps to multiple nodes, rely on explicit spans.
        # This preserves distinct repeated roles such as two directors.
        if len(same_text_nodes) != 1:
            continue
        node = same_text_nodes[0]
        for match in re.finditer(re.escape(node.text), question, flags=re.IGNORECASE):
            key = (match.start(), match.end())
            if key in occupied:
                continue
            spans.append(
                SpanReplacement(
                    start=match.start(),
                    end=match.end(),
                    placeholder=node.placeholder,
                    text=match.group(0),
                )
            )
            occupied.add(key)

    return spans


def _coerce_extraction(
    question: str,
    extracted_nodes: ExtractionResult | dict[str, Any],
) -> ExtractionResult:
    if isinstance(extracted_nodes, ExtractionResult):
        return extracted_nodes
    if not isinstance(extracted_nodes, dict):
        raise TypeError("extracted_nodes must be an ExtractionResult or a mapping.")

    entities = _coerce_node_list(question, extracted_nodes.get("entities", []), "entity")
    type_variables = _coerce_node_list(
        question,
        extracted_nodes.get("type_variables", extracted_nodes.get("typeVariables", [])),
        "type_variable",
    )
    return ExtractionResult(entities=entities, type_variables=type_variables)


def _coerce_node_list(question: str, raw_nodes: Any, kind: str) -> list[ExtractedNode]:
    if not isinstance(raw_nodes, list):
        return []

    nodes: list[ExtractedNode] = []
    for index, raw in enumerate(raw_nodes, start=1):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        semantic_type = str(raw.get("semantic_type", raw.get("type", ""))).strip() or (
            "Entity" if kind == "entity" else "Variable"
        )
        placeholder = str(raw.get("placeholder", "")).strip() or _fallback_placeholder(
            semantic_type,
            kind,
            index,
        )
        occurrence = _coerce_int(raw.get("occurrence"))
        start = _coerce_int(raw.get("start"))
        end = _coerce_int(raw.get("end"))
        start, end = _resolve_node_span(question, text, start, end, occurrence)
        nodes.append(
            ExtractedNode(
                placeholder=placeholder,
                text=text,
                kind=kind,
                semantic_type=semantic_type,
                start=start,
                end=end,
                occurrence=occurrence,
            )
        )
    return nodes


def _select_mask_candidates(question: str, extraction: ExtractionResult) -> list[_MaskCandidate]:
    candidates: list[_MaskCandidate] = []
    for node in extraction.entities:
        if not _is_complex_entity(node):
            continue
        start, end = _resolve_node_span(question, node.text, node.start, node.end, node.occurrence)
        if start is None or end is None or not (0 <= start < end <= len(question)):
            continue
        candidates.append(
            _MaskCandidate(
                node=node,
                start=start,
                end=end,
                original_text=question[start:end],
            )
        )
    return _remove_overlapping_candidates(candidates)


def _is_complex_entity(node: ExtractedNode) -> bool:
    if not node.is_entity:
        return False

    text = node.text.strip()
    token_count = len(re.findall(r"[A-Za-z0-9]+", text))
    semantic_type = node.semantic_type.strip().lower()
    has_complex_punctuation = bool(COMPLEX_ENTITY_PATTERN.search(text))
    is_long_named_entity_type = any(item in semantic_type for item in LONG_NAMED_ENTITY_TYPES)
    return has_complex_punctuation or token_count >= 3 or (
        is_long_named_entity_type and token_count >= 2 and len(text) >= 12
    )


def _assign_entity_masks(candidates: list[_MaskCandidate], extraction: ExtractionResult) -> None:
    candidate_ids = {id(candidate.node) for candidate in candidates}
    reserved = {node.placeholder for node in extraction.nodes if id(node) not in candidate_ids}
    next_index = 0
    for candidate in sorted(candidates, key=lambda item: item.start):
        mask = _entity_mask_label(next_index)
        while mask in reserved:
            next_index += 1
            mask = _entity_mask_label(next_index)
        candidate.mask = mask
        reserved.add(mask)
        next_index += 1


def _compute_masked_spans(candidates: list[_MaskCandidate]) -> None:
    offset = 0
    for candidate in sorted(candidates, key=lambda item: item.start):
        candidate.masked_start = candidate.start + offset
        candidate.masked_end = candidate.masked_start + len(candidate.mask)
        offset += len(candidate.mask) - (candidate.end - candidate.start)


def _build_anchor_extraction(
    original_question: str,
    masked_question: str,
    extraction: ExtractionResult,
    candidates: list[_MaskCandidate],
) -> ExtractionResult:
    candidate_by_node_id = {id(candidate.node): candidate for candidate in candidates}
    replacements = [(candidate.start, candidate.end, candidate.mask) for candidate in candidates]

    entities: list[ExtractedNode] = []
    for node in extraction.entities:
        candidate = candidate_by_node_id.get(id(node))
        if candidate is not None:
            entities.append(
                replace(
                    node,
                    placeholder=candidate.mask,
                    start=candidate.masked_start,
                    end=candidate.masked_end,
                    occurrence=1,
                )
            )
            continue
        start, end = _translate_or_find_span(
            original_question=original_question,
            masked_question=masked_question,
            node=node,
            replacements=replacements,
        )
        entities.append(replace(node, start=start, end=end))

    type_variables: list[ExtractedNode] = []
    for node in extraction.type_variables:
        start, end = _translate_or_find_span(
            original_question=original_question,
            masked_question=masked_question,
            node=node,
            replacements=replacements,
        )
        type_variables.append(replace(node, start=start, end=end))

    return ExtractionResult(entities=entities, type_variables=type_variables)


def _build_mask_mapping(candidates: list[_MaskCandidate]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda item: item.start):
        result[candidate.mask] = {
            "text": candidate.original_text,
            "type": candidate.node.semantic_type,
            "semantic_type": candidate.node.semantic_type,
            "kind": candidate.node.kind,
            "span": {"start": candidate.start, "end": candidate.end},
            "masked_span": {"start": candidate.masked_start, "end": candidate.masked_end},
            "original_placeholder": candidate.node.placeholder,
        }
    return result


def _build_preserved_type_variables(
    original_nodes: list[ExtractedNode],
    masked_nodes: list[ExtractedNode],
) -> list[dict[str, Any]]:
    preserved: list[dict[str, Any]] = []
    for original, masked in zip(original_nodes, masked_nodes):
        preserved.append(
            {
                "placeholder": masked.placeholder,
                "text": original.text,
                "type": original.semantic_type,
                "semantic_type": original.semantic_type,
                "span": _span_dict(original.start, original.end),
                "masked_span": _span_dict(masked.start, masked.end),
            }
        )
    return preserved


def _translate_or_find_span(
    original_question: str,
    masked_question: str,
    node: ExtractedNode,
    replacements: list[tuple[int, int, str]],
) -> tuple[int | None, int | None]:
    if _span_matches_text(original_question, node):
        translated = _translate_span(node.start or 0, node.end or 0, replacements)
        if translated != (None, None):
            return translated
    return _resolve_node_span(masked_question, node.text, None, None, node.occurrence)


def _translate_span(
    start: int,
    end: int,
    replacements: list[tuple[int, int, str]],
) -> tuple[int | None, int | None]:
    offset = 0
    for replacement_start, replacement_end, mask in sorted(replacements, key=lambda item: item[0]):
        if replacement_end <= start:
            offset += len(mask) - (replacement_end - replacement_start)
            continue
        if replacement_start >= end:
            break
        return None, None
    return start + offset, end + offset


def _resolve_node_span(
    question: str,
    text: str,
    start: int | None,
    end: int | None,
    occurrence: int | None,
) -> tuple[int | None, int | None]:
    if _valid_span_bounds(question, start, end) and question[start or 0 : end or 0].lower() == text.lower():
        return start, end

    matches = list(re.finditer(re.escape(text), question, flags=re.IGNORECASE))
    if not matches:
        return start, end
    index = max((occurrence or 1) - 1, 0)
    if index >= len(matches):
        index = 0
    match = matches[index]
    return match.start(), match.end()


def _valid_span_bounds(question: str, start: int | None, end: int | None) -> bool:
    return start is not None and end is not None and 0 <= start < end <= len(question)


def _span_matches_text(question: str, node: ExtractedNode) -> bool:
    if not _valid_span_bounds(question, node.start, node.end):
        return False
    return question[node.start or 0 : node.end or 0].lower() == node.text.lower()


def _remove_overlapping_candidates(candidates: list[_MaskCandidate]) -> list[_MaskCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.start, -(item.end - item.start)))
    result: list[_MaskCandidate] = []
    used: list[tuple[int, int]] = []
    for candidate in ordered:
        if any(not (candidate.end <= start or candidate.start >= end) for start, end in used):
            continue
        result.append(candidate)
        used.append((candidate.start, candidate.end))
    return result


def _entity_mask_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    current = index
    while True:
        label = alphabet[current % len(alphabet)] + label
        current = current // len(alphabet) - 1
        if current < 0:
            break
    return f"Entity{label}"


def _fallback_placeholder(semantic_type: str, kind: str, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9]", "", semantic_type.strip()) or (
        "Entity" if kind == "entity" else "Variable"
    )
    if not base[:1].isalpha():
        base = "Entity" if kind == "entity" else "Variable"
    return f"{base[:1].upper()}{base[1:]}{index}"


def _span_dict(start: int | None, end: int | None) -> dict[str, int] | None:
    if start is None or end is None:
        return None
    return {"start": start, "end": end}


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _remove_overlaps(spans: list[SpanReplacement]) -> list[SpanReplacement]:
    ordered = sorted(spans, key=lambda item: (item.start, -(item.end - item.start)))
    result: list[SpanReplacement] = []
    used: list[tuple[int, int]] = []

    for span in ordered:
        if any(not (span.end <= start or span.start >= end) for start, end in used):
            continue
        result.append(span)
        used.append((span.start, span.end))
    return result
