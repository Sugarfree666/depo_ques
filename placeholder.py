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
DETERMINERS = {"the", "a", "an"}
LEADING_FUNCTION_WORDS = {
    *DETERMINERS,
    "of",
    "in",
    "on",
    "for",
    "from",
    "to",
    "by",
    "with",
    "at",
    "as",
}
LEFT_EXPANSION_BOUNDARIES = {
    "and",
    "or",
    "but",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "what",
    "did",
    "do",
    "does",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "of",
    "in",
    "on",
    "for",
    "from",
    "to",
    "by",
    "with",
    "at",
    "as",
}
NONESSENTIAL_PREMODIFIERS = {
    "same",
    "robust",
    "local",
    "major",
    "minor",
    "large",
    "small",
    "old",
    "new",
    "famous",
    "known",
    "notable",
}
RELATIVE_OR_COMPLEMENT_BOUNDARIES = {
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
}
POS_HINT_BASE_BY_TYPE = {
    "business": "Company",
    "company": "Company",
    "corporation": "Company",
    "organisation": "Organization",
    "organization": "Organization",
    "ceo": "Person",
    "director": "Person",
    "founder": "Person",
    "person": "Person",
    "people": "Person",
    "film": "Movie",
    "movie": "Movie",
    "book": "Book",
    "novel": "Book",
    "album": "Album",
    "song": "Song",
    "university": "Institution",
    "college": "Institution",
    "school": "Institution",
    "city": "City",
    "country": "Country",
    "region": "Region",
    "network": "Network",
    "system": "System",
    "structure": "Structure",
    "farm": "Artifact",
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
    """Mask complex noun phrases while preserving the syntactic scaffold.

    The returned ``anchor_extraction`` is aligned to the masked question: masked
    noun phrases use POS-hinting anchor ids such as CompanyA/MovieA, while simple
    entity/type-variable anchors stay unmasked with shifted character spans.
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
    masked_type_node_ids = {
        id(candidate.node) for candidate in candidates if candidate.node.is_type_variable
    }
    preserved_type_variables = _build_preserved_type_variables(
        original_nodes=extraction.type_variables,
        masked_nodes=anchor_extraction.type_variables,
        masked_original_node_ids=masked_type_node_ids,
    )

    mapping = {node.placeholder: node.text for node in anchor_extraction.nodes}
    for placeholder, info in mask_mapping.items():
        mapping[placeholder] = str(info.get("text", mapping.get(placeholder, placeholder)))
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
    for node in extraction.nodes:
        candidate_span = _maskable_noun_phrase_span(question, node)
        if candidate_span is None:
            continue
        start, end = candidate_span
        candidates.append(
            _MaskCandidate(
                node=node,
                start=start,
                end=end,
                original_text=question[start:end],
            )
        )
    return _remove_overlapping_candidates(candidates)


def _maskable_noun_phrase_span(question: str, node: ExtractedNode) -> tuple[int, int] | None:
    start, end = _resolve_node_span(question, node.text, node.start, node.end, node.occurrence)
    if start is None or end is None or not (0 <= start < end <= len(question)):
        return None

    start, end = _trim_right_at_clause_boundary(question, start, end)
    if node.is_type_variable and _token_count(question[start:end]) <= 1:
        start = _expand_left_premodifiers(question, start)
    start, end = _trim_leading_mask_preserved_words(question, start, end, node)
    start, end = _trim_outer_whitespace(question, start, end)

    if start is None or end is None or not (0 <= start < end <= len(question)):
        return None
    if not _is_complex_noun_phrase(question[start:end], node):
        return None
    return start, end


def _is_complex_noun_phrase(text: str, node: ExtractedNode) -> bool:
    stripped = text.strip()
    token_count = _token_count(stripped)
    if token_count <= 0:
        return False

    semantic_type = node.semantic_type.strip().lower()
    has_complex_punctuation = bool(COMPLEX_ENTITY_PATTERN.search(stripped))
    if node.is_entity:
        is_long_named_entity_type = any(item in semantic_type for item in LONG_NAMED_ENTITY_TYPES)
        return has_complex_punctuation or token_count >= 3 or (
            is_long_named_entity_type and token_count >= 2 and len(stripped) >= 12
        )

    return has_complex_punctuation or token_count >= 2


def _assign_entity_masks(candidates: list[_MaskCandidate], extraction: ExtractionResult) -> None:
    candidate_ids = {id(candidate.node) for candidate in candidates}
    reserved = {node.placeholder for node in extraction.nodes if id(node) not in candidate_ids}
    counters: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda item: item.start):
        base = _pos_hint_base(candidate.node)
        mask = _pos_hint_label(base, counters.get(base, 0))
        while mask in reserved:
            counters[base] = counters.get(base, 0) + 1
            mask = _pos_hint_label(base, counters[base])
        candidate.mask = mask
        reserved.add(mask)
        counters[base] = counters.get(base, 0) + 1


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
        candidate = candidate_by_node_id.get(id(node))
        if candidate is not None:
            type_variables.append(
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
        type_variables.append(replace(node, start=start, end=end))

    return ExtractionResult(entities=entities, type_variables=type_variables)


def _build_mask_mapping(candidates: list[_MaskCandidate]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda item: item.start):
        result[candidate.mask] = {
            "text": candidate.original_text,
            "node_text": candidate.node.text,
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
    masked_original_node_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    masked_original_node_ids = masked_original_node_ids or set()
    preserved: list[dict[str, Any]] = []
    for original, masked in zip(original_nodes, masked_nodes):
        if id(original) in masked_original_node_ids:
            continue
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


def _trim_right_at_clause_boundary(question: str, start: int, end: int) -> tuple[int, int]:
    text = question[start:end]
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9'-]*\b", text):
        if match.group(0).lower() in RELATIVE_OR_COMPLEMENT_BOUNDARIES:
            return _trim_outer_whitespace(question, start, start + match.start())
    return start, end


def _expand_left_premodifiers(question: str, start: int) -> int:
    current = start
    saw_modifier = False
    while True:
        match = re.search(r"\b([A-Za-z][A-Za-z0-9'-]*)\s+$", question[:current])
        if not match:
            break
        word = match.group(1)
        lowered = word.lower()
        if lowered in DETERMINERS:
            if saw_modifier:
                current = match.start(1)
            break
        if lowered in LEFT_EXPANSION_BOUNDARIES or lowered in NONESSENTIAL_PREMODIFIERS:
            break
        if word[:1].isupper():
            break
        saw_modifier = True
        current = match.start(1)
    return current


def _trim_leading_mask_preserved_words(
    question: str,
    start: int,
    end: int,
    node: ExtractedNode,
) -> tuple[int, int]:
    current = start
    while current < end:
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9'-]*)(\s+)", question[current:end])
        if not match:
            break
        lowered = match.group(1).lower()
        if lowered in LEADING_FUNCTION_WORDS or (
            node.is_type_variable and lowered in NONESSENTIAL_PREMODIFIERS
        ):
            current += match.end()
            continue
        break
    return current, end


def _trim_outer_whitespace(question: str, start: int, end: int) -> tuple[int, int]:
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end - 1].isspace():
        end -= 1
    return start, end


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


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


def _pos_hint_base(node: ExtractedNode) -> str:
    combined = f"{node.semantic_type} {node.text}".lower()
    for key, base in POS_HINT_BASE_BY_TYPE.items():
        if re.search(rf"\b{re.escape(key)}\b", combined):
            return base
    return "SomeEntity"


def _pos_hint_label(base: str, index: int) -> str:
    return f"{base}{_letter_suffix(index)}"


def _letter_suffix(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    current = index
    while True:
        label = alphabet[current % len(alphabet)] + label
        current = current // len(alphabet) - 1
        if current < 0:
            break
    return label


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
