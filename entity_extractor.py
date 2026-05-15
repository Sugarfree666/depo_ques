from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from models import ExtractedNode, ExtractionResult
from prompts import ENTITY_EXTRACTION_SYSTEM, build_entity_extraction_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient

GREEK_ORDINALS = [
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
    "Eta",
    "Theta",
    "Iota",
    "Kappa",
]

ROLE_TO_SEMANTIC_TYPE = {
    "ceo": "Person",
    "director": "Person",
    "founder": "Person",
    "cofounder": "Person",
    "co-founder": "Person",
    "president": "Person",
    "author": "Person",
    "actor": "Person",
    "actress": "Person",
    "person": "Person",
    "people": "Person",
    "university": "University",
    "college": "University",
    "school": "School",
    "company": "Company",
    "corporation": "Company",
    "business": "Company",
    "organization": "Organization",
    "organisation": "Organization",
    "city": "City",
    "country": "Country",
    "nationality": "Nationality",
    "film": "Film",
    "movie": "Film",
    "region": "Region",
    "location": "Location",
    "concept": "Concept",
    "animal": "Animal",
    "food": "Food",
    "product": "Product",
    "age": "Age",
    "height": "Height",
    "length": "Length",
    "size": "Size",
    "date": "Date",
    "time": "Time",
    "population": "Population",
    "price": "Price",
    "value": "Value",
}

VALUE_MODIFIER_SEMANTIC_TYPES = {
    "date",
    "duration",
    "measure",
    "measurement",
    "number",
    "quantity",
    "time",
}
DURATION_UNITS = {
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "decade",
    "decades",
    "century",
    "centuries",
}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "hundred",
    "thousand",
    "million",
    "billion",
}
VALUE_MODIFIER_PREFIXES = {
    "about",
    "almost",
    "approximately",
    "around",
    "at least",
    "at most",
    "less than",
    "more than",
    "over",
    "under",
}
TEMPORAL_ATTRIBUTE_TYPES = {"date", "time"}
IMPLICIT_ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "than",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}
COMPARATIVE_ADVERBS_TO_SKIP = {
    "considerably",
    "even",
    "far",
    "much",
    "significantly",
    "slightly",
    "somewhat",
    "still",
}

IMPLICIT_ATTRIBUTE_CUES = {
    "older": ("age", "Age"),
    "oldest": ("age", "Age"),
    "younger": ("age", "Age"),
    "youngest": ("age", "Age"),
    "elder": ("age", "Age"),
    "eldest": ("age", "Age"),
    "taller": ("height", "Height"),
    "tallest": ("height", "Height"),
    "shorter": ("height", "Height"),
    "shortest": ("height", "Height"),
    "longer": ("length", "Length"),
    "longest": ("length", "Length"),
    "larger": ("size", "Size"),
    "largest": ("size", "Size"),
    "bigger": ("size", "Size"),
    "biggest": ("size", "Size"),
    "smaller": ("size", "Size"),
    "smallest": ("size", "Size"),
    "newer": ("date", "Date"),
    "newest": ("date", "Date"),
    "earlier": ("date", "Date"),
    "earliest": ("date", "Date"),
    "later": ("date", "Date"),
    "latest": ("date", "Date"),
    "faster": ("speed", "Speed"),
    "fastest": ("speed", "Speed"),
    "slower": ("speed", "Speed"),
    "slowest": ("speed", "Speed"),
    "heavier": ("weight", "Weight"),
    "heaviest": ("weight", "Weight"),
    "lighter": ("weight", "Weight"),
    "lightest": ("weight", "Weight"),
    "cheaper": ("price", "Price"),
    "cheapest": ("price", "Price"),
    "costlier": ("price", "Price"),
    "costliest": ("price", "Price"),
    "higher": ("value", "Value"),
    "highest": ("value", "Value"),
    "lower": ("value", "Value"),
    "lowest": ("value", "Value"),
}


class EntityExtractor:
    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def extract(self, question: str) -> ExtractionResult:
        payload = self.llm_client.chat_json(
            ENTITY_EXTRACTION_SYSTEM,
            build_entity_extraction_prompt(question),
        )
        entities = self._parse_nodes(payload.get("entities", []), "entity", question)
        type_variables = self._parse_nodes(
            payload.get("type_variables", payload.get("typeVariables", [])),
            "type_variable",
            question,
        )
        result = ExtractionResult(entities=entities, type_variables=type_variables)
        _resolve_implicit_type_variable_spans(question, result.type_variables)
        _infer_missing_implicit_type_variables(question, result)
        _filter_non_anchor_type_variables(question, result)
        _repair_duplicate_surface_spans(question, result.nodes)
        self._normalize_placeholders(result)
        return result

    def _parse_nodes(self, raw_nodes: Any, kind: str, question: str) -> list[ExtractedNode]:
        if not isinstance(raw_nodes, list):
            return []

        nodes: list[ExtractedNode] = []
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            semantic_type = str(raw.get("semantic_type", raw.get("type", ""))).strip()
            semantic_type = normalize_semantic_type(semantic_type or text, kind)
            placeholder = str(raw.get("placeholder", "")).strip()
            start = _coerce_int(raw.get("start"))
            end = _coerce_int(raw.get("end"))
            occurrence = _coerce_int(raw.get("occurrence"))
            start, end = resolve_span(question, text, start, end, occurrence)
            if kind == "type_variable" and not _valid_span_bounds(question, start, end):
                cue_text = str(raw.get("cue_text", raw.get("cue", ""))).strip()
                cue_start = _coerce_int(raw.get("cue_start"))
                cue_end = _coerce_int(raw.get("cue_end"))
                start, end = _resolve_cue_span(question, cue_text, cue_start, cue_end, text, semantic_type)
                if start is not None and end is not None:
                    occurrence = 0
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

    def _normalize_placeholders(self, result: ExtractionResult) -> None:
        used: set[str] = set()
        counters: dict[str, int] = defaultdict(int)

        for node in result.nodes:
            base = normalize_semantic_type(node.semantic_type, node.kind)
            if not _valid_placeholder(node.placeholder) or node.placeholder in used:
                node.placeholder = _next_unused_placeholder(base, counters, used)
            else:
                placeholder_base = _placeholder_base(node.placeholder)
                counters[placeholder_base] += 1
            used.add(node.placeholder)


def normalize_semantic_type(value: str, kind: str) -> str:
    text = re.sub(r"\s+", " ", value.strip())
    lowered = text.lower()
    for key, mapped in ROLE_TO_SEMANTIC_TYPE.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return mapped
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return "Entity" if kind == "entity" else "Variable"
    if kind == "entity" and len(words) > 3:
        return "Entity"
    return "".join(word[:1].upper() + word[1:] for word in words)


def resolve_span(
    question: str,
    text: str,
    start: int | None,
    end: int | None,
    occurrence: int | None,
) -> tuple[int | None, int | None]:
    if start is not None and end is not None and 0 <= start < end <= len(question):
        if question[start:end].lower() == text.lower():
            return start, end

    matches = list(re.finditer(re.escape(text), question, flags=re.IGNORECASE))
    if not matches:
        return start, end
    index = max((occurrence or 1) - 1, 0)
    if index >= len(matches):
        index = 0
    match = matches[index]
    return match.start(), match.end()


def _resolve_implicit_type_variable_spans(question: str, nodes: list[ExtractedNode]) -> None:
    for node in nodes:
        anchor = _resolve_implicit_attribute_anchor(question, node)
        if anchor is not None:
            node.text, node.semantic_type, node.start, node.end = anchor
            node.occurrence = 0
            continue
        if _valid_span(question, node):
            continue
        cue = _find_implicit_attribute_cue(question, node.text, node.semantic_type)
        if cue is None:
            continue
        node.start, node.end = cue
        node.occurrence = 0


def _infer_missing_implicit_type_variables(question: str, result: ExtractionResult) -> None:
    existing_keys = {
        _normalized_surface(node.text) for node in result.type_variables
    } | {
        _normalized_surface(node.semantic_type) for node in result.type_variables
    }
    for cue_word, (text, semantic_type), cue_start, cue_end in _iter_implicit_attribute_cues(question):
        text, semantic_type, start, end = _anchor_for_implicit_cue(
            question,
            cue_word,
            text,
            semantic_type,
            cue_start,
            cue_end,
        )
        key = _normalized_surface(text)
        if key in existing_keys or _normalized_surface(semantic_type) in existing_keys:
            continue
        result.type_variables.append(
            ExtractedNode(
                placeholder="",
                text=text,
                kind="type_variable",
                semantic_type=semantic_type,
                start=start,
                end=end,
                occurrence=0,
            )
        )
        existing_keys.add(key)
        existing_keys.add(_normalized_surface(semantic_type))


def _filter_non_anchor_type_variables(question: str, result: ExtractionResult) -> None:
    result.type_variables = [
        node for node in result.type_variables if not _is_modifier_value_type_variable(question, node)
    ]


def _is_modifier_value_type_variable(question: str, node: ExtractedNode) -> bool:
    if node.occurrence == 0:
        return False
    semantic_type = node.semantic_type.strip().lower()
    text = _normalized_surface(node.text)
    if semantic_type not in VALUE_MODIFIER_SEMANTIC_TYPES:
        return False
    if _is_direct_answer_type(question, text):
        return False
    return _looks_like_quantity_or_duration_value(text)


def _is_direct_answer_type(question: str, text: str) -> bool:
    if not text:
        return False
    prefix = _normalized_surface(question[:80])
    return bool(
        re.search(
            rf"^(what|which|when|how many|how much)\s+(?:\w+\s+){{0,3}}{re.escape(text)}\b",
            prefix,
        )
    )


def _looks_like_quantity_or_duration_value(text: str) -> bool:
    if not text:
        return False
    if any(text == prefix or text.startswith(prefix + " ") for prefix in VALUE_MODIFIER_PREFIXES):
        return True
    words = set(re.findall(r"[a-z0-9]+", text))
    has_number = bool(words & NUMBER_WORDS) or bool(re.search(r"\d", text))
    has_unit = bool(words & DURATION_UNITS)
    return has_number and has_unit


def _resolve_cue_span(
    question: str,
    cue_text: str,
    cue_start: int | None,
    cue_end: int | None,
    text: str,
    semantic_type: str,
) -> tuple[int | None, int | None]:
    if _valid_span_bounds(question, cue_start, cue_end):
        return cue_start, cue_end
    if cue_text:
        start, end = resolve_span(question, cue_text, None, None, None)
        if _valid_span_bounds(question, start, end):
            return start, end
    cue = _find_implicit_attribute_cue(question, text, semantic_type)
    if cue is not None:
        return cue
    return None, None


def _find_implicit_attribute_cue(
    question: str,
    text: str,
    semantic_type: str,
) -> tuple[int, int] | None:
    target_keys = {_normalized_surface(text), _normalized_surface(semantic_type)}
    for _, (cue_text, cue_type), start, end in _iter_implicit_attribute_cues(question):
        if _normalized_surface(cue_text) in target_keys or _normalized_surface(cue_type) in target_keys:
            return start, end
    return None


def _resolve_implicit_attribute_anchor(
    question: str,
    node: ExtractedNode,
) -> tuple[str, str, int, int] | None:
    direct_cue = IMPLICIT_ATTRIBUTE_CUES.get(_normalized_surface(node.text))
    if direct_cue is not None and _valid_span_bounds(question, node.start, node.end):
        return _anchor_for_implicit_cue(
            question,
            _normalized_surface(node.text),
            direct_cue[0],
            direct_cue[1],
            node.start or 0,
            node.end or 0,
        )

    span_cue = _cue_inside_span(question, node.start, node.end)
    if span_cue is not None:
        cue_word, (text, semantic_type), cue_start, cue_end = span_cue
        if _normalized_surface(node.semantic_type) == _normalized_surface(semantic_type) or (
            _normalized_surface(node.text) in {_normalized_surface(text), _normalized_surface(semantic_type)}
        ):
            return _anchor_for_implicit_cue(
                question,
                cue_word,
                text,
                semantic_type,
                cue_start,
                cue_end,
            )

    if _valid_span(question, node):
        return None

    cue = _implicit_attribute_cue_for_type(question, node.text, node.semantic_type)
    if cue is None:
        return None
    cue_word, (text, semantic_type), cue_start, cue_end = cue
    return _anchor_for_implicit_cue(
        question,
        cue_word,
        text,
        semantic_type,
        cue_start,
        cue_end,
    )


def _implicit_attribute_cue_for_type(
    question: str,
    text: str,
    semantic_type: str,
) -> tuple[str, tuple[str, str], int, int] | None:
    target_keys = {_normalized_surface(text), _normalized_surface(semantic_type)}
    for cue_word, (cue_text, cue_type), start, end in _iter_implicit_attribute_cues(question):
        if _normalized_surface(cue_text) in target_keys or _normalized_surface(cue_type) in target_keys:
            return cue_word, (cue_text, cue_type), start, end
    return None


def _cue_inside_span(
    question: str,
    start: int | None,
    end: int | None,
) -> tuple[str, tuple[str, str], int, int] | None:
    if not _valid_span_bounds(question, start, end):
        return None
    for cue_word, attribute, cue_start, cue_end in _iter_implicit_attribute_cues(question):
        if (start or 0) <= cue_start and cue_end <= (end or 0):
            return cue_word, attribute, cue_start, cue_end
    return None


def _anchor_for_implicit_cue(
    question: str,
    cue_word: str,
    text: str,
    semantic_type: str,
    cue_start: int,
    cue_end: int,
) -> tuple[str, str, int, int]:
    if _normalized_surface(semantic_type) in TEMPORAL_ATTRIBUTE_TYPES:
        predicate_anchor = _previous_predicate_anchor(question, cue_start)
        if predicate_anchor is not None:
            predicate_text, start, end = predicate_anchor
            return predicate_text, semantic_type, start, end
    return text, semantic_type, cue_start, cue_end


def _previous_predicate_anchor(question: str, cue_start: int) -> tuple[str, int, int] | None:
    matches = list(re.finditer(r"\b[A-Za-z][A-Za-z'-]*\b", question[:cue_start]))
    for match in reversed(matches):
        word = match.group(0)
        lowered = word.lower()
        if lowered in COMPARATIVE_ADVERBS_TO_SKIP:
            continue
        if lowered in IMPLICIT_ANCHOR_STOPWORDS:
            return None
        if word[:1].isupper():
            return None
        return word, match.start(), match.end()
    return None


def _iter_implicit_attribute_cues(question: str) -> list[tuple[str, tuple[str, str], int, int]]:
    cues: list[tuple[str, tuple[str, str], int, int]] = []
    for match in re.finditer(r"\b[A-Za-z][A-Za-z'-]*\b", question):
        word = match.group(0).lower()
        attribute = IMPLICIT_ATTRIBUTE_CUES.get(word)
        if attribute is None:
            continue
        cues.append((word, attribute, match.start(), match.end()))
    return cues


def _valid_span_bounds(question: str, start: int | None, end: int | None) -> bool:
    return start is not None and end is not None and 0 <= start < end <= len(question)


def _repair_duplicate_surface_spans(question: str, nodes: list[ExtractedNode]) -> None:
    grouped: dict[tuple[str, str], list[ExtractedNode]] = defaultdict(list)
    for node in nodes:
        grouped[(node.kind, _normalized_surface(node.text))].append(node)

    for (_, normalized_text), group in grouped.items():
        if len(group) <= 1 or not normalized_text:
            continue

        matches = list(re.finditer(re.escape(group[0].text), question, flags=re.IGNORECASE))
        if len(matches) < len(group):
            continue

        spans = [(node.start, node.end) for node in group]
        occurrences = [node.occurrence for node in group if node.occurrence is not None]
        has_duplicate_span = len(set(spans)) < len(spans)
        has_duplicate_occurrence = len(set(occurrences)) < len(occurrences)
        has_missing_span = any(not _valid_span(question, node) for node in group)

        if has_duplicate_span or has_duplicate_occurrence or has_missing_span:
            for index, node in enumerate(group):
                match = matches[index]
                node.start = match.start()
                node.end = match.end()
                node.occurrence = index + 1
            continue

        for node in group:
            for index, match in enumerate(matches, start=1):
                if node.start == match.start() and node.end == match.end():
                    node.occurrence = index
                    break


def _valid_span(question: str, node: ExtractedNode) -> bool:
    return (
        node.start is not None
        and node.end is not None
        and 0 <= node.start < node.end <= len(question)
        and question[node.start : node.end].lower() == node.text.lower()
    )


def _normalized_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _next_placeholder(base: str, counters: dict[str, int]) -> str:
    index = counters[base]
    counters[base] += 1
    suffix = GREEK_ORDINALS[index] if index < len(GREEK_ORDINALS) else f"Omega{index + 1}"
    return f"{base}{suffix}"


def _next_unused_placeholder(base: str, counters: dict[str, int], used: set[str]) -> str:
    placeholder = _next_placeholder(base, counters)
    while placeholder in used:
        placeholder = _next_placeholder(base, counters)
    return placeholder


def _placeholder_base(placeholder: str) -> str:
    for suffix in GREEK_ORDINALS:
        if placeholder.endswith(suffix):
            return placeholder[: -len(suffix)] or "Entity"
    return re.sub(r"[^A-Za-z0-9]", "", placeholder) or "Entity"


def _valid_placeholder(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9]*", value))


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
