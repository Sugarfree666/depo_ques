from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from llm_client import LLMClient
from models import ExtractedNode, ExtractionResult
from prompts import ENTITY_EXTRACTION_SYSTEM, build_entity_extraction_prompt

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
}


class EntityExtractor:
    def __init__(self, llm_client: LLMClient) -> None:
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
