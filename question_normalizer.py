from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterable

from models import SemanticNormalizationResult
from prompts import SEMANTIC_QUESTION_NORMALIZATION_SYSTEM, build_semantic_question_normalization_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


PLACEHOLDER_BASES = (
    "Album",
    "Age",
    "Book",
    "City",
    "Company",
    "Country",
    "Date",
    "Entity",
    "Film",
    "Institution",
    "Location",
    "Movie",
    "Nationality",
    "Network",
    "Organization",
    "Person",
    "Population",
    "Region",
    "Series",
    "SomeEntity",
    "Song",
    "Space",
    "System",
    "Time",
    "University",
    "Variable",
    "Work",
)
PLACEHOLDER_RE = re.compile(
    r"\b(?:X\d+(?:_[A-Za-z0-9]+)?|(?:"
    + "|".join(sorted(PLACEHOLDER_BASES, key=len, reverse=True))
    + r")(?:[A-Z][A-Za-z0-9]*|\d+))\b"
)

QUESTION_START_WORDS = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "how",
    "is",
    "may",
    "might",
    "must",
    "should",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "will",
    "would",
}


class SemanticQuestionNormalizer:
    """LLM-backed first step for parser-friendly semantic normalization."""

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.llm_client = llm_client

    def normalize(
        self,
        question: str,
        placeholders: Iterable[str] | None = None,
    ) -> SemanticNormalizationResult:
        original_question = _normalize_space(question)
        explicit_placeholders = list(dict.fromkeys(placeholders or _extract_placeholders(original_question)))
        warnings: list[str] = []

        if self.llm_client is None:
            return SemanticNormalizationResult(
                original_question=original_question,
                normalized_question=original_question,
                changed=False,
                warnings=["Semantic normalization LLM unavailable; using original question."],
            )

        payload: dict[str, Any] = {}
        try:
            payload = self.llm_client.chat_json(
                SEMANTIC_QUESTION_NORMALIZATION_SYSTEM,
                build_semantic_question_normalization_prompt(
                    question=original_question,
                    placeholders=explicit_placeholders,
                ),
            )
        except Exception as exc:
            warnings.append(f"Semantic normalization LLM failed; using original question: {exc}")
            return SemanticNormalizationResult(
                original_question=original_question,
                normalized_question=original_question,
                changed=False,
                warnings=warnings,
                raw_payload=payload or None,
            )

        candidate = _clean_candidate_question(_candidate_from_payload(payload))
        added_type_variables = _parse_added_type_variables(payload.get("added_type_variables", []))
        if not candidate:
            warnings.append("Semantic normalization returned an empty question; using original question.")
            return SemanticNormalizationResult(
                original_question=original_question,
                normalized_question=original_question,
                changed=False,
                added_type_variables=[],
                warnings=warnings,
                raw_payload=payload or None,
            )

        return SemanticNormalizationResult(
            original_question=original_question,
            normalized_question=candidate,
            changed=_normalize_for_compare(candidate) != _normalize_for_compare(original_question),
            added_type_variables=added_type_variables,
            warnings=warnings,
            raw_payload=payload or None,
        )


def _candidate_from_payload(payload: dict[str, Any]) -> str:
    for key in ("normalized_question", "normalizedQuestion", "question"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _parse_added_type_variables(raw_items: Any) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        text = _normalize_space(str(raw.get("text", "")))
        trigger_text = _normalize_space(str(raw.get("trigger_text", raw.get("trigger", ""))))
        reason = _normalize_space(str(raw.get("reason", "")))
        if not text or not trigger_text:
            continue
        result.append({"text": text, "trigger_text": trigger_text, "reason": reason})
    return result


def _clean_candidate_question(candidate: str) -> str:
    cleaned = _normalize_space(candidate.strip().strip("\"'"))
    if not cleaned:
        return ""
    if cleaned.endswith(".") and _looks_like_question(cleaned[:-1] + "?"):
        cleaned = cleaned[:-1] + "?"
    if not cleaned.endswith("?") and _looks_like_question(cleaned):
        cleaned += "?"
    return cleaned


def _looks_like_question(text: str) -> bool:
    first = _first_word(text)
    return first in QUESTION_START_WORDS or bool(
        re.match(r"^\s*(?:in|on|at|from|to|for)\s+(?:what|which)\b", text, flags=re.IGNORECASE)
    )


def _extract_placeholders(text: str) -> list[str]:
    return [match.group(0) for match in PLACEHOLDER_RE.finditer(text)]


def _first_word(text: str) -> str:
    match = re.search(r"[A-Za-z]+", text)
    return match.group(0).lower() if match else ""


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_compare(text: str) -> str:
    return _normalize_space(text).lower()
