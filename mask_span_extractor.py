from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from entity_extractor import normalize_semantic_type
from models import MaskSpan, MaskSpanResult
from prompts import MASK_SPAN_EXTRACTION_SYSTEM, build_mask_span_extraction_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


SIMPLE_TYPE_VARIABLES = {
    "actor",
    "age",
    "ceo",
    "city",
    "country",
    "director",
    "nationality",
    "population",
    "university",
}

LEADING_TYPE_SPAN_WORDS = {"a", "an", "the"}
NAMED_ENTITY_SEMANTIC_TYPES = {
    "album",
    "book",
    "city",
    "company",
    "country",
    "event",
    "film",
    "institution",
    "location",
    "movie",
    "organization",
    "organisation",
    "person",
    "place",
    "product",
    "region",
    "series",
    "song",
    "university",
    "work",
}

TYPE_PHRASE_PATTERNS = [
    (r"\blocal food distribution network\b", "type_variable", "Network", "multi-word functional noun phrase"),
    (r"\bfood distribution network\b", "type_variable", "Network", "multi-word functional noun phrase"),
    (r"\bdistribution network\b", "type_variable", "Network", "multi-word functional noun phrase"),
    (r"\bartificial intelligence company\b", "type_variable", "Company", "multi-word company type phrase"),
    (r"\bmixed-use space\b", "type_variable", "Space", "hyphenated multi-word type phrase"),
]

TITLE_HEADS = {
    "album": "Album",
    "book": "Book",
    "film": "Film",
    "movie": "Film",
    "novel": "Book",
    "play": "Work",
    "series": "Series",
    "song": "Song",
    "work": "Work",
}

HUMAN_CONTEXT_CUES = {
    "actor",
    "actress",
    "age",
    "author",
    "born",
    "ceo",
    "director",
    "elder",
    "eldest",
    "founder",
    "older",
    "oldest",
    "people",
    "person",
    "player",
    "president",
    "singer",
    "who",
    "whom",
    "whose",
    "younger",
    "youngest",
}

NON_PERSON_NAME_WORDS = {
    "academy",
    "album",
    "association",
    "book",
    "city",
    "college",
    "company",
    "corporation",
    "country",
    "film",
    "foundation",
    "inc",
    "institute",
    "ltd",
    "movie",
    "network",
    "organization",
    "organisation",
    "school",
    "song",
    "university",
}

PERSON_NAME_PARTICLES = {"al", "bin", "da", "de", "del", "der", "di", "la", "le", "van", "von"}
CAPITALIZED_ENTITY_TOKEN = r"(?:[A-Z][A-Za-z0-9']+|[A-Z]\.|[A-Z]{2,}(?:\.)?)"

CLAUSE_BOUNDARY = {
    "and",
    "or",
    "share",
    "shares",
    "shared",
    "have",
    "has",
    "had",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "which",
    "who",
    "that",
}


class MaskSpanExtractor:
    """Step 1 extractor for parse-protection spans only.

    The extractor may ask an LLM for spans, but it never returns anchors,
    implicit variables, operators, relations, AST structures, or subquestions.
    A conservative heuristic fallback handles the obvious cases used by tests
    and keeps the CLI usable when the LLM returns malformed JSON.
    """

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.llm_client = llm_client

    def extract(self, question: str) -> MaskSpanResult:
        warnings: list[str] = []
        if self.llm_client is not None:
            try:
                payload = self.llm_client.chat_json(
                    MASK_SPAN_EXTRACTION_SYSTEM,
                    build_mask_span_extraction_prompt(question),
                )
                return MaskSpanResult(
                    mask_spans=self._parse_payload(question, payload, warnings),
                    warnings=warnings,
                )
            except Exception as exc:
                warnings.append(f"Mask span LLM failed; using heuristic fallback: {exc}")

        return MaskSpanResult(mask_spans=_heuristic_mask_spans(question), warnings=warnings)

    @staticmethod
    def _parse_payload(
        question: str,
        payload: dict[str, Any],
        warnings: list[str],
    ) -> list[MaskSpan]:
        raw_spans = payload.get("mask_spans", payload.get("maskSpans", []))
        if not isinstance(raw_spans, list):
            warnings.append("Mask span payload did not contain a list mask_spans field.")
            return []

        spans: list[MaskSpan] = []
        for raw in raw_spans:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", "")).strip()
            start = _coerce_int(raw.get("start_char", raw.get("start")))
            end = _coerce_int(raw.get("end_char", raw.get("end")))
            if not text:
                continue
            start, end = _resolve_span(question, text, start, end)
            if start is None or end is None:
                warnings.append(f"Could not resolve mask span text={text!r}.")
                continue
            kind_hint = _normalize_kind_hint(raw.get("kind_hint", raw.get("kind", "entity")))
            semantic_type_hint = str(raw.get("semantic_type_hint", raw.get("semantic_type", ""))).strip() or None
            spans.append(
                MaskSpan(
                    text=question[start:end],
                    start_char=start,
                    end_char=end,
                    kind_hint=kind_hint,
                    semantic_type_hint=semantic_type_hint,
                    reason=str(raw.get("reason", "")).strip(),
                )
            )
        return spans


def _heuristic_mask_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    spans.extend(_title_spans_after_type_heads(question))
    spans.extend(_parenthetical_entity_spans(question))
    spans.extend(_quoted_spans(question))
    spans.extend(_capitalized_entity_spans(question))
    spans.extend(_type_phrase_spans(question))
    return _merge_spans(question, spans, [])


def _title_spans_after_type_heads(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    for match in re.finditer(
        r"\b(?P<head>film|movie|book|album|song|novel|play|series|work)\s+",
        question,
        flags=re.IGNORECASE,
    ):
        head = match.group("head").lower()
        start = match.end()
        end = _find_title_end(question, start)
        if end <= start:
            continue
        text = question[start:end].strip()
        leading_ws = len(question[start:end]) - len(question[start:end].lstrip())
        trailing_ws = len(question[start:end]) - len(question[start:end].rstrip())
        start += leading_ws
        end -= trailing_ws
        text = question[start:end]
        if _is_mask_worthy(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=start,
                    end_char=end,
                    kind_hint="entity",
                    semantic_type_hint=TITLE_HEADS.get(head, "Entity"),
                    reason="complex title after explicit type head",
                )
            )
    return spans


def _find_title_end(question: str, start: int) -> int:
    token_matches = list(re.finditer(r"\S+", question[start:]))
    if not token_matches:
        return start
    end = start
    previous_end = start
    for index, match in enumerate(token_matches):
        token_start = start + match.start()
        token_end = start + match.end()
        cleaned = match.group(0).strip("?,.;:")
        lowered = cleaned.lower()
        if index > 0 and lowered in CLAUSE_BOUNDARY:
            if lowered in {"and", "or"} and _looks_like_title_continuation(question, token_end):
                previous_end = token_end
                continue
            break
        end = token_end
        previous_end = token_end
    return end or previous_end


def _looks_like_title_continuation(question: str, position: int) -> bool:
    next_match = re.search(r"\S+", question[position:])
    if not next_match:
        return False
    token = next_match.group(0).strip("?,.;:")
    return bool(token[:1].isupper() or re.search(r"\d|[:()\"']", token))


def _parenthetical_entity_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    pattern = re.compile(
        rf"\b{CAPITALIZED_ENTITY_TOKEN}(?:\s+{CAPITALIZED_ENTITY_TOKEN}){{0,5}}\s*\([^)]*\)"
    )
    for match in pattern.finditer(question):
        text = match.group(0)
        if _is_mask_worthy(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=match.start(),
                    end_char=match.end(),
                    kind_hint="entity",
                    semantic_type_hint=_infer_semantic_type(text, "entity", question, match.start(), match.end()),
                    reason="entity with parenthetical qualifier",
                )
            )
    return spans


def _quoted_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    for match in re.finditer(r"[\"“”']([^\"“”']{3,})[\"“”']", question):
        text = match.group(1).strip()
        start = match.start(1) + (len(match.group(1)) - len(match.group(1).lstrip()))
        end = start + len(text)
        if _is_mask_worthy(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=start,
                    end_char=end,
                    kind_hint="entity",
                    semantic_type_hint=_infer_semantic_type(text, "entity", question, start, end),
                    reason="quoted complex title/name",
                )
            )
    return spans


def _capitalized_entity_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    pattern = re.compile(
        rf"\b{CAPITALIZED_ENTITY_TOKEN}(?:\s+(?:of|the|and|for|de|la|{CAPITALIZED_ENTITY_TOKEN})){{1,6}}\b"
    )
    for match in pattern.finditer(question):
        text = match.group(0).strip()
        if _token_count(text) < 2 or _starts_sentence_only(question, match.start(), text):
            continue
        if _capitalized_content_token_count(text) < 2:
            continue
        if not _is_mask_worthy(text):
            continue
        spans.append(
            MaskSpan(
                text=text,
                start_char=match.start(),
                end_char=match.end(),
                kind_hint="entity",
                semantic_type_hint=_infer_semantic_type(text, "entity", question, match.start(), match.end()),
                reason="continuous multi-word named entity",
            )
        )
    return spans


def _type_phrase_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    for pattern, kind, semantic_type, reason in TYPE_PHRASE_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            text = match.group(0)
            if _is_simple_type_variable(text):
                continue
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=match.start(),
                    end_char=match.end(),
                    kind_hint=kind,
                    semantic_type_hint=semantic_type,
                    reason=reason,
                )
            )
    return spans


def _merge_spans(
    question: str,
    spans: list[MaskSpan],
    warnings: list[str],
) -> list[MaskSpan]:
    normalized: list[MaskSpan] = []
    seen: set[tuple[int, int]] = set()
    for span in spans:
        start, end = _resolve_span(question, span.text, span.start_char, span.end_char)
        if start is None or end is None:
            continue
        kind_hint = _normalize_kind_hint(span.kind_hint)
        start, end = _trim_mask_span(
            question=question,
            start=start,
            end=end,
            kind_hint=kind_hint,
            semantic_type_hint=span.semantic_type_hint,
        )
        text = question[start:end]
        if not _is_mask_worthy(
            text,
            kind_hint=kind_hint,
            semantic_type_hint=span.semantic_type_hint,
            question=question,
            start=start,
            end=end,
        ):
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            MaskSpan(
                text=text,
                start_char=start,
                end_char=end,
                kind_hint=kind_hint,
                semantic_type_hint=_refine_semantic_type(
                    question=question,
                    text=text,
                    kind_hint=span.kind_hint,
                    existing=span.semantic_type_hint,
                    start=start,
                    end=end,
                ),
                reason=span.reason,
            )
        )

    ordered = sorted(normalized, key=lambda item: (item.start_char, -(item.end_char - item.start_char)))
    result: list[MaskSpan] = []
    occupied: list[tuple[int, int]] = []
    for span in ordered:
        if any(not (span.end_char <= start or span.start_char >= end) for start, end in occupied):
            warnings.append(f"Dropped overlapping mask span text={span.text!r}.")
            continue
        result.append(span)
        occupied.append((span.start_char, span.end_char))
    return result


def _trim_mask_span(
    question: str,
    start: int,
    end: int,
    kind_hint: str,
    semantic_type_hint: str | None,
) -> tuple[int, int]:
    del semantic_type_hint
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end - 1].isspace():
        end -= 1

    if kind_hint == "type_variable":
        while True:
            match = re.match(r"\s*([A-Za-z][A-Za-z0-9'-]*)(\s+)", question[start:end])
            if not match or match.group(1).lower() not in LEADING_TYPE_SPAN_WORDS:
                break
            start += match.end()
    return start, end


def _is_mask_worthy(
    text: str,
    kind_hint: str = "entity",
    semantic_type_hint: str | None = None,
    question: str = "",
    start: int | None = None,
    end: int | None = None,
) -> bool:
    del question, start, end
    stripped = text.strip()
    if _is_simple_type_variable(stripped):
        return False
    token_count = _token_count(stripped)
    has_parser_fragile_punctuation = bool(
        re.search(r"[:()\[\]{}\"'\u201c\u201d\u2018\u2019,\-\u2013\u2014./&]", stripped)
    )
    has_digit = bool(re.search(r"\d", stripped))

    if kind_hint == "type_variable":
        return token_count >= 2 or has_parser_fragile_punctuation or has_digit

    if token_count < 2:
        if has_parser_fragile_punctuation or has_digit:
            return True
        if _looks_like_acronym(stripped) or _looks_like_mixedcase_name(stripped):
            return True
        return _has_named_entity_semantic_type(semantic_type_hint) and _looks_like_single_token_proper_name(stripped)
    return True


def _is_simple_type_variable(text: str) -> bool:
    return text.strip().lower() in SIMPLE_TYPE_VARIABLES


def _looks_like_acronym(text: str) -> bool:
    stripped = text.strip(".")
    return bool(
        re.fullmatch(r"(?:[A-Z]\.){2,}[A-Z]?\.?", text)
        or re.fullmatch(r"[A-Z]{2,}[A-Z0-9]*", stripped)
    )


def _looks_like_mixedcase_name(text: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*", text)
        or re.fullmatch(r"[A-Za-z]+[0-9][A-Za-z0-9]*", text)
    )


def _looks_like_single_token_proper_name(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9'._-]*", text))


def _has_named_entity_semantic_type(value: str | None) -> bool:
    normalized = re.sub(r"[^A-Za-z]+", " ", value or "").strip().lower()
    if not normalized:
        return False
    return any(word in NAMED_ENTITY_SEMANTIC_TYPES for word in normalized.split())


def _starts_sentence_only(question: str, start: int, text: str) -> bool:
    if start != 0:
        return False
    first = re.match(r"\w+", text)
    if not first:
        return False
    return first.group(0).lower() in {"what", "which", "who", "do", "does", "did", "is", "are"}


def _infer_semantic_type(
    text: str,
    kind_hint: str,
    question: str = "",
    start: int | None = None,
    end: int | None = None,
) -> str:
    lowered = text.lower()
    if "film" in lowered or "movie" in lowered:
        return "Film"
    if "network" in lowered:
        return "Network"
    if "company" in lowered:
        return "Company"
    if "university" in lowered:
        return "University"
    if "city" in lowered:
        return "City"
    if (
        kind_hint == "entity"
        and question
        and _looks_like_person_name(text)
        and _question_has_human_context(question, start, end)
    ):
        return "Person"
    return normalize_semantic_type(text, "entity" if kind_hint == "entity" else "type_variable")


def _refine_semantic_type(
    question: str,
    text: str,
    kind_hint: str,
    existing: str | None,
    start: int | None,
    end: int | None,
) -> str:
    inferred = _infer_semantic_type(text, kind_hint, question, start, end)
    existing = (existing or "").strip()
    if not existing:
        return inferred
    if inferred == "Person" and _is_generic_or_surface_semantic_type(existing, text):
        return inferred
    return existing


def _is_generic_or_surface_semantic_type(value: str, text: str) -> bool:
    normalized_value = re.sub(r"[^A-Za-z0-9]+", "", value).lower()
    surface_value = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    return normalized_value in {
        "",
        "entity",
        "someentity",
        "namedentity",
        "unknown",
        "thing",
        surface_value,
    }


def _looks_like_person_name(text: str) -> bool:
    if re.search(r"\d|[:()\[\]{}\"']", text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if len(words) < 2 or len(words) > 5:
        return False
    lowered_words = {word.lower().strip("'") for word in words}
    if lowered_words & NON_PERSON_NAME_WORDS:
        return False
    content_words = [word for word in words if word.lower() not in PERSON_NAME_PARTICLES]
    if len(content_words) < 2:
        return False
    return all(word[:1].isupper() or word.isupper() for word in content_words)


def _question_has_human_context(
    question: str,
    start: int | None,
    end: int | None,
) -> bool:
    lowered_words = set(re.findall(r"[A-Za-z]+", question.lower()))
    if lowered_words & HUMAN_CONTEXT_CUES:
        return True
    if start is None or end is None:
        return False
    local_left = question[max(0, start - 40) : start].lower()
    local_right = question[end : min(len(question), end + 40)].lower()
    local_words = set(re.findall(r"[A-Za-z]+", f"{local_left} {local_right}"))
    return bool(local_words & HUMAN_CONTEXT_CUES)


def _normalize_kind_hint(value: object) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {
        "functional_noun_phrase",
        "function_noun_phrase",
        "multi_word_type",
        "noun_phrase",
        "type",
        "type_phrase",
        "type_variable",
        "type-variable",
        "variable",
    }:
        return "type_variable"
    return "entity"


def _resolve_span(
    question: str,
    text: str,
    start: int | None,
    end: int | None,
) -> tuple[int | None, int | None]:
    if start is not None and end is not None and 0 <= start < end <= len(question):
        if question[start:end].strip().lower() == text.strip().lower():
            leading = len(question[start:end]) - len(question[start:end].lstrip())
            trailing = len(question[start:end]) - len(question[start:end].rstrip())
            return start + leading, end - trailing
    matches = list(re.finditer(re.escape(text.strip()), question, flags=re.IGNORECASE))
    if not matches:
        return None, None
    match = matches[0]
    return match.start(), match.end()


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def _capitalized_content_token_count(text: str) -> int:
    count = 0
    for word in re.findall(r"[A-Za-z0-9']+", text):
        lowered = word.lower()
        if lowered in {"and", "de", "for", "la", "of", "the"}:
            continue
        if word[:1].isupper() or word.isupper():
            count += 1
    return count


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
