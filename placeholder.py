from __future__ import annotations

import re
from dataclasses import dataclass

from models import ExtractedNode, ExtractionResult, PlaceholderReplacement


@dataclass(frozen=True)
class SpanReplacement:
    start: int
    end: int
    placeholder: str
    text: str


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

