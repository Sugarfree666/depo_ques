from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import networkx as nx

from graph_builder import restored_dependency_edges, restored_dependency_tokens
from models import (
    AnchorSelectionResult,
    DependencyParse,
    MaskReplacement,
    RestoredGraphNodeCandidate,
    SelectedAnchor,
)
from prompts import ANCHOR_SELECTION_SYSTEM, build_anchor_selection_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


ILLEGAL_ANCHOR_TEXT = {
    "after",
    "and",
    "before",
    "both",
    "different",
    "either",
    "first",
    "highest",
    "larger",
    "largest",
    "last",
    "less",
    "more",
    "older",
    "or",
    "same",
    "smaller",
    "youngest",
    "younger",
}

ALLOWED_ANCHOR_KINDS = {"entity", "type_variable"}


class AnchorSelector:
    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.llm_client = llm_client

    def select(
        self,
        original_question: str,
        masked_question: str,
        replacement: MaskReplacement,
        dependency_parse: DependencyParse,
        weighted_graph: nx.Graph,
        restored_graph_node_candidates: list[RestoredGraphNodeCandidate],
    ) -> AnchorSelectionResult:
        del replacement, weighted_graph
        warnings: list[str] = []
        payload: dict[str, Any] = {}
        if self.llm_client is not None:
            try:
                payload = self.llm_client.chat_json(
                    ANCHOR_SELECTION_SYSTEM,
                    build_anchor_selection_prompt(
                        original_question=original_question,
                        masked_question=masked_question,
                        restored_graph_node_candidates=[
                            candidate.to_llm_view()
                            for candidate in restored_graph_node_candidates
                            if _candidate_visible_to_llm(candidate)
                        ],
                        restored_dependency_tokens=restored_dependency_tokens(
                            dependency_parse,
                            restored_graph_node_candidates,
                        ),
                        restored_dependency_edges=restored_dependency_edges(
                            dependency_parse,
                            restored_graph_node_candidates,
                        ),
                    ),
                )
            except Exception as exc:
                warnings.append(f"Anchor selection LLM failed; using fallback: {exc}")
        else:
            warnings.append("Anchor selection LLM unavailable; using fallback.")

        selected = self._parse_and_validate(payload, restored_graph_node_candidates, warnings)
        if not selected:
            selected = _fallback_anchors(restored_graph_node_candidates, warnings)
        return AnchorSelectionResult(
            selected_anchors=selected,
            warnings=warnings,
            raw_payload=payload or None,
        )

    def _parse_and_validate(
        self,
        payload: dict[str, Any],
        candidates: list[RestoredGraphNodeCandidate],
        warnings: list[str],
    ) -> list[SelectedAnchor]:
        raw_items = payload.get("selected_anchors", payload.get("anchors", []))
        if not isinstance(raw_items, list):
            return []
        candidate_by_id = {candidate.node_id: candidate for candidate in candidates}
        selected: list[SelectedAnchor] = []
        used: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            node_id = str(raw.get("node_id", "")).strip()
            text = str(raw.get("text", "")).strip()
            if node_id not in candidate_by_id:
                repaired = _unique_candidate_for_text(text, candidates)
                if repaired is None:
                    warnings.append(f"Dropped anchor with invalid node_id={node_id!r} text={text!r}.")
                    continue
                node_id = repaired.node_id
            candidate = candidate_by_id[node_id]
            anchor_kind = _canonical_anchor_kind(raw.get("anchor_kind", raw.get("kind", "")))
            if anchor_kind not in ALLOWED_ANCHOR_KINDS:
                warnings.append(
                    f"Dropped node_id={node_id} because anchor_kind={raw.get('anchor_kind')!r} is not allowed."
                )
                continue
            if _is_illegal_anchor(candidate, text or candidate.display_text):
                warnings.append(f"Dropped illegal cue/operator anchor text={text or candidate.display_text!r}.")
                continue
            if node_id in used:
                continue
            used.add(node_id)
            selected.append(
                SelectedAnchor(
                    node_id=candidate.node_id,
                    graph_text=candidate.graph_text,
                    restored_text=candidate.restored_text,
                    display_text=candidate.display_text,
                    anchor_kind=anchor_kind,
                    source="graph_node",
                    token_index=candidate.token_index,
                    placeholder=candidate.placeholder,
                    semantic_type_hint=candidate.semantic_type_hint,
                    reason=str(raw.get("reason", "")).strip(),
                )
            )
        return selected


def _candidate_visible_to_llm(candidate: RestoredGraphNodeCandidate) -> bool:
    if candidate.kind_hint == "context":
        return False
    return True


def _fallback_anchors(
    candidates: list[RestoredGraphNodeCandidate],
    warnings: list[str],
) -> list[SelectedAnchor]:
    selected: list[SelectedAnchor] = []
    for candidate in candidates:
        if candidate.is_mask_placeholder and candidate.kind_hint == "entity_candidate":
            _append_fallback_anchor(selected, candidate, "entity")
    for candidate in candidates:
        if candidate.kind_hint != "type_variable_candidate":
            continue
        if _is_illegal_anchor(candidate, candidate.display_text):
            continue
        _append_fallback_anchor(selected, candidate, "type_variable")
    if selected:
        warnings.append("Used conservative explicit-anchor fallback.")
    return selected


def _append_fallback_anchor(
    selected: list[SelectedAnchor],
    candidate: RestoredGraphNodeCandidate,
    anchor_kind: str,
) -> None:
    if any(item.node_id == candidate.node_id for item in selected):
        return
    selected.append(
        SelectedAnchor(
            node_id=candidate.node_id,
            graph_text=candidate.graph_text,
            restored_text=candidate.restored_text,
            display_text=candidate.display_text,
            anchor_kind=anchor_kind,
            source="graph_node",
            token_index=candidate.token_index,
            placeholder=candidate.placeholder,
            semantic_type_hint=candidate.semantic_type_hint,
            reason="conservative fallback",
        )
    )


def _canonical_anchor_kind(value: object) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"entity", "named_entity"}:
        return "entity"
    if lowered in {"type", "type_variable", "type-variable", "variable"}:
        return "type_variable"
    return lowered


def _unique_candidate_for_text(
    text: str,
    candidates: list[RestoredGraphNodeCandidate],
) -> RestoredGraphNodeCandidate | None:
    if not text:
        return None
    normalized = _normalize(text)
    matches = [
        candidate
        for candidate in candidates
        if _normalize(candidate.display_text) == normalized or _normalize(candidate.restored_text) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _is_illegal_anchor(candidate: RestoredGraphNodeCandidate, text: str) -> bool:
    normalized = _normalize(text)
    if normalized in ILLEGAL_ANCHOR_TEXT:
        return True
    if candidate.kind_hint == "cue_candidate":
        return True
    return False


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
