from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import networkx as nx
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
PREDICATE_ONLY_ANCHOR_TEXT = {
    "develop",
    "developed",
    "develops",
    "graduate",
    "graduated",
    "graduates",
    "located",
    "share",
    "shared",
    "shares",
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
        del masked_question, replacement, dependency_parse
        warnings: list[str] = []
        payload: dict[str, Any] = {}
        if self.llm_client is not None:
            try:
                payload = self.llm_client.chat_json(
                    ANCHOR_SELECTION_SYSTEM,
                    build_anchor_selection_prompt(
                        original_question=original_question,
                        restored_graph_node_candidates=[
                            candidate.to_llm_view()
                            for candidate in restored_graph_node_candidates
                        ],
                    ),
                )
            except Exception as exc:
                warnings.append(f"Anchor selection LLM failed; using fallback: {exc}")
        else:
            warnings.append("Anchor selection LLM unavailable; using fallback.")

        selected = self._parse_and_validate(payload, restored_graph_node_candidates, warnings)
        if not selected:
            selected = _fallback_anchors(restored_graph_node_candidates, warnings)
        else:
            selected = _complete_explicit_anchor_coverage(
                selected=selected,
                candidates=restored_graph_node_candidates,
                weighted_graph=weighted_graph,
                original_question=original_question,
                warnings=warnings,
            )
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
                warnings.append(f"Dropped illegal cue/operator/predicate anchor text={text or candidate.display_text!r}.")
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
    del candidate
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


def _complete_explicit_anchor_coverage(
    selected: list[SelectedAnchor],
    candidates: list[RestoredGraphNodeCandidate],
    weighted_graph: nx.Graph,
    original_question: str,
    warnings: list[str],
) -> list[SelectedAnchor]:
    completed = list(selected)
    selected_ids = {anchor.node_id for anchor in completed}

    def add(candidate: RestoredGraphNodeCandidate, anchor_kind: str, reason: str) -> None:
        if candidate.node_id in selected_ids:
            return
        if _is_illegal_anchor(candidate, candidate.display_text):
            return
        selected_ids.add(candidate.node_id)
        completed.append(_anchor_from_candidate(candidate, anchor_kind, reason))

    for candidate in candidates:
        if _is_explicit_entity_candidate(candidate):
            add(candidate, "entity", "programmatic completion: explicit entity candidate")

    for candidate in _focus_type_variable_candidates(original_question, candidates):
        add(candidate, "type_variable", "programmatic completion: explicit question focus")

    for candidate in _cue_scoped_type_variable_candidates(candidates):
        add(candidate, "type_variable", "programmatic completion: explicit attribute near operator cue")

    for candidate in _path_endpoint_type_variable_candidates(completed, candidates, weighted_graph):
        add(candidate, "type_variable", "programmatic completion: type-variable endpoint on anchor evidence path")

    if len(completed) > len(selected):
        warnings.append("Completed explicit anchor coverage with deterministic candidate constraints.")
    return completed


def _anchor_from_candidate(
    candidate: RestoredGraphNodeCandidate,
    anchor_kind: str,
    reason: str,
) -> SelectedAnchor:
    return SelectedAnchor(
        node_id=candidate.node_id,
        graph_text=candidate.graph_text,
        restored_text=candidate.restored_text,
        display_text=candidate.display_text,
        anchor_kind=anchor_kind,
        source="graph_node",
        token_index=candidate.token_index,
        placeholder=candidate.placeholder,
        semantic_type_hint=candidate.semantic_type_hint,
        reason=reason,
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


def _is_explicit_entity_candidate(candidate: RestoredGraphNodeCandidate) -> bool:
    return candidate.kind_hint == "entity_candidate" and not _is_illegal_anchor(candidate, candidate.display_text)


def _is_completable_type_variable_candidate(candidate: RestoredGraphNodeCandidate) -> bool:
    return candidate.kind_hint == "type_variable_candidate" and not _is_illegal_anchor(candidate, candidate.display_text)


def _focus_type_variable_candidates(
    original_question: str,
    candidates: list[RestoredGraphNodeCandidate],
) -> list[RestoredGraphNodeCandidate]:
    del original_question
    ordered = sorted(candidates, key=lambda candidate: candidate.token_index)
    result: list[RestoredGraphNodeCandidate] = []
    focus_words = {"which", "what"}
    stop_words = {"did", "does", "do", "is", "are", "was", "were", "has", "have", "had"}
    for index, candidate in enumerate(ordered):
        if _normalize(candidate.display_text) not in focus_words:
            continue
        for neighbor in ordered[index + 1 : index + 5]:
            normalized = _normalize(neighbor.display_text)
            if normalized in stop_words or normalized in FUNCTION_WORDS_FOR_COMPLETION:
                break
            if _is_completable_type_variable_candidate(neighbor):
                result.append(neighbor)
                break
    return result


def _cue_scoped_type_variable_candidates(
    candidates: list[RestoredGraphNodeCandidate],
) -> list[RestoredGraphNodeCandidate]:
    ordered = sorted(candidates, key=lambda candidate: candidate.token_index)
    result: list[RestoredGraphNodeCandidate] = []
    cue_words = ILLEGAL_ANCHOR_TEXT
    for index, candidate in enumerate(ordered):
        if not _is_completable_type_variable_candidate(candidate):
            continue
        window = ordered[max(0, index - 3) : index]
        if any(_normalize(item.display_text) in cue_words for item in window):
            result.append(candidate)
    return result


def _path_endpoint_type_variable_candidates(
    selected: list[SelectedAnchor],
    candidates: list[RestoredGraphNodeCandidate],
    weighted_graph: nx.Graph,
) -> list[RestoredGraphNodeCandidate]:
    candidate_by_id = {candidate.node_id: candidate for candidate in candidates}
    anchor_node_ids: list[int] = []
    for anchor in selected:
        try:
            token_id = int(anchor.node_id)
        except (TypeError, ValueError):
            continue
        if token_id in weighted_graph and token_id not in anchor_node_ids:
            anchor_node_ids.append(token_id)
    if len(anchor_node_ids) < 2:
        return []

    result: list[RestoredGraphNodeCandidate] = []
    seen: set[str] = set()
    for left_index, left in enumerate(anchor_node_ids):
        for right in anchor_node_ids[left_index + 1 :]:
            try:
                path = nx.shortest_path(weighted_graph, left, right, weight="weight")
            except nx.NetworkXNoPath:
                continue
            for token_id in path:
                candidate = candidate_by_id.get(str(token_id))
                if candidate is None or candidate.node_id in seen:
                    continue
                if not _is_completable_type_variable_candidate(candidate):
                    continue
                seen.add(candidate.node_id)
                result.append(candidate)
    return result


def _is_illegal_anchor(candidate: RestoredGraphNodeCandidate, text: str) -> bool:
    normalized = _normalize(text)
    if normalized in ILLEGAL_ANCHOR_TEXT:
        return True
    if normalized in PREDICATE_ONLY_ANCHOR_TEXT:
        return True
    if candidate.kind_hint == "cue_candidate":
        return True
    if candidate.kind_hint == "context":
        return True
    pos = (candidate.pos or "").upper()
    if pos.startswith("VB") or pos in {"AUX", "VERB"}:
        return True
    return False


FUNCTION_WORDS_FOR_COMPLETION = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
