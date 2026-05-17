from __future__ import annotations

import itertools
import math
import re
from typing import Any, Callable

import networkx as nx

from models import (
    AnchorConnectedSubgraph,
    AnchorEdge,
    AnchorGraph,
    AnchorSelectionResult,
    CoreNLPToken,
    DependencyParse,
    ExtractedNode,
    ExtractionResult,
    GraphNodeCandidate,
    MaskMapping,
    PlaceholderReplacement,
    RestoredAnchorConnectedSubgraph,
    RestoredGraphNodeCandidate,
    SelectedAnchor,
)

LOW_WEIGHT_RELATIONS = {"nsubj", "nsubj:pass", "obj", "iobj", "ccomp", "xcomp"}
MEDIUM_WEIGHT_BASES = {
    "nmod",
    "obl",
    "amod",
    "advmod",
    "nummod",
    "compound",
    "acl",
    "advcl",
    "case",
    "mark",
    "aux",
    "cop",
}
MEDIUM_WEIGHT_RELATIONS = {"nmod:of", "acl:relcl", "aux:pass"}
HIGH_WEIGHT_BASES = {"cc", "det", "punct", "dep"}
COORDINATION_WEIGHT_BASES = {"conj"}
DEFAULT_RELATION_WEIGHT = 3
INFINITE_RELATION_WEIGHT = math.inf

ANCHOR_CUE_WORDS = {
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
    "older",
    "or",
    "same",
    "smaller",
    "younger",
}

FUNCTION_WORDS = {
    "a",
    "an",
    "are",
    "as",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "share",
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

COMMON_TYPE_VARIABLE_WORDS = {
    "actor",
    "age",
    "ceo",
    "city",
    "company",
    "country",
    "director",
    "farm",
    "film",
    "network",
    "nationality",
    "population",
    "region",
    "university",
}


class AnchorGraphError(RuntimeError):
    pass


class GraphBuilder:
    def build_anchor_graph(
        self,
        dependency_parse: DependencyParse,
        extraction: ExtractionResult,
    ) -> AnchorGraph:
        anchors = [node.placeholder for node in extraction.nodes]
        if not anchors:
            raise AnchorGraphError("No entity/type-variable anchors were extracted.")

        weighted_graph = self.build_weighted_dependency_graph(dependency_parse)
        anchor_positions = self._align_anchor_spans(dependency_parse.tokens, extraction.nodes)
        missing = [node for node in extraction.nodes if not anchor_positions.get(node.placeholder)]
        if missing:
            raise AnchorGraphError(_format_missing_alignment_error(missing, anchor_positions))

        closure = self._build_anchor_closure(weighted_graph, extraction.nodes, anchor_positions)
        if closure.number_of_nodes() > 1 and not nx.is_connected(closure):
            components = [sorted(component) for component in nx.connected_components(closure)]
            raise AnchorGraphError(
                "Could not connect all anchors in the weighted dependency graph. Components: "
                + "; ".join(", ".join(component) for component in components)
            )

        anchor_subgraph = self._build_anchor_subgraph(weighted_graph, closure)
        semantic_graph, anchor_edges = self._build_semantic_mst(closure)

        return AnchorGraph(
            graph=semantic_graph,
            edges=anchor_edges,
            anchor_positions=anchor_positions,
            folded_graph=anchor_subgraph,
            weighted_graph=weighted_graph,
            anchor_subgraph=anchor_subgraph,
        )

    def build_weighted_dependency_graph(self, dependency_parse: DependencyParse) -> nx.Graph:
        graph = nx.Graph()
        tokens_by_index = {token.index: token for token in dependency_parse.tokens}
        for token in dependency_parse.tokens:
            graph.add_node(
                token.index,
                kind="token",
                word=token.word,
                text=token.word,
                label=_token_label(token),
                order=token.index,
                pos=token.pos,
                lemma=token.lemma,
                character_offset_begin=token.character_offset_begin,
                character_offset_end=token.character_offset_end,
            )

        for edge in dependency_parse.edges:
            if edge.source_index not in tokens_by_index or edge.target_index not in tokens_by_index:
                continue
            weight = relation_weight(edge.relation)
            _add_or_merge_weighted_edge(
                graph,
                source=edge.source_index,
                target=edge.target_index,
                relation=edge.relation,
                weight=weight,
                directed_source=edge.source_index,
                directed_target=edge.target_index,
            )
        return graph

    def build_graph_node_candidates(
        self,
        dependency_parse: DependencyParse,
        replacement: PlaceholderReplacement,
    ) -> list[GraphNodeCandidate]:
        mapping_by_placeholder = _replacement_mapping_by_placeholder(replacement)
        candidates: list[GraphNodeCandidate] = []
        for token in dependency_parse.tokens:
            mask_info = mapping_by_placeholder.get(token.word)
            if mask_info is not None:
                restored_text = mask_info.original_text
                kind_hint = (
                    "type_variable_candidate"
                    if mask_info.kind_hint == "type_variable"
                    else "entity_candidate"
                )
                char_span = mask_info.original_char_span or None
                semantic_type = mask_info.semantic_type_hint
                placeholder = token.word
            else:
                restored_text = token.word
                kind_hint = _candidate_kind_hint(token)
                char_span = (
                    [token.character_offset_begin, token.character_offset_end]
                    if token.character_offset_begin >= 0 and token.character_offset_end >= 0
                    else None
                )
                semantic_type = _semantic_type_hint_for_token(token)
                placeholder = None
            candidates.append(
                GraphNodeCandidate(
                    node_id=str(token.index),
                    token_index=token.index,
                    graph_text=token.word,
                    placeholder=placeholder,
                    restored_text=restored_text,
                    display_text=restored_text,
                    is_mask_placeholder=mask_info is not None,
                    pos=token.pos,
                    lemma=token.lemma,
                    kind_hint=kind_hint,
                    semantic_type_hint=semantic_type,
                    char_span=char_span,
                    source_token_indices=[token.index],
                )
            )
        return candidates

    def restore_graph_node_candidates(
        self,
        graph_node_candidates: list[GraphNodeCandidate],
        replacement: PlaceholderReplacement,
    ) -> list[RestoredGraphNodeCandidate]:
        del replacement
        restored: list[RestoredGraphNodeCandidate] = []
        for candidate in graph_node_candidates:
            restored.append(
                RestoredGraphNodeCandidate(
                    node_id=candidate.node_id,
                    token_index=candidate.token_index,
                    graph_text=candidate.graph_text,
                    placeholder=candidate.placeholder,
                    restored_text=candidate.restored_text,
                    display_text=candidate.display_text,
                    is_mask_placeholder=candidate.is_mask_placeholder,
                    pos=candidate.pos,
                    lemma=candidate.lemma,
                    kind_hint=candidate.kind_hint,
                    semantic_type_hint=candidate.semantic_type_hint,
                    char_span=candidate.char_span,
                    source_token_indices=list(candidate.source_token_indices),
                    text=candidate.display_text,
                )
            )
        return restored

    def build_anchor_connected_subgraph(
        self,
        weighted_graph: nx.Graph,
        selected_anchors: list[SelectedAnchor] | AnchorSelectionResult,
        graph_node_candidates: list[GraphNodeCandidate],
    ) -> AnchorConnectedSubgraph:
        del graph_node_candidates
        anchor_token_ids = _selected_anchor_token_ids(selected_anchors, weighted_graph)
        subgraph = nx.Graph()
        shortest_paths: list[dict[str, Any]] = []
        if not anchor_token_ids:
            return AnchorConnectedSubgraph(graph=subgraph)

        for node_id in anchor_token_ids:
            if node_id in weighted_graph:
                subgraph.add_node(node_id, **weighted_graph.nodes[node_id])

        if len(anchor_token_ids) == 1:
            return AnchorConnectedSubgraph(
                selected_anchor_node_ids=[str(item) for item in anchor_token_ids],
                nodes=_graph_nodes_payload(subgraph),
                edges=[],
                shortest_paths=[],
                graph=subgraph,
            )

        closure = nx.Graph()
        for token_id in anchor_token_ids:
            closure.add_node(str(token_id), token_index=token_id)
        for left, right in itertools.combinations(anchor_token_ids, 2):
            try:
                path = nx.shortest_path(weighted_graph, left, right, weight=_dijkstra_weight)
            except nx.NetworkXNoPath:
                continue
            weight = _path_weight(weighted_graph, path)
            path_payload = {
                "source": str(left),
                "target": str(right),
                "node_ids": [str(item) for item in path],
                "token_path": list(path),
                "path_words": _path_words(weighted_graph, path),
                "relations": _relations_for_path(weighted_graph, path),
                "weight": weight,
            }
            shortest_paths.append(path_payload)
            closure.add_edge(str(left), str(right), weight=weight, token_path=list(path))

        if closure.number_of_edges() > 0:
            if nx.is_connected(closure):
                closure_edges = nx.minimum_spanning_tree(closure, weight="weight").edges(data=True)
            else:
                closure_edges = closure.edges(data=True)
            for _, _, attrs in closure_edges:
                path = list(attrs.get("token_path", []))
                _add_path_to_subgraph(weighted_graph, subgraph, path)

        for token_id in anchor_token_ids:
            if token_id in weighted_graph and token_id not in subgraph:
                subgraph.add_node(token_id, **weighted_graph.nodes[token_id])

        return AnchorConnectedSubgraph(
            selected_anchor_node_ids=[str(item) for item in anchor_token_ids],
            nodes=_graph_nodes_payload(subgraph),
            edges=_graph_edges_payload(subgraph),
            shortest_paths=shortest_paths,
            graph=subgraph,
        )

    def restore_anchor_connected_subgraph(
        self,
        anchor_connected_subgraph: AnchorConnectedSubgraph,
        replacement: PlaceholderReplacement,
    ) -> RestoredAnchorConnectedSubgraph:
        graph = anchor_connected_subgraph.graph
        if graph is None:
            return RestoredAnchorConnectedSubgraph(
                selected_anchor_node_ids=anchor_connected_subgraph.selected_anchor_node_ids,
            )
        mapping_by_placeholder = _replacement_mapping_by_placeholder(replacement)
        restored_nodes: list[dict[str, Any]] = []
        for node, attrs in sorted(graph.nodes(data=True), key=lambda item: _node_order(graph, item[0])):
            graph_text = str(attrs.get("word", attrs.get("text", node)))
            restored_text = _restore_graph_text(graph_text, mapping_by_placeholder)
            restored_nodes.append(
                {
                    "node_id": str(node),
                    "token_index": int(node) if isinstance(node, int) else node,
                    "graph_text": graph_text,
                    "text": restored_text,
                    "display_text": restored_text,
                    "is_anchor": str(node) in anchor_connected_subgraph.selected_anchor_node_ids,
                    "pos": attrs.get("pos"),
                }
            )

        restored_edges: list[dict[str, Any]] = []
        for source, target, attrs in sorted(
            graph.edges(data=True),
            key=lambda item: (_node_order(graph, item[0]), _node_order(graph, item[1])),
        ):
            source_text = _restore_graph_text(str(graph.nodes[source].get("word", source)), mapping_by_placeholder)
            target_text = _restore_graph_text(str(graph.nodes[target].get("word", target)), mapping_by_placeholder)
            restored_edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "source_text": source_text,
                    "target_text": target_text,
                    "relation": attrs.get("relation", ""),
                    "relations": list(attrs.get("relations", [])),
                    "weight": attrs.get("weight", DEFAULT_RELATION_WEIGHT),
                }
            )

        restored_paths: list[dict[str, Any]] = []
        for path in anchor_connected_subgraph.shortest_paths:
            token_path = [int(item) for item in path.get("token_path", []) if str(item).isdigit()]
            restored_words = [
                _restore_graph_text(str(graph.nodes[token].get("word", token)), mapping_by_placeholder)
                for token in token_path
                if token in graph
            ]
            restored_paths.append(
                {
                    **path,
                    "path_words": restored_words,
                    "display": " -- ".join(
                        f"n{token}: {word}" for token, word in zip(token_path, restored_words)
                    ),
                }
            )

        display_lines = [str(path.get("display", "")) for path in restored_paths if path.get("display")]
        if not display_lines and restored_edges:
            display_lines = [
                f"n{edge['source']}: {edge['source_text']} -- n{edge['target']}: {edge['target_text']}"
                for edge in restored_edges
            ]

        return RestoredAnchorConnectedSubgraph(
            selected_anchor_node_ids=anchor_connected_subgraph.selected_anchor_node_ids,
            nodes=restored_nodes,
            edges=restored_edges,
            shortest_paths=restored_paths,
            display_lines=display_lines,
        )

    def _build_anchor_closure(
        self,
        weighted_graph: nx.Graph,
        nodes: list[ExtractedNode],
        anchor_positions: dict[str, list[int]],
    ) -> nx.Graph:
        closure = nx.Graph()
        for node in nodes:
            positions = anchor_positions.get(node.placeholder, [])
            closure.add_node(
                node.placeholder,
                kind=node.kind,
                text=node.text,
                semantic_type=node.semantic_type,
                order=min(positions or [10**9]),
                token_indices=positions,
            )

        for left, right in itertools.combinations(nodes, 2):
            path = self._shortest_anchor_token_path(
                weighted_graph,
                anchor_positions[left.placeholder],
                anchor_positions[right.placeholder],
            )
            if path is None:
                continue
            weight = _path_weight(weighted_graph, path)
            closure.add_edge(
                left.placeholder,
                right.placeholder,
                weight=weight,
                token_path=path,
                path_words=_path_words(weighted_graph, path),
                relations=_relations_for_path(weighted_graph, path),
            )
        return closure

    @staticmethod
    def _build_anchor_subgraph(weighted_graph: nx.Graph, closure: nx.Graph) -> nx.Graph:
        subgraph = nx.Graph()
        for _, _, attrs in closure.edges(data=True):
            path = list(attrs.get("token_path", []))
            for token_index in path:
                if token_index in weighted_graph:
                    subgraph.add_node(token_index, **weighted_graph.nodes[token_index])
            for source, target in zip(path, path[1:]):
                if weighted_graph.has_edge(source, target):
                    subgraph.add_edge(source, target, **weighted_graph.edges[source, target])
        return subgraph

    @staticmethod
    def _build_semantic_mst(closure: nx.Graph) -> tuple[nx.Graph, list[AnchorEdge]]:
        ranked_closure = closure.copy()
        for source, target, attrs in ranked_closure.edges(data=True):
            attrs["mst_weight"] = _edge_weight(attrs) * 1000 + _mst_tie_penalty(
                ranked_closure,
                source,
                target,
            )
        mst = nx.minimum_spanning_tree(ranked_closure, weight="mst_weight")
        semantic_graph = nx.Graph()
        for node, attrs in closure.nodes(data=True):
            semantic_graph.add_node(node, **attrs)

        anchor_edges: list[AnchorEdge] = []
        for source, target, attrs in mst.edges(data=True):
            edge = AnchorEdge(
                source=source,
                target=target,
                weight=_edge_weight(attrs, default=1),
                token_path=list(attrs.get("token_path", [])),
                path_words=list(attrs.get("path_words", [])),
                relations=list(attrs.get("relations", [])),
            )
            anchor_edges.append(edge)
            semantic_graph.add_edge(
                source,
                target,
                weight=edge.weight,
                token_path=edge.token_path,
                path_words=edge.path_words,
                relations=edge.relations,
            )
        return semantic_graph, anchor_edges

    @staticmethod
    def _shortest_anchor_token_path(
        graph: nx.Graph,
        left_positions: list[int],
        right_positions: list[int],
    ) -> list[int] | None:
        best_path: list[int] | None = None
        for left in left_positions:
            for right in right_positions:
                try:
                    path = nx.shortest_path(graph, left, right, weight=_dijkstra_weight)
                except nx.NetworkXNoPath:
                    continue
                path_key = _path_sort_key(graph, path)
                if best_path is None or path_key < _path_sort_key(graph, best_path):
                    best_path = path
        return best_path

    def _align_anchor_spans(
        self,
        tokens: list[CoreNLPToken],
        nodes: list[ExtractedNode],
    ) -> dict[str, list[int]]:
        raw_alignments: dict[str, list[int]] = {}
        claims: dict[int, list[_TokenClaim]] = {}
        fallback_occurrences = _fallback_occurrences(nodes)

        for node in nodes:
            token_indices: list[int] = []
            used_span = _has_explicit_span(node)
            if used_span:
                for token in tokens:
                    score = _span_overlap_score(
                        node.start or 0,
                        node.end or 0,
                        token.character_offset_begin,
                        token.character_offset_end,
                    )
                    if score >= 0.5:
                        token_indices.append(token.index)
                        claims.setdefault(token.index, []).append(
                            _TokenClaim(
                                placeholder=node.placeholder,
                                score=score,
                                explicit_span=True,
                                span_length=_span_length(node),
                                node_order=nodes.index(node),
                                semantic_score=_semantic_specificity(node),
                            )
                        )

            if not token_indices:
                occurrence = node.occurrence or fallback_occurrences[node.placeholder]
                token_indices = _fallback_text_alignment(tokens, node.text, occurrence)
                for token_index in token_indices:
                    claims.setdefault(token_index, []).append(
                        _TokenClaim(
                            placeholder=node.placeholder,
                            score=1.0,
                            explicit_span=False,
                            span_length=len(node.text),
                            node_order=nodes.index(node),
                            semantic_score=_semantic_specificity(node),
                        )
                    )

            raw_alignments[node.placeholder] = token_indices

        resolved: dict[str, list[int]] = {node.placeholder: [] for node in nodes}
        for token_index, token_claims in claims.items():
            winner = max(token_claims, key=lambda claim: claim.priority)
            resolved[winner.placeholder].append(token_index)

        for placeholder in resolved:
            resolved[placeholder].sort()

        missing = [node for node in nodes if not resolved.get(node.placeholder)]
        if missing:
            raise AnchorGraphError(_format_missing_alignment_error(missing, raw_alignments))
        return resolved

class _TokenClaim:
    def __init__(
        self,
        placeholder: str,
        score: float,
        explicit_span: bool,
        span_length: int,
        node_order: int,
        semantic_score: int,
    ) -> None:
        self.placeholder = placeholder
        self.score = score
        self.explicit_span = explicit_span
        self.span_length = span_length
        self.node_order = node_order
        self.semantic_score = semantic_score

    @property
    def priority(self) -> tuple[int, float, int, int, int]:
        return (
            1 if self.explicit_span else 0,
            self.score,
            self.span_length,
            self.semantic_score,
            -self.node_order,
        )


def restored_dependency_tokens(
    dependency_parse: DependencyParse,
    restored_candidates: list[RestoredGraphNodeCandidate],
) -> list[dict[str, Any]]:
    candidate_by_id = {candidate.node_id: candidate for candidate in restored_candidates}
    tokens: list[dict[str, Any]] = []
    for token in dependency_parse.tokens:
        candidate = candidate_by_id.get(str(token.index))
        text = candidate.display_text if candidate is not None else token.word
        tokens.append(
            {
                "node_id": str(token.index),
                "text": text,
                "pos": token.pos,
                "lemma": token.lemma,
            }
        )
    return tokens


def restored_dependency_edges(
    dependency_parse: DependencyParse,
    restored_candidates: list[RestoredGraphNodeCandidate],
) -> list[dict[str, Any]]:
    candidate_by_id = {candidate.node_id: candidate for candidate in restored_candidates}
    edges: list[dict[str, Any]] = []
    for edge in dependency_parse.edges:
        source = candidate_by_id.get(str(edge.source_index))
        target = candidate_by_id.get(str(edge.target_index))
        source_text = source.display_text if source is not None else edge.source
        target_text = target.display_text if target is not None else edge.target
        edges.append(
            {
                "source_node_id": str(edge.source_index),
                "source_text": source_text,
                "relation": edge.relation,
                "target_node_id": str(edge.target_index),
                "target_text": target_text,
            }
        )
    return edges


def format_dependency_edges(dependency_parse: DependencyParse) -> list[str]:
    return [f"  - {edge.display()}" for edge in dependency_parse.edges]


def format_weighted_graph_edges(graph: nx.Graph) -> list[str]:
    lines: list[str] = []
    for source, target, attrs in sorted(
        graph.edges(data=True),
        key=lambda item: (
            _node_order(graph, item[0]),
            _node_order(graph, item[1]),
            str(item[0]),
            str(item[1]),
        ),
    ):
        relation = "|".join(attrs.get("relations", [])) or attrs.get("relation", "")
        relation_text = f" ({relation})" if relation else ""
        lines.append(
            f"  - {_graph_node_label(graph, source)} --{attrs.get('weight', DEFAULT_RELATION_WEIGHT)}-- "
            f"{_graph_node_label(graph, target)}{relation_text}"
        )
    return lines


def relation_weight(relation: str) -> float:
    normalized = relation.strip()
    base = normalized.split(":", 1)[0]
    if normalized in LOW_WEIGHT_RELATIONS:
        return 1
    if normalized in MEDIUM_WEIGHT_RELATIONS or base in MEDIUM_WEIGHT_BASES:
        return 3
    if base in COORDINATION_WEIGHT_BASES:
        return INFINITE_RELATION_WEIGHT
    if base in HIGH_WEIGHT_BASES:
        return 5
    return DEFAULT_RELATION_WEIGHT


def format_graph_lines(
    graph: nx.Graph,
    label_func: Callable[[str], str] | None = None,
    entity_nodes: list[str] | None = None,
    operator_nodes: list[str] | None = None,
) -> list[str]:
    if label_func is None:
        label_func = lambda node: node
    entity_nodes = entity_nodes or []
    operator_nodes = operator_nodes or []

    if graph.number_of_nodes() == 0:
        return ["  (empty graph)"]
    if graph.number_of_edges() == 0:
        return [f"  {label_func(node)}" for node in sorted(graph.nodes)]

    lines: list[str] = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component).copy()
        component_ops = [node for node in operator_nodes if node in subgraph]
        if component_ops:
            target = component_ops[0]
            starts = [
                node
                for node in sorted(subgraph.nodes, key=lambda n: _node_order(subgraph, n))
                if node != target and subgraph.degree(node) == 1 and node not in operator_nodes
            ]
            if not starts:
                starts = [node for node in subgraph.nodes if node != target]
            for start in starts:
                lines.append("  " + " ---- ".join(label_func(node) for node in nx.shortest_path(subgraph, start, target)))
            continue

        component_entities = [node for node in entity_nodes if node in subgraph]
        if len(component_entities) >= 2:
            target = _choose_multi_entity_target(subgraph, component_entities)
            for start in sorted(component_entities, key=lambda n: _node_order(subgraph, n)):
                try:
                    path = nx.shortest_path(subgraph, start, target)
                except nx.NetworkXNoPath:
                    continue
                lines.append("  " + " ---- ".join(label_func(node) for node in path))
            continue

        if _is_simple_path(subgraph):
            path = _linear_path(subgraph, component_entities[0] if component_entities else None)
            lines.append("  " + " ---- ".join(label_func(node) for node in path))
            continue

        root = component_entities[0] if component_entities else _highest_degree_node(subgraph)
        leaves = [
            node
            for node in sorted(subgraph.nodes, key=lambda n: _node_order(subgraph, n))
            if node != root and subgraph.degree(node) == 1
        ]
        if not leaves:
            leaves = [node for node in subgraph.nodes if node != root]
        for leaf in leaves:
            lines.append("  " + " ---- ".join(label_func(node) for node in nx.shortest_path(subgraph, root, leaf)))

    return lines or ["  (empty graph)"]


def _has_explicit_span(node: ExtractedNode) -> bool:
    return node.start is not None and node.end is not None and node.start < node.end


def _span_overlap_score(span_start: int, span_end: int, token_start: int, token_end: int) -> float:
    if token_start < 0 or token_end <= token_start:
        return 0.0
    overlap = max(0, min(span_end, token_end) - max(span_start, token_start))
    if overlap <= 0:
        return 0.0
    token_len = max(token_end - token_start, 1)
    span_len = max(span_end - span_start, 1)
    return max(overlap / token_len, overlap / span_len)


def _fallback_text_alignment(tokens: list[CoreNLPToken], text: str, occurrence: int) -> list[int]:
    target = _normalize_for_alignment(text)
    if not target:
        return []

    matches: list[list[int]] = []
    sorted_tokens = sorted(tokens, key=lambda token: token.index)
    token_count = max(len(re.findall(r"[A-Za-z0-9]+", text)), 1)
    max_window = min(len(sorted_tokens), token_count + 8)

    for start in range(len(sorted_tokens)):
        for end in range(start + 1, min(len(sorted_tokens), start + max_window) + 1):
            candidate = _normalize_for_alignment(" ".join(token.word for token in sorted_tokens[start:end]))
            if candidate == target:
                matches.append([token.index for token in sorted_tokens[start:end]])

    if not matches:
        return []
    index = max(occurrence - 1, 0)
    if index >= len(matches):
        index = 0
    return matches[index]


def _fallback_occurrences(nodes: list[ExtractedNode]) -> dict[str, int]:
    seen: dict[str, int] = {}
    result: dict[str, int] = {}
    for node in nodes:
        key = _normalize_for_alignment(node.text)
        seen[key] = seen.get(key, 0) + 1
        result[node.placeholder] = seen[key]
    return result


def _normalize_for_alignment(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _span_length(node: ExtractedNode) -> int:
    if _has_explicit_span(node):
        return max((node.end or 0) - (node.start or 0), 1)
    return len(node.text)


def _semantic_specificity(node: ExtractedNode) -> int:
    score = 0
    if node.kind == "entity":
        score += 2
    if node.semantic_type.lower() not in {"entity", "variable", "thing"}:
        score += 1
    return score


def _add_or_merge_weighted_edge(
    graph: nx.Graph,
    source: int,
    target: int,
    relation: str,
    weight: float,
    directed_source: int,
    directed_target: int,
) -> None:
    if graph.has_edge(source, target):
        relations = list(graph.edges[source, target].get("relations", []))
        directed_edges = list(graph.edges[source, target].get("directed_edges", []))
        weight = min(_edge_weight(graph.edges[source, target], default=weight), weight)
    else:
        relations = []
        directed_edges = []

    if relation and relation not in relations:
        relations.append(relation)
    directed_edges.append(
        {
            "source_index": directed_source,
            "target_index": directed_target,
            "relation": relation,
        }
    )
    graph.add_edge(
        source,
        target,
        weight=weight,
        relation="|".join(relations),
        relations=relations,
        directed_edges=directed_edges,
    )


def _path_words(graph: nx.Graph, path: list[Any]) -> list[str]:
    return [str(graph.nodes[node].get("word", node)) for node in path]


def _path_weight(graph: nx.Graph, path: list[Any]) -> float:
    total = 0.0
    for left, right in zip(path, path[1:]):
        total += _edge_weight(graph.edges[left, right])
    return total


def _path_sort_key(graph: nx.Graph, path: list[Any]) -> tuple[float, int, int]:
    coordination_penalty = 0
    for left, right in zip(path, path[1:]):
        coordination_penalty += _coordination_penalty(graph.edges[left, right])
    return (_path_weight(graph, path), coordination_penalty, len(path))


def _dijkstra_weight(source: Any, target: Any, attrs: dict[str, Any]) -> float:
    del source, target
    return _edge_weight(attrs) * 1000 + _coordination_penalty(attrs)


def _edge_weight(attrs: dict[str, Any], default: float = DEFAULT_RELATION_WEIGHT) -> float:
    value = attrs.get("weight", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coordination_penalty(attrs: dict[str, Any]) -> int:
    relations = attrs.get("relations") or []
    for relation in relations:
        if str(relation).split(":", 1)[0] in COORDINATION_WEIGHT_BASES:
            return 1
    return 0


def _mst_tie_penalty(graph: nx.Graph, source: str, target: str) -> int:
    kinds = {graph.nodes[source].get("kind"), graph.nodes[target].get("kind")}
    penalty = 0 if "type_variable" in kinds else 1
    penalty += _coordination_penalty(graph.edges[source, target])
    return penalty


def _relations_for_path(graph: nx.Graph, path: list[Any]) -> list[str]:
    relations: list[str] = []
    for left, right in zip(path, path[1:]):
        edge_relations = graph.edges[left, right].get("relations")
        if edge_relations:
            relations.append("|".join(str(item) for item in edge_relations if item))
        else:
            relations.append(str(graph.edges[left, right].get("relation", "")))
    return relations


def _format_missing_alignment_error(
    nodes: list[ExtractedNode],
    raw_alignments: dict[str, list[int]] | None = None,
) -> str:
    details = []
    for node in nodes:
        raw = ""
        if raw_alignments is not None:
            raw = f", raw_matches={raw_alignments.get(node.placeholder, [])}"
        details.append(
            f"{node.placeholder} text={node.text!r} span=({node.start}, {node.end}) occurrence={node.occurrence}{raw}"
        )
    return "Could not align extracted spans to CoreNLP tokens for anchors: " + "; ".join(details)


def _token_label(token: CoreNLPToken) -> str:
    return f"{token.word}[{token.index}]"


def _graph_node_label(graph: nx.Graph, node: Any) -> str:
    attrs = graph.nodes[node]
    label = attrs.get("label")
    if label:
        return str(label)
    word = attrs.get("word")
    if word:
        return f"{word}[{node}]" if isinstance(node, int) else str(word)
    return str(node)


def _node_order(graph: nx.Graph, node: str) -> int:
    return int(graph.nodes[node].get("order", 10**9))


def _is_simple_path(graph: nx.Graph) -> bool:
    return graph.number_of_edges() == max(graph.number_of_nodes() - 1, 0) and all(
        degree <= 2 for _, degree in graph.degree()
    )


def _linear_path(graph: nx.Graph, preferred_start: str | None = None) -> list[str]:
    if preferred_start in graph:
        start = preferred_start
    else:
        endpoints = [node for node, degree in graph.degree() if degree <= 1]
        start = min(endpoints or list(graph.nodes), key=lambda n: _node_order(graph, n))

    path: list[str] = []
    previous: str | None = None
    current = start
    while current is not None:
        path.append(current)
        next_nodes = [
            node
            for node in sorted(graph.neighbors(current), key=lambda n: _node_order(graph, n))
            if node != previous
        ]
        if not next_nodes:
            break
        previous, current = current, next_nodes[0]
    return path


def _highest_degree_node(graph: nx.Graph) -> str:
    return max(graph.nodes, key=lambda node: (graph.degree(node), -_node_order(graph, node)))


def _choose_multi_entity_target(graph: nx.Graph, entity_nodes: list[str]) -> str:
    candidates = [node for node in graph.nodes if node not in entity_nodes]
    if not candidates:
        return entity_nodes[0]

    def score(node: str) -> tuple[int, int, int]:
        total = 0
        max_distance = 0
        for entity in entity_nodes:
            try:
                distance = nx.shortest_path_length(graph, entity, node)
            except nx.NetworkXNoPath:
                distance = 10**6
            total += distance
            max_distance = max(max_distance, distance)
        return (max_distance, total, -_node_order(graph, node))

    return min(candidates, key=score)


def _replacement_mapping_by_placeholder(replacement: PlaceholderReplacement) -> dict[str, MaskMapping]:
    result: dict[str, MaskMapping] = {}
    for mapping in getattr(replacement, "mask_mappings", []) or []:
        if isinstance(mapping, MaskMapping):
            result[mapping.placeholder] = mapping
    for placeholder, info in getattr(replacement, "mask_mapping", {}).items():
        if placeholder in result:
            continue
        span = info.get("span") or {}
        masked_span = info.get("masked_span") or {}
        result[placeholder] = MaskMapping(
            placeholder=placeholder,
            original_text=str(info.get("text", placeholder)),
            kind_hint=str(info.get("kind", "entity")),
            semantic_type_hint=str(info.get("semantic_type", info.get("type", ""))) or None,
            original_char_span=_span_payload_to_list(span),
            masked_char_span=_span_payload_to_list(masked_span),
        )
    return result


def _span_payload_to_list(value: Any) -> list[int]:
    if not isinstance(value, dict):
        return []
    start = value.get("start")
    end = value.get("end")
    if start is None or end is None:
        return []
    return [int(start), int(end)]


def _candidate_kind_hint(token: CoreNLPToken) -> str:
    word = token.word.strip()
    lowered = word.lower()
    if lowered in ANCHOR_CUE_WORDS:
        return "cue_candidate"
    if lowered in FUNCTION_WORDS or re.fullmatch(r"\W+", word):
        return "context"
    pos = (token.pos or "").upper()
    if pos in {"NNP", "NNPS", "PROPN"}:
        return "entity_candidate"
    if pos.startswith("NN") or pos == "NOUN" or lowered in COMMON_TYPE_VARIABLE_WORDS or word.isupper():
        return "type_variable_candidate"
    if token.pos is None and re.fullmatch(r"[A-Za-z][A-Za-z0-9'-]*", word):
        return "type_variable_candidate"
    return "context"


def _semantic_type_hint_for_token(token: CoreNLPToken) -> str | None:
    lowered = token.word.lower()
    mapping = {
        "actor": "Person",
        "ceo": "Person",
        "city": "City",
        "company": "Company",
        "country": "Country",
        "director": "Person",
        "film": "Film",
        "network": "Network",
        "nationality": "Nationality",
        "population": "Population",
        "university": "University",
    }
    return mapping.get(lowered)


def _restore_graph_text(graph_text: str, mapping_by_placeholder: dict[str, MaskMapping]) -> str:
    mapping = mapping_by_placeholder.get(graph_text)
    if mapping is None:
        return graph_text
    return mapping.original_text


def _selected_anchor_token_ids(
    selected_anchors: list[SelectedAnchor] | AnchorSelectionResult,
    weighted_graph: nx.Graph,
) -> list[int]:
    result: list[int] = []
    anchors = (
        selected_anchors.selected_anchors
        if isinstance(selected_anchors, AnchorSelectionResult)
        else selected_anchors
    )
    for anchor in anchors:
        try:
            token_id = int(anchor.node_id)
        except (TypeError, ValueError):
            continue
        if token_id in weighted_graph and token_id not in result:
            result.append(token_id)
    return result


def _add_path_to_subgraph(weighted_graph: nx.Graph, subgraph: nx.Graph, path: list[Any]) -> None:
    for token_index in path:
        if token_index in weighted_graph:
            subgraph.add_node(token_index, **weighted_graph.nodes[token_index])
    for source, target in zip(path, path[1:]):
        if weighted_graph.has_edge(source, target):
            subgraph.add_edge(source, target, **weighted_graph.edges[source, target])


def _graph_nodes_payload(graph: nx.Graph) -> list[dict[str, Any]]:
    return [
        {
            "node_id": str(node),
            "token_index": int(node) if isinstance(node, int) else node,
            "graph_text": attrs.get("word", attrs.get("text", node)),
            "pos": attrs.get("pos"),
            "lemma": attrs.get("lemma"),
        }
        for node, attrs in sorted(graph.nodes(data=True), key=lambda item: _node_order(graph, item[0]))
    ]


def _graph_edges_payload(graph: nx.Graph) -> list[dict[str, Any]]:
    return [
        {
            "source": str(source),
            "target": str(target),
            "relation": attrs.get("relation", ""),
            "relations": list(attrs.get("relations", [])),
            "weight": attrs.get("weight", DEFAULT_RELATION_WEIGHT),
        }
        for source, target, attrs in sorted(
            graph.edges(data=True),
            key=lambda item: (_node_order(graph, item[0]), _node_order(graph, item[1])),
        )
    ]
