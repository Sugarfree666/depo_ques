from __future__ import annotations

import itertools
import re
from typing import Any, Callable

import networkx as nx

from models import AnchorEdge, AnchorGraph, CoreNLPToken, DependencyParse, ExtractedNode, ExtractionResult


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

        folded_graph, anchor_positions = self._fold_dependency_graph(dependency_parse, extraction)
        missing = [anchor for anchor in anchors if anchor not in folded_graph]
        if missing:
            raise AnchorGraphError(
                "The following anchors could not be aligned to CoreNLP token spans: "
                + ", ".join(missing)
            )

        closure = nx.Graph()
        for node in extraction.nodes:
            first_position = min(anchor_positions.get(node.placeholder, [10**9]))
            closure.add_node(
                node.placeholder,
                kind=node.kind,
                text=node.text,
                semantic_type=node.semantic_type,
                order=first_position,
            )

        for left, right in itertools.combinations(anchors, 2):
            path = self._shortest_anchor_path(folded_graph, left, right)
            if path is None:
                continue
            relations = _relations_for_path(folded_graph, path)
            closure.add_edge(
                left,
                right,
                weight=max(len(path) - 1, 1),
                token_path=path,
                path_words=_path_words(folded_graph, path),
                relations=relations,
            )

        if closure.number_of_nodes() > 1 and not nx.is_connected(closure):
            components = [sorted(component) for component in nx.connected_components(closure)]
            raise AnchorGraphError(
                "Could not connect all anchors in the folded dependency graph. Components: "
                + "; ".join(", ".join(component) for component in components)
            )

        mst = nx.minimum_spanning_tree(closure, weight="weight")
        anchor_graph = nx.Graph()
        for node, attrs in closure.nodes(data=True):
            anchor_graph.add_node(node, **attrs)

        anchor_edges: list[AnchorEdge] = []
        for source, target, attrs in mst.edges(data=True):
            edge = AnchorEdge(
                source=source,
                target=target,
                weight=int(attrs.get("weight", 1)),
                token_path=list(attrs.get("token_path", [])),
                path_words=list(attrs.get("path_words", [])),
                relations=list(attrs.get("relations", [])),
            )
            anchor_edges.append(edge)
            anchor_graph.add_edge(
                source,
                target,
                weight=edge.weight,
                token_path=edge.token_path,
                path_words=edge.path_words,
                relations=edge.relations,
            )

        return AnchorGraph(graph=anchor_graph, edges=anchor_edges, anchor_positions=anchor_positions)

    def _fold_dependency_graph(
        self,
        dependency_parse: DependencyParse,
        extraction: ExtractionResult,
    ) -> tuple[nx.Graph, dict[str, list[int]]]:
        tokens_by_index = {token.index: token for token in dependency_parse.tokens}
        anchor_positions = self._align_anchor_spans(dependency_parse.tokens, extraction.nodes)
        token_to_anchor = _resolve_token_anchor_conflicts(anchor_positions, extraction.nodes)
        all_internal_tokens = set(token_to_anchor)

        graph = nx.Graph()
        for token in dependency_parse.tokens:
            if token.index in all_internal_tokens:
                continue
            graph.add_node(
                token.index,
                kind="token",
                word=token.word,
                text=token.word,
                order=token.index,
                character_offset_begin=token.character_offset_begin,
                character_offset_end=token.character_offset_end,
            )

        for node in extraction.nodes:
            internal_tokens = sorted(anchor_positions.get(node.placeholder, []))
            if not internal_tokens:
                continue
            first_token = min(internal_tokens)
            graph.add_node(
                node.placeholder,
                kind=node.kind,
                word=node.placeholder,
                text=node.text,
                semantic_type=node.semantic_type,
                order=first_token,
                folded_token_indices=internal_tokens,
            )

        for edge in dependency_parse.edges:
            source = token_to_anchor.get(edge.source_index, edge.source_index)
            target = token_to_anchor.get(edge.target_index, edge.target_index)
            if source == target:
                continue
            self._ensure_folded_endpoint(graph, source, tokens_by_index, extraction)
            self._ensure_folded_endpoint(graph, target, tokens_by_index, extraction)
            _add_or_merge_edge(
                graph,
                source,
                target,
                relation=edge.relation,
                directed_source=edge.source_index,
                directed_target=edge.target_index,
            )

        return graph, anchor_positions

    def _align_anchor_spans(
        self,
        tokens: list[CoreNLPToken],
        nodes: list[ExtractedNode],
    ) -> dict[str, list[int]]:
        alignments: dict[str, list[int]] = {}
        scores: dict[tuple[str, int], float] = {}

        for node in nodes:
            matched: list[int] = []
            if node.start is not None and node.end is not None and node.start < node.end:
                for token in tokens:
                    score = _span_overlap_score(
                        node.start,
                        node.end,
                        token.character_offset_begin,
                        token.character_offset_end,
                    )
                    if score >= 0.5:
                        matched.append(token.index)
                        scores[(node.placeholder, token.index)] = score

            if not matched:
                matched = _fallback_text_alignment(tokens, node)
                for token_index in matched:
                    scores[(node.placeholder, token_index)] = 1.0

            alignments[node.placeholder] = matched

        resolved = _resolve_alignment_conflicts(alignments, scores, nodes)
        missing = [node.placeholder for node in nodes if not resolved.get(node.placeholder)]
        if missing:
            raise AnchorGraphError(
                "Could not align extracted spans to CoreNLP tokens for anchors: "
                + ", ".join(missing)
            )
        return resolved

    @staticmethod
    def _ensure_folded_endpoint(
        graph: nx.Graph,
        node_id: int | str,
        tokens_by_index: dict[int, CoreNLPToken],
        extraction: ExtractionResult,
    ) -> None:
        if node_id in graph:
            return
        if isinstance(node_id, int) and node_id in tokens_by_index:
            token = tokens_by_index[node_id]
            graph.add_node(
                node_id,
                kind="token",
                word=token.word,
                text=token.word,
                order=token.index,
                character_offset_begin=token.character_offset_begin,
                character_offset_end=token.character_offset_end,
            )
            return
        node = extraction.placeholder_to_node.get(str(node_id))
        if node is not None:
            graph.add_node(
                node.placeholder,
                kind=node.kind,
                word=node.placeholder,
                text=node.text,
                semantic_type=node.semantic_type,
                order=node.start if node.start is not None else 10**9,
            )

    @staticmethod
    def _shortest_anchor_path(
        graph: nx.Graph,
        left: str,
        right: str,
    ) -> list[Any] | None:
        try:
            return nx.shortest_path(graph, left, right, weight="weight")
        except nx.NetworkXNoPath:
            return None


def format_dependency_edges(dependency_parse: DependencyParse) -> list[str]:
    return [f"  - {edge.display()}" for edge in dependency_parse.edges]


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


def _span_overlap_score(span_start: int, span_end: int, token_start: int, token_end: int) -> float:
    if token_start < 0 or token_end <= token_start:
        return 0.0
    overlap = max(0, min(span_end, token_end) - max(span_start, token_start))
    if overlap <= 0:
        return 0.0
    token_len = max(token_end - token_start, 1)
    span_len = max(span_end - span_start, 1)
    return max(overlap / token_len, overlap / span_len)


def _fallback_text_alignment(tokens: list[CoreNLPToken], node: ExtractedNode) -> list[int]:
    target = _normalize_for_alignment(node.text)
    if not target:
        return []

    matches: list[list[int]] = []
    token_count = max(len(re.findall(r"[A-Za-z0-9]+", node.text)), 1)
    max_window = min(len(tokens), token_count + 8)
    sorted_tokens = sorted(tokens, key=lambda token: token.index)

    for start in range(len(sorted_tokens)):
        for end in range(start + 1, min(len(sorted_tokens), start + max_window) + 1):
            candidate = _normalize_for_alignment(" ".join(token.word for token in sorted_tokens[start:end]))
            if candidate == target:
                matches.append([token.index for token in sorted_tokens[start:end]])

    if not matches:
        return []
    occurrence_index = max((node.occurrence or 1) - 1, 0)
    if occurrence_index >= len(matches):
        occurrence_index = 0
    return matches[occurrence_index]


def _normalize_for_alignment(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _resolve_alignment_conflicts(
    alignments: dict[str, list[int]],
    scores: dict[tuple[str, int], float],
    nodes: list[ExtractedNode],
) -> dict[str, list[int]]:
    node_by_placeholder = {node.placeholder: node for node in nodes}
    token_claims: dict[int, list[str]] = {}
    for placeholder, token_indices in alignments.items():
        for token_index in token_indices:
            token_claims.setdefault(token_index, []).append(placeholder)

    winners: dict[int, str] = {}
    for token_index, placeholders in token_claims.items():
        winners[token_index] = min(
            placeholders,
            key=lambda placeholder: (
                -scores.get((placeholder, token_index), 0.0),
                _span_length(node_by_placeholder[placeholder]),
                node_by_placeholder[placeholder].start if node_by_placeholder[placeholder].start is not None else 10**9,
            ),
        )

    resolved: dict[str, list[int]] = {placeholder: [] for placeholder in alignments}
    for token_index, placeholder in winners.items():
        resolved[placeholder].append(token_index)
    for placeholder in resolved:
        resolved[placeholder].sort()
    return resolved


def _resolve_token_anchor_conflicts(
    anchor_positions: dict[str, list[int]],
    nodes: list[ExtractedNode],
) -> dict[int, str]:
    node_order = {node.placeholder: index for index, node in enumerate(nodes)}
    token_to_anchor: dict[int, str] = {}
    for placeholder in sorted(anchor_positions, key=lambda item: node_order.get(item, 10**9)):
        for token_index in anchor_positions[placeholder]:
            token_to_anchor[token_index] = placeholder
    return token_to_anchor


def _span_length(node: ExtractedNode) -> int:
    if node.start is None or node.end is None:
        return len(node.text)
    return max(node.end - node.start, 1)


def _add_or_merge_edge(
    graph: nx.Graph,
    source: int | str,
    target: int | str,
    relation: str,
    directed_source: int,
    directed_target: int,
) -> None:
    if graph.has_edge(source, target):
        relations = list(graph.edges[source, target].get("relations", []))
        directed_edges = list(graph.edges[source, target].get("directed_edges", []))
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
        weight=1,
        relation="|".join(relations),
        relations=relations,
        directed_edges=directed_edges,
    )


def _path_words(graph: nx.Graph, path: list[Any]) -> list[str]:
    return [str(graph.nodes[node].get("word", node)) for node in path]


def _relations_for_path(graph: nx.Graph, path: list[Any]) -> list[str]:
    relations: list[str] = []
    for left, right in zip(path, path[1:]):
        edge_relations = graph.edges[left, right].get("relations")
        if edge_relations:
            relations.append("|".join(str(item) for item in edge_relations if item))
        else:
            relations.append(str(graph.edges[left, right].get("relation", "")))
    return relations


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

