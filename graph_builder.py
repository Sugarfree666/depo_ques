from __future__ import annotations

import itertools
import re
from typing import Any, Callable

import networkx as nx

from models import AnchorEdge, AnchorGraph, CoreNLPToken, DependencyParse, ExtractedNode, ExtractionResult

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
HIGH_WEIGHT_BASES = {"conj", "cc", "det", "punct", "dep"}
DEFAULT_RELATION_WEIGHT = 3


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
        mst = nx.minimum_spanning_tree(closure, weight="weight")
        semantic_graph = nx.Graph()
        for node, attrs in closure.nodes(data=True):
            semantic_graph.add_node(node, **attrs)

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
        best_weight = float("inf")
        for left in left_positions:
            for right in right_positions:
                try:
                    path = nx.shortest_path(graph, left, right, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                weight = _path_weight(graph, path)
                if weight < best_weight or (
                    weight == best_weight and best_path is not None and len(path) < len(best_path)
                ):
                    best_path = path
                    best_weight = weight
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


def relation_weight(relation: str) -> int:
    normalized = relation.strip()
    base = normalized.split(":", 1)[0]
    if normalized in LOW_WEIGHT_RELATIONS:
        return 1
    if normalized in MEDIUM_WEIGHT_RELATIONS or base in MEDIUM_WEIGHT_BASES:
        return 3
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
    weight: int,
    directed_source: int,
    directed_target: int,
) -> None:
    if graph.has_edge(source, target):
        relations = list(graph.edges[source, target].get("relations", []))
        directed_edges = list(graph.edges[source, target].get("directed_edges", []))
        weight = min(int(graph.edges[source, target].get("weight", weight)), weight)
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


def _path_weight(graph: nx.Graph, path: list[Any]) -> int:
    total = 0
    for left, right in zip(path, path[1:]):
        total += int(graph.edges[left, right].get("weight", DEFAULT_RELATION_WEIGHT))
    return total


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
