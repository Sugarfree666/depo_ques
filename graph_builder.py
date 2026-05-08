from __future__ import annotations

import itertools
from collections import deque
from typing import Callable

import networkx as nx

from models import AnchorEdge, AnchorGraph, DependencyParse, ExtractedNode, ExtractionResult


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

        token_graph = nx.Graph()
        token_words = {token.index: token.word for token in dependency_parse.tokens}
        for token in dependency_parse.tokens:
            token_graph.add_node(token.index, word=token.word)

        for edge in dependency_parse.edges:
            token_graph.add_edge(
                edge.source_index,
                edge.target_index,
                weight=1,
                relation=edge.relation,
            )

        anchor_positions = {
            anchor: [index for index, word in token_words.items() if word == anchor]
            for anchor in anchors
        }
        missing = [anchor for anchor, positions in anchor_positions.items() if not positions]
        if missing:
            raise AnchorGraphError(
                "The following anchors were not found as CoreNLP tokens after placeholder replacement: "
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
            path = self._shortest_anchor_path(
                token_graph,
                anchor_positions[left],
                anchor_positions[right],
            )
            if path is None:
                continue
            relations = _relations_for_path(token_graph, path)
            closure.add_edge(
                left,
                right,
                weight=max(len(path) - 1, 1),
                token_path=path,
                path_words=[token_words.get(index, str(index)) for index in path],
                relations=relations,
            )

        if not nx.is_connected(closure):
            components = [sorted(component) for component in nx.connected_components(closure)]
            raise AnchorGraphError(
                "Could not connect all anchors in the dependency graph. Components: "
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

    @staticmethod
    def _shortest_anchor_path(
        graph: nx.Graph,
        left_positions: list[int],
        right_positions: list[int],
    ) -> list[int] | None:
        best_path: list[int] | None = None
        for left in left_positions:
            for right in right_positions:
                try:
                    path = nx.shortest_path(graph, left, right, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                if best_path is None or len(path) < len(best_path):
                    best_path = path
        return best_path


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


def _relations_for_path(graph: nx.Graph, path: list[int]) -> list[str]:
    relations: list[str] = []
    for left, right in zip(path, path[1:]):
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
        # Prefer later non-entity answer/result variables when distances tie.
        return (max_distance, total, -_node_order(graph, node))

    return min(candidates, key=score)

