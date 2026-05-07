from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import re
from typing import Any

import networkx as nx

from hypergraph_rag.models import ASTEdge, ASTNode, DependencyParse, ExtractedQueryUnits


EDGE_KIND_RELATION = "relation"
EDGE_KIND_OPERATOR_INPUT = "operator_input"
EDGE_KIND_OPERATOR_OUTPUT = "operator_output"


def construct_query_ast(
    question: str,
    extracted_units: ExtractedQueryUnits,
    dependency_parse: DependencyParse,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph["question"] = question
    graph.graph["branch_hints"] = [item.to_dict() for item in extracted_units.branch_hints]
    graph.graph["raw_extraction_output"] = extracted_units.raw_response
    graph.graph["dependency_warnings"] = list(dependency_parse.warnings)

    answer_node_ids: set[str] = set()
    node_ids: set[str] = set()
    anchor_lookup = _build_anchor_lookup(question, dependency_parse)

    for item in extracted_units.entities + extracted_units.type_variables:
        node = ASTNode(
            id=item.id,
            label=item.label,
            node_kind=item.kind,
            semantic_type=item.semantic_type,
            source_span=item.source_span,
            is_answer=item.is_answer,
            metadata=dict(item.metadata),
        )
        node_dict = node.to_dict()
        node_dict["anchor"] = _resolve_anchor(question, item.source_span, item.label, anchor_lookup)
        graph.add_node(node.id, **node_dict)
        node_ids.add(node.id)
        if item.is_answer:
            answer_node_ids.add(item.id)

    relation_units = sorted(
        extracted_units.relations,
        key=lambda item: (
            item.order_hint if item.order_hint is not None else 10_000,
            _node_anchor(graph, item.source_id),
            _node_anchor(graph, item.target_id),
            item.id,
        ),
    )

    edge_ids: set[str] = set()
    for index, relation in enumerate(relation_units, start=1):
        if relation.source_id not in node_ids:
            raise ValueError(f"Unknown relation source node: {relation.source_id}")
        if relation.target_id not in node_ids:
            raise ValueError(f"Unknown relation target node: {relation.target_id}")
        edge = ASTEdge(
            id=_dedupe_id(_sanitize_id(relation.id, f"edge_{index}"), edge_ids),
            source_id=relation.source_id,
            target_id=relation.target_id,
            relation=_sanitize_id(relation.relation, f"relation_{index}"),
            edge_kind=EDGE_KIND_RELATION,
            order=relation.order_hint or index,
            confidence=relation.confidence,
            metadata={
                "surface": relation.surface,
                **relation.metadata,
            },
        )
        graph.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
        edge_ids.add(edge.id)

    operator_node_ids: list[str] = []
    for index, operator in enumerate(extracted_units.operators, start=1):
        operator_name = _sanitize_id(operator.operator, f"operator_{index}")
        operator_node_id = _dedupe_id(_sanitize_id(operator.id, f"op_{operator_name}_{index}"), node_ids)
        output_id = operator.output_id or f"x_{operator_name}_{index}_result"
        output_id = _sanitize_id(output_id, f"x_{operator_name}_{index}_result")

        if output_id not in node_ids:
            output_node = ASTNode(
                id=output_id,
                label=operator.output_label or operator_name,
                node_kind="type_variable",
                semantic_type=operator.output_type,
                is_answer=True,
                metadata={"created_by_operator": operator_node_id},
            )
            graph.add_node(output_node.id, **output_node.to_dict(), anchor=len(question))
            node_ids.add(output_node.id)

        answer_node_ids.add(output_id)

        operator_node = ASTNode(
            id=operator_node_id,
            label=operator_name,
            node_kind="operator",
            semantic_type="LogicalOperator",
            metadata={
                "operator": operator_name,
                "description": operator.description,
                "input_ids": list(operator.input_ids),
                "output_id": output_id,
                "operator_order": index,
                **operator.metadata,
            },
        )
        graph.add_node(operator_node.id, **operator_node.to_dict(), anchor=len(question) + index)
        node_ids.add(operator_node.id)
        operator_node_ids.append(operator_node.id)

        for input_index, input_id in enumerate(operator.input_ids, start=1):
            if input_id not in node_ids:
                raise ValueError(f"Unknown operator input node: {input_id}")
            input_edge = ASTEdge(
                id=_dedupe_id(f"{operator_node.id}_arg{input_index}", edge_ids),
                source_id=input_id,
                target_id=operator_node.id,
                relation=operator_name,
                edge_kind=EDGE_KIND_OPERATOR_INPUT,
                source_role=f"arg{input_index}",
                order=10_000 + index,
            )
            graph.add_edge(input_edge.source_id, input_edge.target_id, **input_edge.to_dict())
            edge_ids.add(input_edge.id)

        output_edge = ASTEdge(
            id=_dedupe_id(f"{operator_node.id}_out", edge_ids),
            source_id=operator_node.id,
            target_id=output_id,
            relation=operator_name,
            edge_kind=EDGE_KIND_OPERATOR_OUTPUT,
            source_role="output",
            order=20_000 + index,
        )
        graph.add_edge(output_edge.source_id, output_edge.target_id, **output_edge.to_dict())
        edge_ids.add(output_edge.id)

    graph.graph["answer_node_ids"] = sorted(answer_node_ids)
    graph.graph["operator_node_ids"] = operator_node_ids

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Constructed query graph is not a DAG.")

    return graph


def graph_to_dict(graph: nx.DiGraph) -> dict[str, Any]:
    return {
        "question": graph.graph.get("question", ""),
        "answer_node_ids": list(graph.graph.get("answer_node_ids", [])),
        "branch_hints": list(graph.graph.get("branch_hints", [])),
        "nodes": [
            {"id": node_id, **dict(attributes)}
            for node_id, attributes in graph.nodes(data=True)
        ],
        "edges": [
            dict(attributes)
            for _, _, attributes in graph.edges(data=True)
        ],
    }


def graph_to_edge_lines(graph: nx.DiGraph) -> list[str]:
    lines: list[str] = []
    for edge in iter_relation_edges(graph):
        source = graph.nodes[edge["source_id"]]
        target = graph.nodes[edge["target_id"]]
        lines.append(
            f"{_display_node(source)} --[{edge['relation']}]--> {_display_node(target)}"
        )

    for operator_node_id in iter_operator_node_ids(graph):
        node = graph.nodes[operator_node_id]
        input_ids = list(node["metadata"].get("input_ids", []))
        output_id = str(node["metadata"].get("output_id", ""))
        operands = " and ".join(input_ids) if len(input_ids) == 2 else ", ".join(input_ids)
        output_node = graph.nodes[output_id]
        lines.append(
            f"{operands} --[{node['metadata']['operator']}]--> {_display_node(output_node)}"
        )
    return lines


def iter_relation_edges(graph: nx.DiGraph) -> list[dict[str, Any]]:
    edges = [
        dict(attributes)
        for _, _, attributes in graph.edges(data=True)
        if attributes.get("edge_kind") == EDGE_KIND_RELATION
    ]
    return sorted(edges, key=lambda item: (item.get("order", 0), item["id"]))


def iter_operator_node_ids(graph: nx.DiGraph) -> list[str]:
    node_ids = [
        node_id
        for node_id, attributes in graph.nodes(data=True)
        if attributes.get("node_kind") == "operator"
    ]
    return sorted(
        node_ids,
        key=lambda node_id: (
            graph.nodes[node_id].get("metadata", {}).get("operator_order", 0),
            node_id,
        ),
    )


def _display_node(node: dict[str, Any]) -> str:
    if node["node_kind"] == "entity":
        return node["label"]
    semantic_type = node.get("semantic_type", "")
    if semantic_type:
        return f"{node['id']}:{semantic_type}"
    return node["id"]


def _sanitize_id(value: str, default_prefix: str) -> str:
    candidate = value.strip()
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_").lower()
    if not candidate:
        candidate = default_prefix
    if not re.match(r"^[A-Za-z_]", candidate):
        candidate = f"{default_prefix}_{candidate}"
    return candidate


def _dedupe_id(candidate: str, existing_ids: set[str]) -> str:
    if candidate not in existing_ids:
        return candidate
    index = 2
    while f"{candidate}_{index}" in existing_ids:
        index += 1
    return f"{candidate}_{index}"


def _build_anchor_lookup(
    question: str,
    dependency_parse: DependencyParse,
) -> dict[str, list[int]]:
    anchors: dict[str, list[int]] = defaultdict(list)
    lower_question = question.lower()
    for span in dependency_parse.named_entities + dependency_parse.noun_chunks:
        anchors[span.text.lower()].append(span.start_char)
    for token in dependency_parse.tokens:
        anchors[token.text.lower()].append(token.start_char)
    anchors[lower_question].append(0)
    return anchors


def _resolve_anchor(
    question: str,
    source_span: str,
    label: str,
    anchor_lookup: dict[str, list[int]],
) -> int:
    candidates = [source_span.strip().lower(), label.strip().lower()]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in anchor_lookup and anchor_lookup[candidate]:
            return min(anchor_lookup[candidate])
        index = question.lower().find(candidate)
        if index >= 0:
            return index
    return len(question)


def _node_anchor(graph: nx.DiGraph, node_id: str) -> int:
    if node_id not in graph:
        return 10_000
    return int(graph.nodes[node_id].get("anchor", 10_000))
