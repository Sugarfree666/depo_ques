from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import networkx as nx

from models import AnchorGraph, ASTResult, ExtractionResult, OperatorSelection, PlaceholderReplacement
from models import (
    AnchorSelectionResult,
    RestoredAnchorConnectedSubgraph,
    SelectedAnchor,
    SemanticASTEdge,
    SemanticASTNode,
    SemanticASTPrimaryOperator,
    SemanticASTResult,
)
from prompts import (
    ALLOWED_OPERATORS,
    OPERATOR_SELECTION_SYSTEM,
    SEMANTIC_AST_OPTIMIZATION_SYSTEM,
    build_operator_prompt,
    build_semantic_ast_optimization_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


class ASTBuilder:
    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def build(
        self,
        question: str,
        extraction: ExtractionResult,
        replacement: PlaceholderReplacement,
        anchor_graph: AnchorGraph,
    ) -> ASTResult:
        node_lookup = extraction.placeholder_to_node
        anchor_nodes = [
            {
                "placeholder": node.placeholder,
                "text": node.text,
                "kind": node.kind,
                "semantic_type": node.semantic_type,
            }
            for node in extraction.nodes
        ]
        anchor_edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "collapsed_dependency_path": edge.path_words,
                "relations": edge.relations,
            }
            for edge in anchor_graph.edges
        ]

        payload = self.llm_client.chat_json(
            OPERATOR_SELECTION_SYSTEM,
            build_operator_prompt(question, anchor_nodes, anchor_edges),
        )
        operators = self._parse_operators(payload.get("operators", []), anchor_graph.graph)
        if not operators:
            operators = [OperatorSelection(operator="NONE", attach_to=[])]

        ast_graph = anchor_graph.graph.copy()
        for selection in operators:
            if selection.operator == "NONE":
                continue
            operator_node = self._operator_node_id(ast_graph, selection.operator)
            ast_graph.add_node(
                operator_node,
                kind="operator",
                text=selection.operator,
                semantic_type="Operator",
                order=10**9 + len(ast_graph.nodes),
            )
            attach_to = selection.attach_to or self._default_attach_nodes(selection.operator, ast_graph)
            selection.attach_to = attach_to
            for anchor in attach_to:
                if anchor in ast_graph:
                    ast_graph.add_edge(anchor, operator_node, operator=True, operator_name=selection.operator)

        labels = dict(replacement.mapping)
        for node, attrs in ast_graph.nodes(data=True):
            if attrs.get("kind") == "operator":
                labels[node] = str(attrs.get("text", node))
            elif node in node_lookup:
                labels.setdefault(node, node_lookup[node].text)
        return ASTResult(graph=ast_graph, operators=operators, label_by_placeholder=labels)

    @staticmethod
    def _parse_operators(raw: Any, graph: nx.Graph) -> list[OperatorSelection]:
        if not isinstance(raw, list):
            return []
        selections: list[OperatorSelection] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            operator = _canonical_operator(str(item.get("operator", "NONE")).strip().upper())
            if operator not in ALLOWED_OPERATORS:
                continue
            attach_to = item.get("attach_to", [])
            if not isinstance(attach_to, list):
                attach_to = []
            valid_attach = [str(anchor) for anchor in attach_to if str(anchor) in graph]
            valid_attach = _normalize_operator_attachment(operator, valid_attach, graph)
            explanation = str(item.get("explanation", ""))
            selections.append(
                OperatorSelection(
                    operator=operator,
                    attach_to=valid_attach,
                    explanation=explanation,
                )
            )
        return selections

    @staticmethod
    def _operator_node_id(graph: nx.Graph, operator: str) -> str:
        if operator not in graph:
            return operator
        index = 2
        while f"{operator}_{index}" in graph:
            index += 1
        return f"{operator}_{index}"

    @staticmethod
    def _default_attach_nodes(operator: str, graph: nx.Graph) -> list[str]:
        non_operator_nodes = [
            node for node, attrs in graph.nodes(data=True) if attrs.get("kind") != "operator"
        ]
        if not non_operator_nodes:
            return []
        if operator.startswith("COMPARE"):
            type_nodes = [
                node
                for node in non_operator_nodes
                if graph.nodes[node].get("kind") == "type_variable"
            ]
            if type_nodes:
                return [max(type_nodes, key=lambda node: graph.nodes[node].get("order", 0))]
        return [max(non_operator_nodes, key=lambda node: graph.degree(node))]


def _canonical_operator(operator: str) -> str:
    aliases = {
        "COMPARE_DIFFERENT": "COMPARE_DIFF",
        "AND": "LOGICAL_AND",
        "OR": "LOGICAL_OR",
        "BRIDGE": "NONE",
        "COUNT": "NONE",
        "FILTER": "NONE",
    }
    return aliases.get(operator, operator)


def _replacement_view(replacement: PlaceholderReplacement) -> dict[str, Any]:
    return {
        "original_question": replacement.original_question,
        "masked_question": replacement.masked_question,
        "mask_mappings": [
            mapping.to_dict() if hasattr(mapping, "to_dict") else mapping
            for mapping in getattr(replacement, "mask_mappings", [])
        ],
        "mask_mapping": replacement.mask_mapping,
    }


def _parse_primary_operator(
    raw: Any,
    warnings: list[str],
) -> SemanticASTPrimaryOperator:
    if isinstance(raw, str):
        operator = _canonical_operator(raw.strip().upper())
        if operator not in ALLOWED_OPERATORS:
            warnings.append(f"Invalid primary_operator={operator!r}; using NONE.")
            operator = "NONE"
        return SemanticASTPrimaryOperator(operator=operator)
    if not isinstance(raw, dict):
        return SemanticASTPrimaryOperator(operator="NONE")
    operator = _canonical_operator(str(raw.get("operator", "NONE")).strip().upper())
    if operator not in ALLOWED_OPERATORS:
        warnings.append(f"Invalid primary_operator={operator!r}; using NONE.")
        operator = "NONE"
    inputs = raw.get("inputs", [])
    if not isinstance(inputs, list):
        inputs = []
    return SemanticASTPrimaryOperator(
        operator=operator,
        inputs=[str(item) for item in inputs],
        output=str(raw.get("output", "answer") or "answer"),
        cue_text=str(raw.get("cue_text", raw.get("cue", "")) or ""),
        explanation=str(raw.get("explanation", "") or ""),
    )


def _parse_semantic_node(
    raw: Any,
    valid_graph_node_ids: set[str],
    selected_entity_texts: set[str],
    warnings: list[str],
) -> SemanticASTNode | None:
    if not isinstance(raw, dict):
        return None
    node_id = str(raw.get("id", "")).strip()
    label = str(raw.get("label", "")).strip()
    kind = str(raw.get("kind", "")).strip()
    if not node_id or not label or kind not in {
        "entity",
        "type_variable",
        "implicit_type_variable",
        "operator",
        "variable",
    }:
        warnings.append(f"Dropped invalid semantic AST node: {raw!r}.")
        return None
    source_graph_nodes = raw.get("source_graph_nodes", [])
    if not isinstance(source_graph_nodes, list):
        source_graph_nodes = []
    source_graph_nodes = [str(item) for item in source_graph_nodes]
    invalid_sources = [item for item in source_graph_nodes if item not in valid_graph_node_ids]
    if invalid_sources:
        warnings.append(
            f"Dropped invalid source_graph_nodes for semantic node {node_id}: {invalid_sources}."
        )
        source_graph_nodes = [item for item in source_graph_nodes if item in valid_graph_node_ids]

    source = str(raw.get("source", "derived") or "derived")
    if kind == "implicit_type_variable" and not (
        str(raw.get("cue_text", "") or "").strip()
        or str(raw.get("created_from_cue", "") or "").strip()
    ):
        warnings.append(f"Dropped implicit semantic node {node_id} without cue_text.")
        return None
    if kind == "entity":
        allowed_source = source in {"selected_anchor", "mask", "derived"}
        grounded_in_selected = _norm(label) in selected_entity_texts or bool(source_graph_nodes)
        if not allowed_source or not grounded_in_selected:
            warnings.append(f"Dropped ungrounded/invented entity node {node_id}: {label!r}.")
            return None

    token_indices = raw.get("source_token_indices", [])
    if not isinstance(token_indices, list):
        token_indices = []
    coerced_token_indices: list[int] = []
    for item in token_indices:
        try:
            coerced_token_indices.append(int(item))
        except (TypeError, ValueError):
            continue

    return SemanticASTNode(
        id=node_id,
        label=label,
        kind=kind,
        semantic_type=str(raw.get("semantic_type", "") or "") or None,
        source=source,
        source_graph_nodes=source_graph_nodes,
        source_token_indices=coerced_token_indices,
        grounding_text=str(raw.get("grounding_text", "") or ""),
        cue_text=str(raw.get("cue_text", raw.get("created_from_cue", "")) or ""),
    )


def _parse_semantic_edge(
    raw: Any,
    node_ids: set[str],
    warnings: list[str],
) -> SemanticASTEdge | None:
    if not isinstance(raw, dict):
        return None
    source = str(raw.get("source", "")).strip()
    target = str(raw.get("target", "")).strip()
    if source not in node_ids or target not in node_ids:
        warnings.append(f"Dropped semantic edge with unknown endpoints: {raw!r}.")
        return None
    support_path = raw.get("support_path", [])
    if not isinstance(support_path, list):
        support_path = []
    relations = raw.get("support_dependency_relations", raw.get("relations", []))
    if not isinstance(relations, list):
        relations = []
    return SemanticASTEdge(
        source=source,
        target=target,
        edge_type=str(raw.get("edge_type", "attribute") or "attribute"),
        relation_hint=str(raw.get("relation_hint", "") or ""),
        support_path=[str(item) for item in support_path],
        support_dependency_relations=[str(item) for item in relations],
    )


def _fallback_semantic_ast(
    selected_anchors: list[SelectedAnchor],
    restored_anchor_connected_subgraph: RestoredAnchorConnectedSubgraph,
    warnings: list[str],
    raw_payload: dict[str, Any] | None,
) -> SemanticASTResult:
    nodes: list[SemanticASTNode] = []
    id_by_graph_node: dict[str, str] = {}
    counters: dict[str, int] = {}
    for anchor in selected_anchors:
        base = _slug(anchor.display_text or anchor.restored_text or anchor.graph_text)
        counters[base] = counters.get(base, 0) + 1
        node_id = f"{base}_{counters[base]}"
        id_by_graph_node[anchor.node_id] = node_id
        nodes.append(
            SemanticASTNode(
                id=node_id,
                label=anchor.display_text,
                kind=anchor.anchor_kind,
                semantic_type=anchor.semantic_type_hint,
                source="selected_anchor",
                source_graph_nodes=[anchor.node_id],
                source_token_indices=[anchor.token_index] if anchor.token_index is not None else [],
                grounding_text=anchor.display_text,
            )
        )

    edges: list[SemanticASTEdge] = []
    for path in restored_anchor_connected_subgraph.shortest_paths:
        source = id_by_graph_node.get(str(path.get("source")))
        target = id_by_graph_node.get(str(path.get("target")))
        if source is None or target is None:
            continue
        edges.append(
            SemanticASTEdge(
                source=source,
                target=target,
                edge_type="related",
                relation_hint="syntactic evidence path",
                support_path=[str(item) for item in path.get("path_words", [])],
                support_dependency_relations=[str(item) for item in path.get("relations", [])],
            )
        )
    warnings.append("Used conservative semantic AST fallback with primary_operator=NONE.")
    return SemanticASTResult(
        status="fallback",
        primary_operator=SemanticASTPrimaryOperator(operator="NONE"),
        nodes=nodes,
        edges=edges,
        warnings=warnings,
        raw_payload=raw_payload,
    )


def _slug(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    if not words:
        return "node"
    return "_".join(words[:3])


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _normalize_operator_attachment(operator: str, attach_to: list[str], graph: nx.Graph) -> list[str]:
    if not operator.startswith("COMPARE") or not attach_to:
        return attach_to
    if any(graph.nodes[node].get("kind") == "type_variable" for node in attach_to):
        return attach_to

    attribute_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("kind") == "type_variable"
    ]
    if not attribute_nodes:
        return attach_to
        return [max(attribute_nodes, key=lambda node: graph.nodes[node].get("order", 0))]


class SemanticASTOptimizer:
    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.llm_client = llm_client

    def optimize(
        self,
        original_question: str,
        replacement: PlaceholderReplacement,
        selected_anchors: list[SelectedAnchor] | AnchorSelectionResult,
        restored_anchor_connected_subgraph: RestoredAnchorConnectedSubgraph,
    ) -> SemanticASTResult:
        selected_anchor_list = (
            selected_anchors.selected_anchors
            if isinstance(selected_anchors, AnchorSelectionResult)
            else selected_anchors
        )
        warnings: list[str] = []
        payload: dict[str, Any] = {}
        if self.llm_client is not None:
            try:
                payload = self.llm_client.chat_json(
                    SEMANTIC_AST_OPTIMIZATION_SYSTEM,
                    build_semantic_ast_optimization_prompt(
                        original_question=original_question,
                        replacement=_replacement_view(replacement),
                        selected_anchors=[anchor.to_llm_view() for anchor in selected_anchor_list],
                        restored_anchor_connected_subgraph=restored_anchor_connected_subgraph.to_dict(),
                        allowed_operators=ALLOWED_OPERATORS,
                    ),
                )
            except Exception as exc:
                warnings.append(f"Semantic AST optimization LLM failed; using fallback: {exc}")
        else:
            warnings.append("Semantic AST optimization LLM unavailable; using fallback.")

        result = self._parse_and_validate(
            payload=payload,
            selected_anchors=selected_anchor_list,
            restored_anchor_connected_subgraph=restored_anchor_connected_subgraph,
            warnings=warnings,
        )
        if result is None:
            return _fallback_semantic_ast(
                selected_anchors=selected_anchor_list,
                restored_anchor_connected_subgraph=restored_anchor_connected_subgraph,
                warnings=warnings,
                raw_payload=payload or None,
            )
        result.warnings.extend(warnings)
        result.raw_payload = payload or None
        return result

    def _parse_and_validate(
        self,
        payload: dict[str, Any],
        selected_anchors: list[SelectedAnchor],
        restored_anchor_connected_subgraph: RestoredAnchorConnectedSubgraph,
        warnings: list[str],
    ) -> SemanticASTResult | None:
        if not payload:
            return None
        valid_graph_node_ids = {
            str(node.get("node_id"))
            for node in restored_anchor_connected_subgraph.nodes
            if node.get("node_id") is not None
        } | {anchor.node_id for anchor in selected_anchors}
        selected_entity_texts = {
            _norm(anchor.display_text)
            for anchor in selected_anchors
            if anchor.anchor_kind == "entity"
        }
        primary_operator = _parse_primary_operator(payload.get("primary_operator"), warnings)
        raw_nodes = payload.get("nodes", [])
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            warnings.append("Semantic AST payload did not contain list nodes/edges.")
            return None

        nodes: list[SemanticASTNode] = []
        seen_ids: set[str] = set()
        for raw in raw_nodes:
            node = _parse_semantic_node(raw, valid_graph_node_ids, selected_entity_texts, warnings)
            if node is None:
                continue
            if node.id in seen_ids:
                warnings.append(f"Dropped duplicate semantic AST node id={node.id!r}.")
                continue
            seen_ids.add(node.id)
            nodes.append(node)
        if not nodes:
            warnings.append("Semantic AST payload had no valid nodes.")
            return None

        node_ids = {node.id for node in nodes}
        edges: list[SemanticASTEdge] = []
        for raw in raw_edges:
            edge = _parse_semantic_edge(raw, node_ids, warnings)
            if edge is not None:
                edges.append(edge)

        if primary_operator.operator != "NONE":
            missing_inputs = [item for item in primary_operator.inputs if item not in node_ids]
            if missing_inputs:
                warnings.append(
                    "Primary operator inputs not present in AST nodes: " + ", ".join(missing_inputs)
                )
                primary_operator.inputs = [item for item in primary_operator.inputs if item in node_ids]
            if not primary_operator.cue_text and not primary_operator.explanation:
                warnings.append("Non-NONE primary operator lacked cue_text/explanation; using NONE.")
                primary_operator = SemanticASTPrimaryOperator(operator="NONE")

        status = str(payload.get("status", "ok")).strip() or "ok"
        return SemanticASTResult(
            status=status,
            primary_operator=primary_operator,
            nodes=nodes,
            edges=edges,
            warnings=[],
            raw_payload=payload,
        )
