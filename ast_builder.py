from __future__ import annotations

from typing import Any

import networkx as nx

from llm_client import LLMClient
from models import AnchorGraph, ASTResult, ExtractionResult, OperatorSelection, PlaceholderReplacement
from prompts import ALLOWED_OPERATORS, OPERATOR_SELECTION_SYSTEM, build_operator_prompt


class ASTBuilder:
    def __init__(self, llm_client: LLMClient) -> None:
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
            if selection.operator in {"NONE", "BRIDGE"}:
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
                labels[node] = node_lookup[node].text
        return ASTResult(graph=ast_graph, operators=operators, label_by_placeholder=labels)

    @staticmethod
    def _parse_operators(raw: Any, graph: nx.Graph) -> list[OperatorSelection]:
        if not isinstance(raw, list):
            return []
        selections: list[OperatorSelection] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            operator = str(item.get("operator", "NONE")).strip().upper()
            if operator not in ALLOWED_OPERATORS:
                continue
            attach_to = item.get("attach_to", [])
            if not isinstance(attach_to, list):
                attach_to = []
            valid_attach = [str(anchor) for anchor in attach_to if str(anchor) in graph]
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

