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
    label_by_graph_node: dict[str, str],
    selected_entity_texts: set[str],
    warnings: list[str],
) -> SemanticASTNode | None:
    if not isinstance(raw, dict):
        return None
    node_id = str(raw.get("id", "")).strip()
    raw_label = str(raw.get("label", "")).strip()
    kind = str(raw.get("kind", "")).strip()
    if not node_id or not raw_label or kind not in {
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
    grounding_text = str(raw.get("grounding_text", "") or "")
    label = _normalize_semantic_node_label(
        node_id=node_id,
        raw_label=raw_label,
        kind=kind,
        grounding_text=grounding_text,
        source_graph_nodes=source_graph_nodes,
        label_by_graph_node=label_by_graph_node,
    )
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
        grounding_text=grounding_text,
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


def _normalize_semantic_node_label(
    node_id: str,
    raw_label: str,
    kind: str,
    grounding_text: str,
    source_graph_nodes: list[str],
    label_by_graph_node: dict[str, str],
) -> str:
    if kind == "operator":
        return raw_label

    graph_labels = [
        label_by_graph_node[node_id]
        for node_id in source_graph_nodes
        if node_id in label_by_graph_node and label_by_graph_node[node_id].strip()
    ]
    graph_label = graph_labels[0] if graph_labels else ""

    label = raw_label.strip()
    if _is_bad_semantic_label(label, node_id, kind):
        if grounding_text and not _is_bad_semantic_label(grounding_text, node_id, kind):
            label = grounding_text.strip()
        elif graph_label:
            label = graph_label.strip()
        else:
            label = _identifier_to_label(label or node_id)

    if kind in {"type_variable", "implicit_type_variable", "variable"} and graph_label:
        if _looks_like_relation_phrase(label) and not _looks_like_relation_phrase(graph_label):
            label = graph_label.strip()

    if _looks_like_identifier(label):
        label = _identifier_to_label(label)
    return label or raw_label


def _is_bad_semantic_label(label: str, node_id: str, kind: str) -> bool:
    if not label:
        return True
    if label == node_id:
        return True
    if kind in {"type_variable", "implicit_type_variable", "variable"} and _looks_like_identifier(label):
        return True
    return False


def _looks_like_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*_[0-9]+", value.strip()))


def _identifier_to_label(value: str) -> str:
    text = re.sub(r"_[0-9]+$", "", value.strip())
    return text.replace("_", " ")


def _looks_like_relation_phrase(value: str) -> bool:
    return bool(re.search(r"\b(of|by|from|in|for|to|with|that|who|which)\b", value.lower()))


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


def _orient_edges_for_inference(
    original_question: str,
    nodes: list[SemanticASTNode],
    edges: list[SemanticASTEdge],
    primary_operator: SemanticASTPrimaryOperator,
    warnings: list[str],
) -> list[SemanticASTEdge]:
    if len(edges) <= 1:
        return edges

    node_by_id = {node.id: node for node in nodes}
    roots = _infer_inference_roots(original_question, nodes, primary_operator)
    if not roots:
        return edges

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    edge_lookup: dict[frozenset[str], SemanticASTEdge] = {}
    for edge in edges:
        key = frozenset({edge.source, edge.target})
        if len(key) != 2:
            continue
        graph.add_edge(edge.source, edge.target)
        edge_lookup.setdefault(key, edge)

    oriented_edges: list[SemanticASTEdge] = []
    used_keys: set[frozenset[str]] = set()
    for component in nx.connected_components(graph):
        component_roots = [root for root in roots if root in component]
        if not component_roots:
            component_roots = [_best_component_root(component, node_by_id, original_question, primary_operator)]
        distances = _multi_source_distances(graph.subgraph(component), component_roots)
        component_edges = sorted(
            graph.subgraph(component).edges,
            key=lambda item: (
                min(distances.get(item[0], 10**9), distances.get(item[1], 10**9)),
                max(distances.get(item[0], 10**9), distances.get(item[1], 10**9)),
                item[0],
                item[1],
            ),
        )
        for source, target in component_edges:
            key = frozenset({source, target})
            edge = edge_lookup.get(key)
            if edge is None or key in used_keys:
                continue
            source_distance = distances.get(source, 10**9)
            target_distance = distances.get(target, 10**9)
            if target_distance < source_distance:
                source, target = target, source
            used_keys.add(key)
            oriented_edges.append(_copy_edge_with_direction(edge, source, target, warnings))

    for edge in edges:
        key = frozenset({edge.source, edge.target})
        if key not in used_keys:
            oriented_edges.append(edge)
    return oriented_edges


def _materialize_operator_node(
    nodes: list[SemanticASTNode],
    edges: list[SemanticASTEdge],
    primary_operator: SemanticASTPrimaryOperator,
) -> None:
    if primary_operator.operator == "NONE":
        return
    existing_node_ids = {node.id for node in nodes}
    operator_node_id = _operator_node_id(primary_operator.operator, existing_node_ids)
    if operator_node_id not in existing_node_ids:
        nodes.append(
            SemanticASTNode(
                id=operator_node_id,
                label=primary_operator.operator,
                kind="operator",
                semantic_type="Operator",
                source="operator_from_cue",
                grounding_text=primary_operator.cue_text,
                cue_text=primary_operator.cue_text,
            )
        )
    existing_edges = {(edge.source, edge.target, edge.edge_type) for edge in edges}
    for input_node in primary_operator.inputs:
        key = (input_node, operator_node_id, "operator")
        if key in existing_edges:
            continue
        edges.append(
            SemanticASTEdge(
                source=input_node,
                target=operator_node_id,
                edge_type="operator",
                relation_hint=primary_operator.operator,
                support_path=[primary_operator.cue_text] if primary_operator.cue_text else [],
                support_dependency_relations=[],
            )
        )


def _operator_node_id(operator: str, existing_node_ids: set[str]) -> str:
    base = "operator_" + operator.lower()
    if base not in existing_node_ids:
        return base
    index = 2
    while f"{base}_{index}" in existing_node_ids:
        index += 1
    return f"{base}_{index}"


def _infer_inference_roots(
    original_question: str,
    nodes: list[SemanticASTNode],
    primary_operator: SemanticASTPrimaryOperator,
) -> list[str]:
    entity_roots = [
        node.id
        for node in nodes
        if node.kind == "entity" and node.source in {"selected_anchor", "mask", "derived"}
    ]
    if entity_roots:
        return entity_roots

    explicit_type_nodes = [node for node in nodes if node.kind == "type_variable"]
    if primary_operator.operator in {"COMPARE_GREATER", "COMPARE_LESS"}:
        comparison_subjects = [
            node.id
            for node in explicit_type_nodes
            if node.id not in set(primary_operator.inputs)
        ]
        if comparison_subjects:
            return comparison_subjects

    if primary_operator.operator in {"ARGMAX", "ARGMIN"} and primary_operator.output:
        output_roots = [node.id for node in nodes if node.id == primary_operator.output]
        if output_roots:
            return output_roots

    focus_labels = _answer_focus_labels(original_question)
    focus_roots = [
        node.id
        for node in explicit_type_nodes
        if _norm(node.label) in focus_labels or any(_norm(part) in focus_labels for part in node.label.split())
    ]
    if focus_roots:
        return focus_roots

    if explicit_type_nodes:
        return [explicit_type_nodes[0].id]
    return []


def _best_component_root(
    component: set[str],
    node_by_id: dict[str, SemanticASTNode],
    original_question: str,
    primary_operator: SemanticASTPrimaryOperator,
) -> str:
    focus_labels = _answer_focus_labels(original_question)

    def score(node_id: str) -> tuple[int, int]:
        node = node_by_id[node_id]
        if node.kind == "entity":
            kind_score = 0
        elif node.kind == "type_variable" and node.id == primary_operator.output:
            kind_score = 1
        elif node.kind == "type_variable" and _norm(node.label) in focus_labels:
            kind_score = 2
        elif node.kind == "type_variable":
            kind_score = 3
        else:
            kind_score = 4
        return (kind_score, len(node_id))

    return min(component, key=score)


def _multi_source_distances(graph: nx.Graph, roots: list[str]) -> dict[str, int]:
    distances: dict[str, int] = {}
    queue: list[str] = []
    for root in roots:
        if root in graph and root not in distances:
            distances[root] = 0
            queue.append(root)
    while queue:
        current = queue.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _copy_edge_with_direction(
    edge: SemanticASTEdge,
    source: str,
    target: str,
    warnings: list[str],
) -> SemanticASTEdge:
    if edge.source == source and edge.target == target:
        return edge
    warnings.append(f"Reoriented semantic edge {edge.source}->{edge.target} to {source}->{target}.")
    return SemanticASTEdge(
        source=source,
        target=target,
        edge_type=edge.edge_type,
        relation_hint=edge.relation_hint,
        support_path=edge.support_path,
        support_dependency_relations=edge.support_dependency_relations,
    )


def _answer_focus_labels(question: str) -> set[str]:
    text = _norm(question)
    labels: set[str] = set()
    patterns = [
        r"^(?:which|what)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,3})\b",
        r"^(?:who)\b",
        r"^what\s+is\s+(?:the|a|an)?\s*([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,3})\b",
    ]
    stop_words = {
        "did",
        "does",
        "do",
        "has",
        "have",
        "had",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "that",
        "which",
        "who",
    }
    cue_words = {
        "different",
        "highest",
        "larger",
        "largest",
        "older",
        "same",
        "smallest",
        "younger",
    }
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if pattern == r"^(?:who)\b":
            labels.add("person")
            labels.add("people")
            continue
        phrase = match.group(1)
        words = [word for word in phrase.split() if word not in stop_words and word not in cue_words]
        for index in range(len(words)):
            labels.add(" ".join(words[index:]))
        for word in words:
            labels.add(word)
        if words:
            labels.add(words[-1])
    return labels


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
            original_question=original_question,
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
        original_question: str,
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
        label_by_graph_node = {
            str(node.get("node_id")): str(node.get("text", node.get("display_text", "")))
            for node in restored_anchor_connected_subgraph.nodes
            if node.get("node_id") is not None
        }
        for anchor in selected_anchors:
            label_by_graph_node.setdefault(anchor.node_id, anchor.display_text)
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
            node = _parse_semantic_node(
                raw,
                valid_graph_node_ids,
                label_by_graph_node,
                selected_entity_texts,
                warnings,
            )
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
        edges = _orient_edges_for_inference(
            original_question=original_question,
            nodes=nodes,
            edges=edges,
            primary_operator=primary_operator,
            warnings=warnings,
        )

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
            else:
                _materialize_operator_node(nodes, edges, primary_operator)

        status = str(payload.get("status", "ok")).strip() or "ok"
        return SemanticASTResult(
            status=status,
            primary_operator=primary_operator,
            nodes=nodes,
            edges=edges,
            warnings=[],
            raw_payload=payload,
        )
