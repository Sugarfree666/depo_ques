from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from models import ASTResult, AtomicSubquestion, ExtractionResult
from prompts import ONE_HOP_SUBQUESTION_SYSTEM, build_one_hop_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


@dataclass
class _GenerationState:
    counter: int = 1

    def next_serial_var(self) -> str:
        value = f"X{self.counter}"
        self.counter += 1
        return value


class SubquestionGenerator:
    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(
        self,
        original_question: str,
        ast: ASTResult,
        extraction: ExtractionResult,
    ) -> list[AtomicSubquestion]:
        operator_node = self._first_operator(ast)
        if operator_node:
            return self._generate_operator(original_question, ast, extraction, operator_node)
        return self._generate_general(original_question, ast, extraction)

    def _generate_general(
        self,
        original_question: str,
        ast: ASTResult,
        extraction: ExtractionResult,
    ) -> list[AtomicSubquestion]:
        graph = self._anchor_only_graph(ast.graph)
        starts = [node.placeholder for node in extraction.entities if node.placeholder in graph]
        if not starts:
            starts = [node for node in graph.nodes if graph.degree(node) <= 1] or list(graph.nodes)
        starts = sorted(starts, key=lambda node: graph.nodes[node].get("order", 10**9))

        state = _GenerationState()
        questions: list[AtomicSubquestion] = []
        visited_edges: set[frozenset[str]] = set()

        for start in starts:
            self._walk_general(
                original_question=original_question,
                ast=ast,
                graph=graph,
                current=start,
                parent=None,
                current_display=ast.display_label(start),
                current_original=ast.display_label(start),
                state=state,
                questions=questions,
                visited_edges=visited_edges,
            )

        return questions

    def _walk_general(
        self,
        original_question: str,
        ast: ASTResult,
        graph: nx.Graph,
        current: str,
        parent: str | None,
        current_display: str,
        current_original: str,
        state: _GenerationState,
        questions: list[AtomicSubquestion],
        visited_edges: set[frozenset[str]],
    ) -> None:
        neighbors = sorted(graph.neighbors(current), key=lambda node: graph.nodes[node].get("order", 10**9))
        for neighbor in neighbors:
            if neighbor == parent:
                continue
            edge_key = frozenset({current, neighbor})
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            edge_hint = _edge_hint(graph, current, neighbor)
            answer_var = state.next_serial_var()
            question_text = self._one_hop_question(
                original_question=original_question,
                source_display=current_display,
                target_display=ast.display_label(neighbor),
                source_original=current_original,
                target_original=ast.display_label(neighbor),
                answer_variable=answer_var,
                edge_hint=edge_hint,
            )
            questions.append(
                AtomicSubquestion(
                    index=len(questions) + 1,
                    question=question_text,
                    answer_variable=answer_var,
                    source_node=current,
                    target_node=neighbor,
                )
            )
            self._walk_general(
                original_question=original_question,
                ast=ast,
                graph=graph,
                current=neighbor,
                parent=current,
                current_display=answer_var,
                current_original=ast.display_label(neighbor),
                state=state,
                questions=questions,
                visited_edges=visited_edges,
            )

    def _generate_operator(
        self,
        original_question: str,
        ast: ASTResult,
        extraction: ExtractionResult,
        operator_node: str,
    ) -> list[AtomicSubquestion]:
        graph = self._anchor_only_graph(ast.graph)
        attach_nodes = [node for node in ast.graph.neighbors(operator_node) if node in graph]
        if not attach_nodes:
            attach_nodes = [self._choose_compare_target(graph, extraction)]
        target = attach_nodes[0]

        entities = [node.placeholder for node in extraction.entities if node.placeholder in graph]
        if not entities:
            entities = [node for node in graph.nodes if graph.degree(node) <= 1 and node != target]
        entities = sorted(entities, key=lambda node: graph.nodes[node].get("order", 10**9))

        operator_name = str(ast.graph.nodes[operator_node].get("text", operator_node))
        use_direct_implicit_attribute = (
            operator_name.startswith("COMPARE")
            and _is_implicit_type_variable(target, extraction)
        )
        questions: list[AtomicSubquestion] = []
        final_vars: list[str] = []
        used_edges: set[tuple[str, str, int]] = set()

        for branch_index, entity in enumerate(entities, start=1):
            if use_direct_implicit_attribute:
                path = [entity, target]
            else:
                try:
                    path = nx.shortest_path(graph, entity, target)
                except nx.NetworkXNoPath:
                    continue
            current_display = ast.display_label(entity)
            current_original = ast.display_label(entity)
            branch_final = None
            for step_index, (source, target_node) in enumerate(zip(path, path[1:]), start=1):
                edge_identity = (source, target_node, branch_index)
                if edge_identity in used_edges:
                    continue
                used_edges.add(edge_identity)
                edge_hint = _edge_hint(graph, source, target_node)
                if step_index == 1:
                    answer_var = f"X{branch_index}"
                elif target_node == target:
                    answer_var = f"X{branch_index}_{_slug(ast.display_label(target_node))}"
                else:
                    answer_var = f"X{branch_index}_{step_index}"
                question_text = self._one_hop_question(
                    original_question=original_question,
                    source_display=current_display,
                    target_display=ast.display_label(target_node),
                    source_original=current_original,
                    target_original=ast.display_label(target_node),
                    answer_variable=answer_var,
                    edge_hint=edge_hint,
                )
                questions.append(
                    AtomicSubquestion(
                        index=len(questions) + 1,
                        question=question_text,
                        answer_variable=answer_var,
                        source_node=source,
                        target_node=target_node,
                    )
                )
                current_display = answer_var
                current_original = ast.display_label(target_node)
                branch_final = answer_var
            if branch_final:
                final_vars.append(branch_final)

        min_vars = 2 if operator_name.startswith("COMPARE") or operator_name in {"INTERSECTION", "UNION", "DIFFERENCE"} else 1
        if len(final_vars) >= min_vars:
            compare_question = _operator_question(operator_name, final_vars)
            questions.append(
                AtomicSubquestion(
                    index=len(questions) + 1,
                    question=compare_question,
                    answer_variable=None,
                    operator=operator_name,
                )
            )
        return questions

    def _one_hop_question(
        self,
        original_question: str,
        source_display: str,
        target_display: str,
        source_original: str,
        target_original: str,
        answer_variable: str,
        edge_hint: str | None,
    ) -> str:
        payload = self.llm_client.chat_json(
            ONE_HOP_SUBQUESTION_SYSTEM,
            build_one_hop_prompt(
                original_question=original_question,
                source_display=source_display,
                target_display=target_display,
                source_original=source_original,
                target_original=target_original,
                answer_variable=answer_variable,
                edge_hint=edge_hint,
            ),
        )
        question = str(payload.get("question", "")).strip()
        if not question:
            raise RuntimeError("LLM returned an empty one-hop subquestion.")
        return _enforce_source_variable_binding(question, source_display, source_original)

    @staticmethod
    def _anchor_only_graph(graph: nx.Graph) -> nx.Graph:
        result = graph.copy()
        operator_nodes = [
            node for node, attrs in result.nodes(data=True) if attrs.get("kind") == "operator"
        ]
        result.remove_nodes_from(operator_nodes)
        return result

    @staticmethod
    def _first_operator(ast: ASTResult) -> str | None:
        for node, attrs in ast.graph.nodes(data=True):
            if attrs.get("kind") == "operator":
                return node
        return None

    @staticmethod
    def _choose_compare_target(graph: nx.Graph, extraction: ExtractionResult) -> str:
        type_nodes = [node.placeholder for node in extraction.type_variables if node.placeholder in graph]
        if type_nodes:
            return max(type_nodes, key=lambda node: graph.nodes[node].get("order", 0))
        return max(graph.nodes, key=lambda node: graph.degree(node))


def _edge_hint(graph: nx.Graph, source: str, target: str) -> str | None:
    if not graph.has_edge(source, target):
        return None
    attrs = graph.edges[source, target]
    relations = attrs.get("relations") or []
    path_words = attrs.get("path_words") or []
    pieces = []
    if relations:
        pieces.append("relations=" + "/".join(str(item) for item in relations if item))
    if path_words:
        pieces.append("dependency_path=" + " -> ".join(str(item) for item in path_words))
    return "; ".join(pieces) if pieces else None


def _slug(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    if not words:
        return "value"
    return "_".join(words[-2:]) if len(words) > 2 else "_".join(words)


def _enforce_source_variable_binding(question: str, source_display: str, source_original: str) -> str:
    if not _is_answer_variable(source_display) or _contains_variable(question, source_display):
        return question

    fixed = _replace_source_text(question, source_original, source_display)
    if fixed != question:
        return fixed
    return f"For {source_display}, {question[:1].lower()}{question[1:]}" if question else question


def _is_answer_variable(value: str) -> bool:
    return bool(re.fullmatch(r"X\d+(?:_[A-Za-z0-9_]+)?", value.strip()))


def _contains_variable(question: str, variable: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])", question))


def _replace_source_text(question: str, source_original: str, variable: str) -> str:
    source_words = re.findall(r"[A-Za-z0-9]+", source_original)
    if not source_words:
        return question
    escaped = r"\s+".join(re.escape(word) for word in source_words)
    pattern = re.compile(rf"\b(?:the|a|an)?\s*{escaped}\b", flags=re.IGNORECASE)
    return pattern.sub(variable, question, count=1)


def _is_implicit_type_variable(placeholder: str, extraction: ExtractionResult) -> bool:
    for node in extraction.type_variables:
        if node.placeholder == placeholder:
            return node.occurrence == 0
    return False


def _operator_question(operator: str, variables: list[str]) -> str:
    if operator == "COMPARE_DIFF":
        return f"Are {' and '.join(variables)} different?"
    if operator == "COMPARE_SAME":
        return f"Are {' and '.join(variables)} the same?"
    if operator == "INTERSECTION":
        return f"What values are common to {' and '.join(variables)}?"
    if operator == "UNION":
        return f"What values are in either {' or '.join(variables)}?"
    if operator == "DIFFERENCE":
        return f"What values are in {variables[0]} but not in {variables[1]}?" if len(variables) >= 2 else f"What is the difference for {', '.join(variables)}?"
    if operator == "COMPARE_GREATER":
        return f"Which is greater, {' or '.join(variables)}?"
    if operator == "COMPARE_LESS":
        return f"Which is less, {' or '.join(variables)}?"
    if operator == "ARGMAX":
        return f"Which has the maximum value among {', '.join(variables)}?"
    if operator == "ARGMIN":
        return f"Which has the minimum value among {', '.join(variables)}?"
    if operator == "LOGICAL_OR":
        return f"Does either {' or '.join(variables)} satisfy the condition?"
    if operator == "LOGICAL_AND":
        return f"Do {' and '.join(variables)} all satisfy the condition?"
    return f"Apply {operator} to {', '.join(variables)}."
