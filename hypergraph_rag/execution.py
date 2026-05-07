from __future__ import annotations

from collections import defaultdict
from typing import Any
import json

import networkx as nx

from hypergraph_rag.clients import (
    ATOMIC_EDGE_SYSTEM_PROMPT,
    LLMClient,
    OpenAICompatibleClient,
)
from hypergraph_rag.models import ExecutionStep
from hypergraph_rag.query_ast import iter_operator_node_ids, iter_relation_edges


def generate_atomic_subquestions(
    graph: nx.DiGraph,
    original_question: str,
    client: LLMClient | None = None,
) -> list[ExecutionStep]:
    llm_client = client or OpenAICompatibleClient()
    relation_edges = iter_relation_edges(graph)

    step_specs: dict[str, ExecutionStep] = {}
    dependency_map: dict[str, list[str]] = {}
    level_cache: dict[str, int] = {}
    sort_keys: dict[str, tuple[int, int, str]] = {}
    producer_by_node: dict[str, str] = {}

    for edge in relation_edges:
        source_node = _node_payload(graph, edge["source_id"])
        target_node = _node_payload(graph, edge["target_id"])
        prompt_payload = {
            "original_question": original_question,
            "source_node": source_node,
            "target_node": target_node,
            "relation": edge["relation"],
            "edge_id": edge["id"],
        }
        response = llm_client.complete_json(
            system_prompt=ATOMIC_EDGE_SYSTEM_PROMPT,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            purpose="atomic_subquestion",
        )
        question = _read_atomic_question(response, edge["id"])
        temp_id = edge["id"]
        dependencies = []
        source_producer = producer_by_node.get(edge["source_id"])
        if source_producer:
            dependencies.append(source_producer)

        step_specs[temp_id] = ExecutionStep(
            step_id=temp_id,
            kind="atomic_question",
            source_node_ids=[edge["source_id"]],
            target_node_ids=[edge["target_id"]],
            relation_or_operator=edge["relation"],
            natural_language_question=question,
            output_variable=edge["target_id"],
            dependencies=dependencies,
            execution_level=0,
            metadata={"edge_id": edge["id"]},
        )
        dependency_map[temp_id] = dependencies
        sort_keys[temp_id] = (0, int(edge.get("order", 0)), temp_id)
        producer_by_node[edge["target_id"]] = temp_id

    for operator_index, operator_node_id in enumerate(iter_operator_node_ids(graph), start=1):
        node = graph.nodes[operator_node_id]
        operator_name = str(node["metadata"].get("operator", node["label"]))
        input_ids = list(node["metadata"].get("input_ids", []))
        output_id = str(node["metadata"].get("output_id", ""))
        dependencies = _unique_preserve_order(
            producer_by_node[input_id]
            for input_id in input_ids
            if input_id in producer_by_node
        )
        step_specs[operator_node_id] = ExecutionStep(
            step_id=operator_node_id,
            kind="logical_operation",
            source_node_ids=input_ids,
            target_node_ids=[output_id] if output_id else [],
            relation_or_operator=operator_name,
            natural_language_question=_render_logical_operation(operator_name, input_ids),
            output_variable=output_id,
            dependencies=dependencies,
            execution_level=0,
            metadata={"operator_node_id": operator_node_id},
        )
        dependency_map[operator_node_id] = dependencies
        operator_order = int(node.get("metadata", {}).get("operator_order", operator_index))
        sort_keys[operator_node_id] = (1, operator_order, operator_node_id)
        if output_id:
            producer_by_node[output_id] = operator_node_id

    for temp_id in step_specs:
        step_specs[temp_id].execution_level = _resolve_execution_level(
            temp_id,
            dependency_map,
            level_cache,
        )

    ordered_temp_ids = sorted(
        step_specs,
        key=lambda item: (
            step_specs[item].execution_level,
            sort_keys[item][0],
            sort_keys[item][1],
            sort_keys[item][2],
        ),
    )

    final_id_map = {
        temp_id: f"step_{index:03d}"
        for index, temp_id in enumerate(ordered_temp_ids, start=1)
    }

    final_steps: list[ExecutionStep] = []
    for temp_id in ordered_temp_ids:
        step = step_specs[temp_id]
        final_steps.append(
            ExecutionStep(
                step_id=final_id_map[temp_id],
                kind=step.kind,
                source_node_ids=list(step.source_node_ids),
                target_node_ids=list(step.target_node_ids),
                relation_or_operator=step.relation_or_operator,
                natural_language_question=step.natural_language_question,
                output_variable=step.output_variable,
                dependencies=[final_id_map[item] for item in step.dependencies],
                execution_level=step.execution_level,
                metadata=dict(step.metadata),
            )
        )

    return final_steps


def build_execution_plan(
    graph: nx.DiGraph,
    original_question: str,
    client: LLMClient | None = None,
) -> list[ExecutionStep]:
    return generate_atomic_subquestions(graph, original_question, client=client)


def group_steps_by_level(steps: list[ExecutionStep]) -> dict[int, list[ExecutionStep]]:
    grouped: dict[int, list[ExecutionStep]] = defaultdict(list)
    for step in steps:
        grouped[step.execution_level].append(step)
    return dict(sorted(grouped.items()))


def _node_payload(graph: nx.DiGraph, node_id: str) -> dict[str, Any]:
    node = graph.nodes[node_id]
    return {
        "id": node_id,
        "label": node["label"],
        "node_kind": node["node_kind"],
        "semantic_type": node.get("semantic_type", ""),
        "source_span": node.get("source_span", ""),
    }


def _read_atomic_question(response: dict[str, Any], edge_id: str) -> str:
    for key in ("question", "atomic_subquestion"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Atomic question response for edge {edge_id} did not contain a question.")


def _render_logical_operation(operator_name: str, input_ids: list[str]) -> str:
    if operator_name == "compare_eq" and len(input_ids) == 2:
        return f"Compare whether {input_ids[0]} equals {input_ids[1]}."
    if operator_name == "intersection":
        return f"Compute the intersection of {', '.join(input_ids)}."
    if operator_name == "count" and input_ids:
        return f"Count the items in {input_ids[0]}."
    if operator_name == "argmax" and input_ids:
        return f"Select the maximum-scoring item from {input_ids[0]}."
    if operator_name == "argmin" and input_ids:
        return f"Select the minimum-scoring item from {input_ids[0]}."
    return f"Apply {operator_name} to {', '.join(input_ids)}."


def _resolve_execution_level(
    step_id: str,
    dependency_map: dict[str, list[str]],
    level_cache: dict[str, int],
) -> int:
    if step_id in level_cache:
        return level_cache[step_id]
    dependencies = dependency_map.get(step_id, [])
    if not dependencies:
        level_cache[step_id] = 0
        return 0
    level = max(
        _resolve_execution_level(dependency_id, dependency_map, level_cache)
        for dependency_id in dependencies
    ) + 1
    level_cache[step_id] = level
    return level


def _unique_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
