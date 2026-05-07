from __future__ import annotations

import networkx as nx

from hypergraph_rag.clients import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    EXTRACTION_SYSTEM_PROMPT,
    LLMClient,
    OpenAICompatibleClient,
)
from hypergraph_rag.execution import build_execution_plan
from hypergraph_rag.models import (
    BranchHint,
    DecompositionResult,
    DependencyParse,
    ExtractedQueryUnits,
    OperatorUnit,
    QueryUnitNode,
    RelationUnit,
)
from hypergraph_rag.parsing import DEFAULT_SPACY_MODEL, build_dependency_tree
from hypergraph_rag.query_ast import construct_query_ast


def extract_query_units(
    question: str,
    client: LLMClient | None = None,
) -> ExtractedQueryUnits:
    llm_client = client or OpenAICompatibleClient()
    response = llm_client.complete_json(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=question,
        purpose="extract_query_units",
    )
    return _parse_extracted_query_units(response)


def construct_query_ast_from_units(
    question: str,
    extracted_units: ExtractedQueryUnits,
    dependency_parse: DependencyParse,
) -> nx.DiGraph:
    return construct_query_ast(question, extracted_units, dependency_parse)


class QueryDecomposer:
    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        spacy_model: str = DEFAULT_SPACY_MODEL,
    ) -> None:
        self.client = client or OpenAICompatibleClient()
        self.spacy_model = spacy_model

    def decompose_question(
        self,
        question: str,
        *,
        question_id: str | None = None,
    ) -> tuple[DecompositionResult, nx.DiGraph]:
        extracted_units = extract_query_units(question, client=self.client)
        dependency_parse = build_dependency_tree(question, model_name=self.spacy_model)
        graph = construct_query_ast(question, extracted_units, dependency_parse)
        execution_steps = build_execution_plan(
            graph,
            question,
            client=self.client,
        )
        result = DecompositionResult(
            question=question,
            question_id=question_id,
            extracted_units=extracted_units,
            dependency_parse=dependency_parse,
            execution_steps=execution_steps,
            raw_extraction_output=extracted_units.raw_response,
            graph_metadata={"answer_node_ids": list(graph.graph.get("answer_node_ids", []))},
        )
        return result, graph


def build_client(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = 90,
    retries: int = 2,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        retries=retries,
    )


def _parse_extracted_query_units(payload: dict) -> ExtractedQueryUnits:
    entities = [
        QueryUnitNode(
            id=_sanitize_id(str(item["id"]), "entity"),
            label=str(item["label"]).strip(),
            semantic_type=str(item.get("type", "")).strip(),
            source_span=str(item.get("span", "")).strip(),
            kind="entity",
            metadata=_metadata_without_keys(item, {"id", "label", "type", "span"}),
        )
        for item in payload.get("entities", [])
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    ]

    type_variables = [
        QueryUnitNode(
            id=_sanitize_id(str(item["id"]), "x_variable"),
            label=str(item["label"]).strip(),
            semantic_type=str(item.get("type", "")).strip(),
            source_span=str(item.get("span", "")).strip(),
            kind="type_variable",
            is_answer=bool(item.get("is_answer", False)),
            metadata=_metadata_without_keys(
                item,
                {"id", "label", "type", "span", "is_answer"},
            ),
        )
        for item in payload.get("type_variables", [])
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    ]

    node_ids = {item.id for item in entities + type_variables}

    relations = [
        RelationUnit(
            id=_sanitize_id(str(item.get("id", "")), "relation"),
            source_id=_sanitize_id(str(item["source_id"]), "source"),
            target_id=_sanitize_id(str(item["target_id"]), "target"),
            relation=_sanitize_id(str(item["relation"]), "relation"),
            surface=str(item.get("surface", "")).strip(),
            confidence=_to_float(item.get("confidence", 0.0)),
            order_hint=_to_int_or_none(item.get("order_hint")),
            metadata=_metadata_without_keys(
                item,
                {
                    "id",
                    "source_id",
                    "target_id",
                    "relation",
                    "surface",
                    "confidence",
                    "order_hint",
                },
            ),
        )
        for item in payload.get("relations", [])
        if isinstance(item, dict)
        and str(item.get("source_id", "")).strip()
        and str(item.get("target_id", "")).strip()
        and str(item.get("relation", "")).strip()
    ]

    operators = [
        OperatorUnit(
            id=_sanitize_id(str(item.get("id", "")), "operator"),
            operator=_sanitize_id(str(item.get("operator", "")), "operator"),
            input_ids=[
                _sanitize_id(str(node_id), "node")
                for node_id in item.get("input_ids", [])
                if str(node_id).strip()
            ],
            output_id=_sanitize_id(str(item.get("output_id", "")), "x_result")
            if str(item.get("output_id", "")).strip()
            else "",
            output_label=str(item.get("output_label", "")).strip(),
            output_type=str(item.get("output_type", "Boolean")).strip() or "Boolean",
            description=str(item.get("description", "")).strip(),
            metadata=_metadata_without_keys(
                item,
                {
                    "id",
                    "operator",
                    "input_ids",
                    "output_id",
                    "output_label",
                    "output_type",
                    "description",
                },
            ),
        )
        for item in payload.get("operators", [])
        if isinstance(item, dict) and str(item.get("operator", "")).strip()
    ]

    branch_hints = [
        BranchHint(
            kind=str(item.get("kind", "")).strip() or "hint",
            node_ids=[
                _sanitize_id(str(node_id), "node")
                for node_id in item.get("node_ids", [])
                if str(node_id).strip()
            ],
            description=str(item.get("description", "")).strip(),
        )
        for item in payload.get("branch_hints", [])
        if isinstance(item, dict)
    ]

    if not relations:
        raise ValueError("LLM extraction did not return any relation units.")

    for relation in relations:
        if relation.source_id not in node_ids:
            raise ValueError(f"Relation source node was not defined: {relation.source_id}")
        if relation.target_id not in node_ids:
            raise ValueError(f"Relation target node was not defined: {relation.target_id}")

    for operator in operators:
        for input_id in operator.input_ids:
            if input_id not in node_ids:
                raise ValueError(f"Operator input node was not defined: {input_id}")

    return ExtractedQueryUnits(
        entities=entities,
        type_variables=type_variables,
        relations=relations,
        operators=operators,
        branch_hints=branch_hints,
        raw_response=payload,
    )


def _sanitize_id(value: str, default_prefix: str) -> str:
    import re

    candidate = value.strip()
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_").lower()
    if not candidate:
        candidate = default_prefix
    if not re.match(r"^[A-Za-z_]", candidate):
        candidate = f"{default_prefix}_{candidate}"
    return candidate


def _metadata_without_keys(item: dict, keys: set[str]) -> dict:
    return {key: value for key, value in item.items() if key not in keys}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int_or_none(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
