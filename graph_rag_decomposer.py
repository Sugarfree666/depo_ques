#!/usr/bin/env python3
"""
Query-AST decomposer for entity/type-variable based Graph-RAG research.

The pipeline is intentionally close to the research idea:
1. Ask gpt-4o-mini to identify entities, type variables, relation words, and a
   compiler-style Query AST.
2. Validate the AST locally.
3. Traverse every one-hop graph edge.
4. Ask gpt-4o-mini to turn each one-hop edge into an atomic sub-question.

The script uses only the Python standard library and an OpenAI-compatible
chat/completions endpoint, so it works with api_key + base_url without requiring
the openai package.

Examples:
    python graph_rag_decomposer.py --index 0
    python graph_rag_decomposer.py --question "Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?"
    python graph_rag_decomposer.py --question-file questions.json --index 3 --format json
    python graph_rag_decomposer.py --mock --question "Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_QUESTION_FILE = "questions.json"


AST_SYSTEM_PROMPT = """
You are a compiler-style semantic parser for complex question decomposition in Graph-RAG.
Your job is to convert one complex natural-language question into a Query AST over entities and type variables.

Return JSON only. Do not use markdown.

Required JSON schema:
{
  "question_type": "multi_hop|parallel|hybrid",
  "root_nodes": ["node_id"],
  "answer_nodes": ["node_id"],
  "nodes": [
    {
      "id": "stable_node_id",
      "label": "surface text or concise variable name",
      "kind": "entity|type_variable|role_variable|value_variable",
      "semantic_type": "optional type, e.g. Person, University, City, Nationality, AI_Company",
      "source_span": "exact or near-exact phrase from the question"
    }
  ],
  "edges": [
    {
      "id": "e1",
      "order": 1,
      "source": "node_id",
      "target": "node_id",
      "relation": "short_relation_name",
      "surface": "relation phrase in the original question",
      "direction": "why the source -> target direction is correct",
      "confidence": 0.0
    }
  ],
  "operations": [
    {
      "id": "op1",
      "operator": "equals|not_equals|compare|count|argmax|argmin|intersection|union|and|or",
      "left": "node_id",
      "right": "node_id",
      "output": "optional output node id",
      "description": "short description"
    }
  ],
  "notes": "short optional note"
}

Rules:
- Extract the minimal entities/type variables needed to answer the question.
- Use concrete named things as kind=entity.
- Use generic answer slots such as company, CEO, university, city, director, nationality as variables.
- Variable ids should start with x_, for example x_company, x_ceo, x_university, x_city.
- Every edge must be one atomic one-hop relation. Never merge two relations into one edge.
- Preserve intermediate variables whenever an answer feeds a later hop.
- For multi-hop questions, build a chain or small tree from grounded entity to final answer variable.
- For parallel questions, build separate branches and put comparison/equality in operations.
- Orient retrieval edges from known/current subject to the next unknown target.
- Put all compare/same/different logic in operations, not retrieval edges.
- Keep relation names concise and machine-readable, such as developed_by, ceo_of, graduated_from, located_in, directed_by, has_nationality.
- The output language of labels can follow the question, but ids and relation names should be ASCII.

Two examples:
Question: Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?
AST edges should be equivalent to:
AlphaGo --developed_by--> x_company:AI_Company --ceo_of--> x_ceo:Person --graduated_from--> x_university:University --located_in--> x_city:City

Question: Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?
AST edges should be equivalent to two branches:
Ten9Eight --directed_by--> x_director_1 --has_nationality--> x_nationality_1
Sabotage --directed_by--> x_director_2 --has_nationality--> x_nationality_2
operation equals(x_nationality_1, x_nationality_2)
""".strip()


ATOMIC_SYSTEM_PROMPT = """
You generate atomic one-hop sub-questions from a validated Query AST.

Return JSON only. Do not use markdown.

Required JSON schema:
{
  "atomic_questions": [
    {
      "edge_id": "e1",
      "question": "one natural-language atomic question",
      "input_binding": "source variable id if the source is a variable, else empty string",
      "output_binding": "target variable id if the target is a variable, else empty string",
      "depends_on": ["edge_id"]
    }
  ],
  "operation_questions": [
    {
      "operation_id": "op1",
      "question": "optional final non-retrieval operation question",
      "depends_on": ["edge_id"]
    }
  ]
}

Rules:
- Generate exactly one atomic question for each retrieval edge.
- Use the same language as the original question.
- Ask only for the target node of that edge.
- Do not include later hops or comparison logic in a retrieval atomic question.
- If the source node is a concrete entity, use its label literally.
- If the source node is a variable, use its id literally, such as x_company or x_director_1.
- If the target node is a variable, output_binding must be that target id.
- For parallel comparisons, generate retrieval questions for every branch, then put equality/same/different checks in operation_questions.
- Keep every atomic question short and directly answerable by one relation.
""".strip()


@dataclass
class ASTNode:
    id: str
    label: str
    kind: str
    semantic_type: str = ""
    source_span: str = ""


@dataclass
class ASTEdge:
    id: str
    order: int
    source: str
    target: str
    relation: str
    surface: str = ""
    direction: str = ""
    confidence: float = 0.0


@dataclass
class ASTOperation:
    id: str
    operator: str
    left: str = ""
    right: str = ""
    output: str = ""
    description: str = ""


@dataclass
class QueryAST:
    question: str
    question_type: str
    root_nodes: List[str]
    answer_nodes: List[str]
    nodes: List[ASTNode]
    edges: List[ASTEdge]
    operations: List[ASTOperation] = field(default_factory=list)
    notes: str = ""


@dataclass
class AtomicQuestion:
    edge_id: str
    question: str
    input_binding: str = ""
    output_binding: str = ""
    depends_on: List[str] = field(default_factory=list)


@dataclass
class OperationQuestion:
    operation_id: str
    question: str
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Decomposition:
    question: str
    ast: QueryAST
    atomic_questions: List[AtomicQuestion]
    operation_questions: List[OperationQuestion]
    model: str


class DecompositionError(RuntimeError):
    pass


def sanitize_id(value: str, prefix: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if not value:
        value = prefix
    if not re.match(r"^[A-Za-z_]", value):
        value = f"{prefix}_{value}"
    return value


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def is_placeholder_ref(value: str) -> bool:
    normalized = sanitize_id(value, "placeholder")
    return normalized in {
        "",
        "none",
        "null",
        "nil",
        "n_a",
        "na",
        "optional",
        "optional_output",
        "optional_output_node",
        "optional_output_node_id",
        "output_node_id",
        "optional_node_id",
        "not_applicable",
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise DecompositionError("Model response did not contain a JSON object.")
        depth = 0
        in_string = False
        escape = False
        end = -1
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end < 0:
            raise DecompositionError("Model response contained incomplete JSON.")
        parsed = json.loads(text[start:end])

    if not isinstance(parsed, dict):
        raise DecompositionError("Model response JSON must be an object.")
    return parsed


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ChatCompletionsJSONClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 90,
        use_response_format: bool = True,
        retries: int = 2,
    ) -> None:
        if not api_key:
            raise DecompositionError(
                "Missing API key. Pass --api-key or set OPENAI_API_KEY."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.use_response_format = use_response_format
        self.retries = retries

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.path in {"", "/"}:
            return f"{self.base_url}/v1/chat/completions"
        return f"{self.base_url}/chat/completions"

    def complete_json(self, system_prompt: str, user_prompt: str, purpose: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}

        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                return self._post_json(payload)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = DecompositionError(
                    f"{purpose} request failed with HTTP {exc.code}: {body}"
                )
                if (
                    exc.code == 400
                    and "response_format" in body
                    and payload.pop("response_format", None) is not None
                ):
                    continue
            except (urllib.error.URLError, TimeoutError, DecompositionError) as exc:
                last_error = exc

            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))

        raise DecompositionError(str(last_error))

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DecompositionError(
                f"Unexpected chat/completions response shape: {response_payload!r}"
            ) from exc
        return extract_json_object(content)


class MockJSONClient:
    """Deterministic local client for smoke tests without an API call."""

    model = "mock-gpt-4o-mini"

    def complete_json(self, system_prompt: str, user_prompt: str, purpose: str) -> Dict[str, Any]:
        if purpose == "ast":
            return self._mock_ast(user_prompt)
        if purpose == "atomic":
            return self._mock_atomic(user_prompt)
        raise DecompositionError(f"Unknown mock purpose: {purpose}")

    @staticmethod
    def _mock_ast(question: str) -> Dict[str, Any]:
        if "AlphaGo" in question:
            return {
                "question_type": "multi_hop",
                "root_nodes": ["alphago"],
                "answer_nodes": ["x_university", "x_city"],
                "nodes": [
                    {
                        "id": "alphago",
                        "label": "AlphaGo",
                        "kind": "entity",
                        "semantic_type": "AI_System",
                        "source_span": "AlphaGo",
                    },
                    {
                        "id": "x_company",
                        "label": "artificial intelligence company",
                        "kind": "type_variable",
                        "semantic_type": "AI_Company",
                        "source_span": "artificial intelligence company",
                    },
                    {
                        "id": "x_ceo",
                        "label": "CEO",
                        "kind": "role_variable",
                        "semantic_type": "Person",
                        "source_span": "CEO",
                    },
                    {
                        "id": "x_university",
                        "label": "university",
                        "kind": "type_variable",
                        "semantic_type": "University",
                        "source_span": "university",
                    },
                    {
                        "id": "x_city",
                        "label": "city",
                        "kind": "type_variable",
                        "semantic_type": "City",
                        "source_span": "city",
                    },
                ],
                "edges": [
                    {
                        "id": "e1",
                        "order": 1,
                        "source": "alphago",
                        "target": "x_company",
                        "relation": "developed_by",
                        "surface": "company that developed AlphaGo",
                        "direction": "start from AlphaGo and retrieve developer company",
                        "confidence": 0.99,
                    },
                    {
                        "id": "e2",
                        "order": 2,
                        "source": "x_company",
                        "target": "x_ceo",
                        "relation": "ceo_of",
                        "surface": "CEO of the company",
                        "direction": "retrieve CEO from company",
                        "confidence": 0.99,
                    },
                    {
                        "id": "e3",
                        "order": 3,
                        "source": "x_ceo",
                        "target": "x_university",
                        "relation": "graduated_from",
                        "surface": "CEO graduate from university",
                        "direction": "retrieve university from CEO",
                        "confidence": 0.99,
                    },
                    {
                        "id": "e4",
                        "order": 4,
                        "source": "x_university",
                        "target": "x_city",
                        "relation": "located_in",
                        "surface": "university located in city",
                        "direction": "retrieve city from university",
                        "confidence": 0.99,
                    },
                ],
                "operations": [],
                "notes": "mock ast",
            }

        if "Ten9Eight" in question or "Sabotage" in question:
            return {
                "question_type": "parallel",
                "root_nodes": ["ten9eight", "sabotage_1936"],
                "answer_nodes": ["x_nationality_1", "x_nationality_2"],
                "nodes": [
                    {
                        "id": "ten9eight",
                        "label": "Ten9Eight: Shoot For The Moon",
                        "kind": "entity",
                        "semantic_type": "Film",
                        "source_span": "Ten9Eight: Shoot For The Moon",
                    },
                    {
                        "id": "x_director_1",
                        "label": "director",
                        "kind": "role_variable",
                        "semantic_type": "Person",
                        "source_span": "director of film Ten9Eight: Shoot For The Moon",
                    },
                    {
                        "id": "x_nationality_1",
                        "label": "nationality",
                        "kind": "value_variable",
                        "semantic_type": "Nationality",
                        "source_span": "nationality",
                    },
                    {
                        "id": "sabotage_1936",
                        "label": "Sabotage (1936 Film)",
                        "kind": "entity",
                        "semantic_type": "Film",
                        "source_span": "Sabotage (1936 Film)",
                    },
                    {
                        "id": "x_director_2",
                        "label": "director",
                        "kind": "role_variable",
                        "semantic_type": "Person",
                        "source_span": "director of film Sabotage (1936 Film)",
                    },
                    {
                        "id": "x_nationality_2",
                        "label": "nationality",
                        "kind": "value_variable",
                        "semantic_type": "Nationality",
                        "source_span": "nationality",
                    },
                ],
                "edges": [
                    {
                        "id": "e1",
                        "order": 1,
                        "source": "ten9eight",
                        "target": "x_director_1",
                        "relation": "directed_by",
                        "surface": "director of film Ten9Eight",
                        "direction": "retrieve director from film",
                        "confidence": 0.99,
                    },
                    {
                        "id": "e2",
                        "order": 2,
                        "source": "x_director_1",
                        "target": "x_nationality_1",
                        "relation": "has_nationality",
                        "surface": "director nationality",
                        "direction": "retrieve nationality from director",
                        "confidence": 0.99,
                    },
                    {
                        "id": "e3",
                        "order": 3,
                        "source": "sabotage_1936",
                        "target": "x_director_2",
                        "relation": "directed_by",
                        "surface": "director of film Sabotage",
                        "direction": "retrieve director from film",
                        "confidence": 0.99,
                    },
                    {
                        "id": "e4",
                        "order": 4,
                        "source": "x_director_2",
                        "target": "x_nationality_2",
                        "relation": "has_nationality",
                        "surface": "director nationality",
                        "direction": "retrieve nationality from director",
                        "confidence": 0.99,
                    },
                ],
                "operations": [
                    {
                        "id": "op1",
                        "operator": "equals",
                        "left": "x_nationality_1",
                        "right": "x_nationality_2",
                        "output": "",
                        "description": "compare whether the two nationalities are the same",
                    }
                ],
                "notes": "mock ast",
            }

        return {
            "question_type": "hybrid",
            "root_nodes": ["question_context"],
            "answer_nodes": ["x_answer"],
            "nodes": [
                {
                    "id": "question_context",
                    "label": "question context",
                    "kind": "entity",
                    "semantic_type": "Context",
                    "source_span": question[:80],
                },
                {
                    "id": "x_answer",
                    "label": "answer",
                    "kind": "type_variable",
                    "semantic_type": "Answer",
                    "source_span": "answer",
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "order": 1,
                    "source": "question_context",
                    "target": "x_answer",
                    "relation": "asks_for",
                    "surface": question,
                    "direction": "generic mock fallback",
                    "confidence": 0.1,
                }
            ],
            "operations": [],
            "notes": "generic mock fallback",
        }

    @staticmethod
    def _mock_atomic(user_prompt: str) -> Dict[str, Any]:
        payload = extract_json_object(user_prompt)
        nodes = {node["id"]: node for node in payload["ast"]["nodes"]}
        atomic_questions = []
        for edge in payload["ast"]["edges"]:
            source = nodes[edge["source"]]
            target = nodes[edge["target"]]
            source_text = source["id"] if source["kind"] != "entity" else source["label"]
            relation = edge["relation"]
            if relation == "developed_by":
                question = f"Which {target['label']} developed {source_text}?"
            elif relation == "ceo_of":
                question = f"Who is the CEO of {source_text}?"
            elif relation == "graduated_from":
                question = f"Which university did {source_text} graduate from?"
            elif relation == "located_in":
                question = f"Which city is {source_text} located in?"
            elif relation == "directed_by":
                question = f"Who directed {source_text}?"
            elif relation == "has_nationality":
                question = f"What is the nationality of {source_text}?"
            else:
                question = f"What {target['label']} is related to {source_text} by {relation}?"
            atomic_questions.append(
                {
                    "edge_id": edge["id"],
                    "question": question,
                    "input_binding": source["id"] if source["kind"] != "entity" else "",
                    "output_binding": target["id"] if target["kind"] != "entity" else "",
                    "depends_on": _incoming_edge_ids(payload["ast"]["edges"], edge["source"]),
                }
            )

        operation_questions = []
        for operation in payload["ast"].get("operations", []):
            operation_questions.append(
                {
                    "operation_id": operation["id"],
                    "question": f"Do {operation.get('left', '')} and {operation.get('right', '')} satisfy {operation.get('operator', 'compare')}?",
                    "depends_on": _incoming_edge_ids(
                        payload["ast"]["edges"],
                        operation.get("left", ""),
                    )
                    + _incoming_edge_ids(payload["ast"]["edges"], operation.get("right", "")),
                }
            )
        return {
            "atomic_questions": atomic_questions,
            "operation_questions": operation_questions,
        }


def _incoming_edge_ids(edges: Sequence[Dict[str, Any]], target_node: str) -> List[str]:
    return [str(edge.get("id", "")) for edge in edges if edge.get("target") == target_node]


def build_query_ast(question: str, payload: Dict[str, Any]) -> QueryAST:
    raw_nodes = as_list(payload.get("nodes"))
    raw_edges = as_list(payload.get("edges"))
    raw_operations = as_list(payload.get("operations"))

    if len(raw_nodes) < 2:
        raise DecompositionError("AST must contain at least two nodes.")
    if not raw_edges:
        raise DecompositionError("AST must contain at least one retrieval edge.")

    nodes: List[ASTNode] = []
    used_node_ids: Dict[str, int] = {}
    original_to_clean: Dict[str, str] = {}

    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            continue
        original_id = str(raw_node.get("id") or raw_node.get("label") or f"node_{index}")
        clean_id = sanitize_id(original_id, f"n{index}")
        if clean_id in used_node_ids:
            used_node_ids[clean_id] += 1
            clean_id = f"{clean_id}_{used_node_ids[clean_id]}"
        else:
            used_node_ids[clean_id] = 1
        original_to_clean[original_id] = clean_id
        nodes.append(
            ASTNode(
                id=clean_id,
                label=str(raw_node.get("label") or original_id).strip(),
                kind=normalize_kind(str(raw_node.get("kind", "type_variable"))),
                semantic_type=str(raw_node.get("semantic_type", "")).strip(),
                source_span=str(raw_node.get("source_span", "")).strip(),
            )
        )

    node_ids = {node.id for node in nodes}
    if len(node_ids) < 2:
        raise DecompositionError("AST node normalization produced fewer than two nodes.")

    edges: List[ASTEdge] = []
    used_edge_ids: Dict[str, int] = {}
    for index, raw_edge in enumerate(raw_edges, start=1):
        if not isinstance(raw_edge, dict):
            continue
        source = remap_id(str(raw_edge.get("source", "")), original_to_clean)
        target = remap_id(str(raw_edge.get("target", "")), original_to_clean)
        if source not in node_ids or target not in node_ids:
            raise DecompositionError(
                f"Edge {raw_edge!r} references unknown source or target."
            )
        if source == target:
            raise DecompositionError(f"Edge {raw_edge!r} has identical source and target.")

        edge_id = sanitize_id(str(raw_edge.get("id") or f"e{index}"), f"e{index}")
        if edge_id in used_edge_ids:
            used_edge_ids[edge_id] += 1
            edge_id = f"{edge_id}_{used_edge_ids[edge_id]}"
        else:
            used_edge_ids[edge_id] = 1

        edges.append(
            ASTEdge(
                id=edge_id,
                order=parse_int(raw_edge.get("order"), index),
                source=source,
                target=target,
                relation=sanitize_id(str(raw_edge.get("relation", f"relation_{index}")), "relation"),
                surface=str(raw_edge.get("surface", "")).strip(),
                direction=str(raw_edge.get("direction", "")).strip(),
                confidence=parse_float(raw_edge.get("confidence"), 0.0),
            )
        )

    if not edges:
        raise DecompositionError("No valid edges were found in AST.")

    operations: List[ASTOperation] = []
    for index, raw_operation in enumerate(raw_operations, start=1):
        if not isinstance(raw_operation, dict):
            continue
        operations.append(
            ASTOperation(
                id=sanitize_id(str(raw_operation.get("id") or f"op{index}"), f"op{index}"),
                operator=sanitize_id(
                    str(raw_operation.get("operator", "compare")), "compare"
                ),
                left=remap_id(str(raw_operation.get("left", "")), original_to_clean),
                right=remap_id(str(raw_operation.get("right", "")), original_to_clean),
                output=remap_id(str(raw_operation.get("output", "")), original_to_clean),
                description=str(raw_operation.get("description", "")).strip(),
            )
        )

    validate_operations(operations, node_ids)
    ordered_edges = order_edges(edges)

    return QueryAST(
        question=question,
        question_type=normalize_question_type(str(payload.get("question_type", "hybrid"))),
        root_nodes=[
            remap_id(str(item), original_to_clean)
            for item in as_list(payload.get("root_nodes"))
            if remap_id(str(item), original_to_clean) in node_ids
        ],
        answer_nodes=[
            remap_id(str(item), original_to_clean)
            for item in as_list(payload.get("answer_nodes"))
            if remap_id(str(item), original_to_clean) in node_ids
        ],
        nodes=nodes,
        edges=ordered_edges,
        operations=operations,
        notes=str(payload.get("notes", "")).strip(),
    )


def remap_id(value: str, mapping: Dict[str, str]) -> str:
    value = value.strip()
    if not value or is_placeholder_ref(value):
        return ""
    return mapping.get(value, sanitize_id(value, "id"))


def normalize_kind(value: str) -> str:
    value = value.strip().lower()
    allowed = {"entity", "type_variable", "role_variable", "value_variable"}
    if value in allowed:
        return value
    if value in {"type", "variable"}:
        return "type_variable"
    if value in {"role"}:
        return "role_variable"
    if value in {"value", "attribute"}:
        return "value_variable"
    return "type_variable"


def normalize_question_type(value: str) -> str:
    value = value.strip().lower()
    if value in {"multi_hop", "parallel", "hybrid"}:
        return value
    if value in {"multihop", "chain"}:
        return "multi_hop"
    return "hybrid"


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_operations(operations: Sequence[ASTOperation], node_ids: set[str]) -> None:
    for operation in operations:
        for field_name in ("left", "right", "output"):
            value = getattr(operation, field_name)
            if value and value not in node_ids:
                raise DecompositionError(
                    f"Operation {operation.id} references unknown {field_name}: {value}"
                )


def order_edges(edges: Sequence[ASTEdge]) -> List[ASTEdge]:
    return sorted(edges, key=lambda edge: (edge.order, edge.id))


def ast_to_payload(ast: QueryAST) -> Dict[str, Any]:
    return {
        "question": ast.question,
        "question_type": ast.question_type,
        "root_nodes": ast.root_nodes,
        "answer_nodes": ast.answer_nodes,
        "nodes": [asdict(node) for node in ast.nodes],
        "edges": [asdict(edge) for edge in ast.edges],
        "operations": [asdict(operation) for operation in ast.operations],
        "notes": ast.notes,
    }


def generate_ast(client: Any, question: str) -> QueryAST:
    user_prompt = f"Question:\n{question}"
    payload = client.complete_json(AST_SYSTEM_PROMPT, user_prompt, purpose="ast")
    return build_query_ast(question, payload)


def generate_atomic_questions(client: Any, ast: QueryAST) -> Tuple[List[AtomicQuestion], List[OperationQuestion]]:
    user_payload = {
        "original_question": ast.question,
        "ast": ast_to_payload(ast),
        "edge_order": [edge.id for edge in ast.edges],
    }
    payload = client.complete_json(
        ATOMIC_SYSTEM_PROMPT,
        json.dumps(user_payload, ensure_ascii=False, indent=2),
        purpose="atomic",
    )

    atomic_questions = parse_atomic_questions(payload, ast)
    operation_questions = parse_operation_questions(payload, ast)
    return atomic_questions, operation_questions


def parse_atomic_questions(payload: Dict[str, Any], ast: QueryAST) -> List[AtomicQuestion]:
    by_edge = {edge.id: edge for edge in ast.edges}
    source_by_edge = {edge.id: edge.source for edge in ast.edges}
    target_by_edge = {edge.id: edge.target for edge in ast.edges}
    nodes = {node.id: node for node in ast.nodes}

    raw_items = as_list(payload.get("atomic_questions"))
    parsed_by_edge: Dict[str, AtomicQuestion] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        edge_id = sanitize_id(str(raw_item.get("edge_id", "")), "edge")
        if edge_id not in by_edge:
            continue
        question = str(raw_item.get("question", "")).strip()
        if not question:
            continue
        edge = by_edge[edge_id]
        source_node = nodes[source_by_edge[edge_id]]
        target_node = nodes[target_by_edge[edge_id]]
        parsed_by_edge[edge_id] = AtomicQuestion(
            edge_id=edge_id,
            question=question,
            input_binding=str(raw_item.get("input_binding", "")).strip()
            or (source_node.id if source_node.kind != "entity" else ""),
            output_binding=str(raw_item.get("output_binding", "")).strip()
            or (target_node.id if target_node.kind != "entity" else ""),
            depends_on=[
                sanitize_id(str(item), "edge")
                for item in as_list(raw_item.get("depends_on"))
                if sanitize_id(str(item), "edge") in by_edge
            ],
        )

    ordered: List[AtomicQuestion] = []
    for edge in ast.edges:
        if edge.id in parsed_by_edge:
            ordered.append(parsed_by_edge[edge.id])
        else:
            ordered.append(fallback_atomic_question(edge, nodes, ast.question))
    return ordered


def parse_operation_questions(payload: Dict[str, Any], ast: QueryAST) -> List[OperationQuestion]:
    operation_ids = {operation.id for operation in ast.operations}
    edge_ids = {edge.id for edge in ast.edges}
    parsed: List[OperationQuestion] = []
    for raw_item in as_list(payload.get("operation_questions")):
        if not isinstance(raw_item, dict):
            continue
        operation_id = sanitize_id(str(raw_item.get("operation_id", "")), "op")
        if operation_id not in operation_ids:
            continue
        question = str(raw_item.get("question", "")).strip()
        if not question:
            continue
        parsed.append(
            OperationQuestion(
                operation_id=operation_id,
                question=question,
                depends_on=[
                    sanitize_id(str(item), "edge")
                    for item in as_list(raw_item.get("depends_on"))
                    if sanitize_id(str(item), "edge") in edge_ids
                ],
            )
        )

    existing_ids = {item.operation_id for item in parsed}
    for operation in ast.operations:
        if operation.id not in existing_ids:
            parsed.append(fallback_operation_question(operation, ast.edges))
    return parsed


def fallback_atomic_question(edge: ASTEdge, nodes: Dict[str, ASTNode], original_question: str) -> AtomicQuestion:
    source = nodes[edge.source]
    target = nodes[edge.target]
    source_text = source.label if source.kind == "entity" else source.id
    target_text = target.semantic_type or target.label
    relation = edge.relation

    if uses_chinese(original_question):
        if relation in {"ceo_of", "has_ceo"}:
            question = f"{source_text}的CEO是谁？"
        elif relation in {"graduated_from", "graduate_from"}:
            question = f"{source_text}毕业于哪所大学？"
        elif relation in {"located_in", "location"}:
            question = f"{source_text}位于哪座城市？"
        elif relation in {"developed_by", "created_by"}:
            question = f"哪个{target_text}研发了{source_text}？"
        elif relation in {"directed_by"}:
            question = f"谁导演了{source_text}？"
        elif relation in {"has_nationality", "nationality"}:
            question = f"{source_text}的国籍是什么？"
        else:
            question = f"与{source_text}通过{relation}关系相连的{target_text}是什么？"
    else:
        if relation in {"ceo_of", "has_ceo"}:
            question = f"Who is the CEO of {source_text}?"
        elif relation in {"graduated_from", "graduate_from"}:
            question = f"Which university did {source_text} graduate from?"
        elif relation in {"located_in", "location"}:
            question = f"Which city is {source_text} located in?"
        elif relation in {"developed_by", "created_by"}:
            question = f"Which {target_text} developed {source_text}?"
        elif relation in {"directed_by"}:
            question = f"Who directed {source_text}?"
        elif relation in {"has_nationality", "nationality"}:
            question = f"What is the nationality of {source_text}?"
        else:
            question = f"What {target_text} is connected to {source_text} by {relation}?"

    return AtomicQuestion(
        edge_id=edge.id,
        question=question,
        input_binding=source.id if source.kind != "entity" else "",
        output_binding=target.id if target.kind != "entity" else "",
        depends_on=[],
    )


def fallback_operation_question(operation: ASTOperation, edges: Sequence[ASTEdge]) -> OperationQuestion:
    depends_on = [
        edge.id
        for edge in edges
        if edge.target in {operation.left, operation.right, operation.output}
    ]
    return OperationQuestion(
        operation_id=operation.id,
        question=f"Check whether {operation.left} and {operation.right} satisfy {operation.operator}.",
        depends_on=depends_on,
    )


def uses_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def decompose_question(client: Any, question: str, model: str) -> Decomposition:
    ast = generate_ast(client, question)
    atomic_questions, operation_questions = generate_atomic_questions(client, ast)
    return Decomposition(
        question=question,
        ast=ast,
        atomic_questions=atomic_questions,
        operation_questions=operation_questions,
        model=model,
    )


def load_questions(path: Path) -> List[str]:
    data = read_json(path)
    if not isinstance(data, list):
        raise DecompositionError("Question file must contain a JSON array.")

    questions: List[str] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            question = item.strip()
        elif isinstance(item, dict):
            question = ""
            for key in ("question", "query", "text"):
                if key in item:
                    question = str(item[key]).strip()
                    break
        else:
            question = ""

        if not question:
            raise DecompositionError(f"Unsupported question record at index {index}: {item!r}")
        questions.append(question)
    return questions


def resolve_questions(args: argparse.Namespace) -> List[Tuple[int, str]]:
    if args.question:
        return [(-1, args.question.strip())]

    if args.stdin:
        question = sys.stdin.read().strip()
        if not question:
            raise DecompositionError("--stdin was used but stdin is empty.")
        return [(-1, question)]

    question_file = Path(args.question_file)
    if not question_file.exists():
        raise DecompositionError(
            f"Question file not found: {question_file}. Pass --question for custom input."
        )

    questions = load_questions(question_file)
    if args.all:
        indexed = list(enumerate(questions))
        if args.limit is not None:
            indexed = indexed[: args.limit]
        return indexed

    if args.index < 0 or args.index >= len(questions):
        raise DecompositionError(
            f"--index {args.index} is out of range. {question_file} contains {len(questions)} questions."
        )
    return [(args.index, questions[args.index])]


def decomposition_to_dict(item: Decomposition, source_index: int) -> Dict[str, Any]:
    return {
        "source_index": source_index,
        "question": item.question,
        "model": item.model,
        "query_ast": ast_to_payload(item.ast),
        "atomic_questions": [asdict(question) for question in item.atomic_questions],
        "operation_questions": [asdict(question) for question in item.operation_questions],
    }


def render_decomposition(item: Decomposition, source_index: int) -> str:
    lines: List[str] = []
    index_text = "custom" if source_index < 0 else str(source_index)
    node_map = {node.id: node for node in item.ast.nodes}

    lines.append(f"问题索引: {index_text}")
    lines.append(f"原问题: {item.question}")
    lines.append(f"问题类型: {item.ast.question_type}")
    lines.append("")
    lines.append("实体/类型变量节点:")
    for node in item.ast.nodes:
        type_text = f": {node.semantic_type}" if node.semantic_type else ""
        lines.append(f"  - {node.id} = {node.label} [{node.kind}{type_text}]")

    lines.append("")
    lines.append("语法树 / Query AST 一跳边:")
    for edge in item.ast.edges:
        source = node_map[edge.source]
        target = node_map[edge.target]
        source_text = format_node_for_edge(source)
        target_text = format_node_for_edge(target)
        surface = f"  surface={edge.surface}" if edge.surface else ""
        lines.append(f"  - ({edge.id}) {source_text} --{edge.relation}--> {target_text}{surface}")

    if item.ast.operations:
        lines.append("")
        lines.append("操作节点:")
        for operation in item.ast.operations:
            right = f", {operation.right}" if operation.right else ""
            lines.append(f"  - ({operation.id}) {operation.operator}({operation.left}{right})")

    lines.append("")
    lines.append("原子子问题:")
    for index, question in enumerate(item.atomic_questions, start=1):
        binding = render_binding(question.input_binding, question.output_binding)
        lines.append(f"  {index}. [{question.edge_id}{binding}] {question.question}")

    if item.operation_questions:
        lines.append("")
        lines.append("非检索操作问题:")
        for index, question in enumerate(item.operation_questions, start=1):
            deps = f" depends_on={','.join(question.depends_on)}" if question.depends_on else ""
            lines.append(f"  {index}. [{question.operation_id}{deps}] {question.question}")

    return "\n".join(lines)


def format_node_for_edge(node: ASTNode) -> str:
    if node.kind == "entity":
        return node.label
    if node.semantic_type:
        return f"{node.id}: {node.semantic_type}"
    return f"{node.id}: {node.label}"


def render_binding(input_binding: str, output_binding: str) -> str:
    parts: List[str] = []
    if input_binding:
        parts.append(f"input={input_binding}")
    if output_binding:
        parts.append(f"output={output_binding}")
    if not parts:
        return ""
    return "; " + "; ".join(parts)


def build_client(args: argparse.Namespace) -> Any:
    if args.mock:
        return MockJSONClient()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    return ChatCompletionsJSONClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        use_response_format=not args.no_response_format,
        retries=args.retries,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Query AST and one-hop atomic sub-questions for Graph-RAG."
    )
    parser.add_argument("--question", type=str, help="Custom complex question.")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read one custom question from stdin.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=DEFAULT_QUESTION_FILE,
        help=f"JSON question file. Default: {DEFAULT_QUESTION_FILE}",
    )
    parser.add_argument("--index", type=int, default=0, help="Question index in question file.")
    parser.add_argument("--all", action="store_true", help="Process all questions in the file.")
    parser.add_argument("--limit", type=int, help="Limit number of questions when --all is used.")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Disable response_format=json_object for providers that do not support it.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument("--output", type=str, help="Write full JSON results to this path.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run deterministic local smoke test without calling the API.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    client = build_client(args)
    model = getattr(client, "model", args.model)
    questions = resolve_questions(args)

    results: List[Dict[str, Any]] = []
    text_blocks: List[str] = []

    for source_index, question in questions:
        decomposition = decompose_question(client, question, model=model)
        results.append(decomposition_to_dict(decomposition, source_index))
        text_blocks.append(render_decomposition(decomposition, source_index))

    payload = {
        "model": model,
        "base_url": "mock" if args.mock else args.base_url,
        "count": len(results),
        "results": results,
    }

    if args.output:
        write_json(Path(args.output), payload)

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(text_blocks))
        if args.output:
            print(f"\n完整 JSON 已写入: {Path(args.output).resolve()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DecompositionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
