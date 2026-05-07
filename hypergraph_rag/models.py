from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryUnitNode:
    id: str
    label: str
    semantic_type: str
    source_span: str = ""
    kind: str = "entity"
    is_answer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RelationUnit:
    id: str
    source_id: str
    target_id: str
    relation: str
    surface: str = ""
    confidence: float = 0.0
    order_hint: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OperatorUnit:
    id: str
    operator: str
    input_ids: list[str]
    output_id: str = ""
    output_label: str = ""
    output_type: str = "Boolean"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BranchHint:
    kind: str
    node_ids: list[str]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExtractedQueryUnits:
    entities: list[QueryUnitNode]
    type_variables: list[QueryUnitNode]
    relations: list[RelationUnit]
    operators: list[OperatorUnit]
    branch_hints: list[BranchHint]
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [item.to_dict() for item in self.entities],
            "type_variables": [item.to_dict() for item in self.type_variables],
            "relations": [item.to_dict() for item in self.relations],
            "operators": [item.to_dict() for item in self.operators],
            "branch_hints": [item.to_dict() for item in self.branch_hints],
            "raw_response": self.raw_response,
        }


@dataclass(slots=True)
class DependencyToken:
    index: int
    text: str
    lemma: str
    pos: str
    dep: str
    head: str
    head_index: int
    is_stop: bool = False
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpanRecord:
    text: str
    label: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DependencyParse:
    model_name: str
    tokens: list[DependencyToken]
    noun_chunks: list[SpanRecord]
    named_entities: list[SpanRecord]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tokens": [item.to_dict() for item in self.tokens],
            "noun_chunks": [item.to_dict() for item in self.noun_chunks],
            "named_entities": [item.to_dict() for item in self.named_entities],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ASTNode:
    id: str
    label: str
    node_kind: str
    semantic_type: str = ""
    source_span: str = ""
    is_answer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ASTEdge:
    id: str
    source_id: str
    target_id: str
    relation: str
    edge_kind: str = "relation"
    source_role: str = ""
    order: int = 0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionStep:
    step_id: str
    kind: str
    source_node_ids: list[str]
    target_node_ids: list[str]
    relation_or_operator: str
    natural_language_question: str
    output_variable: str
    dependencies: list[str]
    execution_level: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QuestionRecord:
    question: str
    question_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.question_id, "question": self.question}


@dataclass(slots=True)
class DecompositionResult:
    question: str
    question_id: str | None
    extracted_units: ExtractedQueryUnits
    dependency_parse: DependencyParse
    execution_steps: list[ExecutionStep]
    raw_extraction_output: dict[str, Any]
    graph_metadata: dict[str, Any] = field(default_factory=dict)
