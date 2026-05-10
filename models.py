from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionRecord:
    question: str
    qid: str | None = None


@dataclass
class ExtractedNode:
    placeholder: str
    text: str
    kind: str
    semantic_type: str
    start: int | None = None
    end: int | None = None
    occurrence: int | None = None

    @property
    def is_entity(self) -> bool:
        return self.kind == "entity"

    @property
    def is_type_variable(self) -> bool:
        return self.kind == "type_variable"


@dataclass
class ExtractionResult:
    entities: list[ExtractedNode] = field(default_factory=list)
    type_variables: list[ExtractedNode] = field(default_factory=list)

    @property
    def nodes(self) -> list[ExtractedNode]:
        return [*self.entities, *self.type_variables]

    @property
    def placeholder_to_node(self) -> dict[str, ExtractedNode]:
        return {node.placeholder: node for node in self.nodes}


@dataclass
class PlaceholderReplacement:
    question: str
    mapping: dict[str, str]
    replacements: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CoreNLPToken:
    index: int
    word: str
    character_offset_begin: int = -1
    character_offset_end: int = -1
    lemma: str | None = None
    pos: str | None = None


@dataclass
class DependencyEdge:
    source: str
    relation: str
    target: str
    source_index: int
    target_index: int

    def display(self) -> str:
        return f"{self.source} --{self.relation}--> {self.target}"


@dataclass
class DependencyParse:
    tokens: list[CoreNLPToken]
    edges: list[DependencyEdge]
    raw: dict[str, Any] | None = None


@dataclass
class AnchorEdge:
    source: str
    target: str
    weight: int
    token_path: list[Any]
    path_words: list[str]
    relations: list[str]

    def display(self) -> str:
        return f"{self.source} ---- {self.target}"


@dataclass
class AnchorGraph:
    graph: Any
    edges: list[AnchorEdge]
    anchor_positions: dict[str, list[int]]
    folded_graph: Any | None = None


@dataclass
class OperatorSelection:
    operator: str
    attach_to: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class ASTResult:
    graph: Any
    operators: list[OperatorSelection]
    label_by_placeholder: dict[str, str]

    def display_label(self, node: str) -> str:
        return self.label_by_placeholder.get(node, node)


@dataclass
class AtomicSubquestion:
    index: int
    question: str
    answer_variable: str | None = None
    source_node: str | None = None
    target_node: str | None = None
    operator: str | None = None
