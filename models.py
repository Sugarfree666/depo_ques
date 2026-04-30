from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


@dataclass(frozen=True)
class Mention:
    id: str
    text: str
    kind: str
    type_hint: str


@dataclass(frozen=True)
class ParserToken:
    index: int
    text: str
    lemma: str
    pos: str
    dep: str
    head_index: int
    head_text: str


@dataclass(frozen=True)
class NounChunk:
    text: str
    start: int
    end: int
    root: str


@dataclass(frozen=True)
class SurfaceLink:
    text: str
    target_text: str
    link_type: str


@dataclass(frozen=True)
class ParserOutput:
    question: str
    parser_name: str
    tokens: list[ParserToken]
    noun_chunks: list[NounChunk]
    surface_links: list[SurfaceLink]


@dataclass(frozen=True)
class TypedClause:
    id: str
    text: str
    predicate: str
    clause_type: str
    subject: str
    object: str


@dataclass(frozen=True)
class RelationNode:
    id: str
    predicate: str
    subject: str
    object: str
    source_clause: str
    clause_type: str


@dataclass(frozen=True)
class QueryAST:
    question: str
    mentions: list[Mention]
    relations: list[RelationNode]
    root_answer_variable: str
    dependencies: list[tuple[str, str]]


@dataclass(frozen=True)
class SubquestionPlan:
    id: str
    edge_id: str
    predicate: str
    known_arg: str
    unknown_arg: str
    unknown_type: str


@dataclass(frozen=True)
class AtomicSubquestion:
    id: str
    edge_id: str
    question: str


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    return value
