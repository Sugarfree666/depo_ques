from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    original_question: str | None = None
    replacements: list[dict[str, Any]] = field(default_factory=list)
    mask_mapping: dict[str, dict[str, Any]] = field(default_factory=dict)
    mask_mappings: list["MaskMapping"] = field(default_factory=list)
    preserved_type_variables: list[dict[str, Any]] = field(default_factory=list)
    anchor_extraction: ExtractionResult | None = None

    @property
    def masked_question(self) -> str:
        return self.question

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["masked_question"] = self.masked_question
        return data


class MaskReplacement(PlaceholderReplacement):
    """Replacement produced by the new mask-only Step 1 pipeline.

    This intentionally inherits the legacy ``PlaceholderReplacement`` fields so
    existing CLI/debug code and tests can continue to read ``question``,
    ``masked_question``, ``mask_mapping``, and ``mapping``.
    """

    def __init__(
        self,
        question: str | None = None,
        mapping: dict[str, str] | None = None,
        original_question: str | None = None,
        masked_question: str | None = None,
        replacements: list[dict[str, Any]] | None = None,
        mask_mapping: dict[str, dict[str, Any]] | None = None,
        mask_mappings: list["MaskMapping"] | None = None,
        preserved_type_variables: list[dict[str, Any]] | None = None,
        anchor_extraction: ExtractionResult | None = None,
    ) -> None:
        resolved_question = question if question is not None else masked_question
        if resolved_question is None:
            raise TypeError("MaskReplacement requires question or masked_question.")
        super().__init__(
            question=resolved_question,
            mapping=mapping or {},
            original_question=original_question,
            replacements=replacements or [],
            mask_mapping=mask_mapping or {},
            mask_mappings=mask_mappings or [],
            preserved_type_variables=preserved_type_variables or [],
            anchor_extraction=anchor_extraction,
        )


@dataclass
class MaskSpan:
    text: str
    start_char: int
    end_char: int
    kind_hint: str = "entity"
    semantic_type_hint: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaskSpanResult:
    mask_spans: list[MaskSpan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaskMapping:
    placeholder: str
    original_text: str
    kind_hint: str
    semantic_type_hint: str | None = None
    original_char_span: list[int] = field(default_factory=list)
    masked_char_span: list[int] = field(default_factory=list)
    token_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        return f"{_token_label(self.source, self.source_index)} --{self.relation}--> {_token_label(self.target, self.target_index)}"


@dataclass
class DependencyParse:
    tokens: list[CoreNLPToken]
    edges: list[DependencyEdge]
    raw: dict[str, Any] | None = None


@dataclass
class GraphNodeCandidate:
    node_id: str
    token_index: int
    graph_text: str
    placeholder: str | None = None
    restored_text: str = ""
    display_text: str = ""
    is_mask_placeholder: bool = False
    pos: str | None = None
    lemma: str | None = None
    kind_hint: str = "context"
    semantic_type_hint: str | None = None
    char_span: list[int] | None = None
    source_token_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.restored_text:
            self.restored_text = self.graph_text
        if not self.display_text:
            self.display_text = self.restored_text
        if not self.source_token_indices:
            self.source_token_indices = [self.token_index]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_llm_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "node_id": self.node_id,
            "text": self.display_text,
            "pos": self.pos,
            "kind_hint": self.kind_hint,
        }
        if self.semantic_type_hint:
            view["semantic_type_hint"] = self.semantic_type_hint
        return view


@dataclass
class RestoredGraphNodeCandidate(GraphNodeCandidate):
    """Candidate object after placeholder text has been restored for LLM display."""

    text: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.text:
            self.text = self.display_text

    def to_llm_view(self) -> dict[str, Any]:
        view = super().to_llm_view()
        view["text"] = self.text or self.display_text
        return view


@dataclass
class SelectedAnchor:
    node_id: str
    graph_text: str
    restored_text: str
    display_text: str
    anchor_kind: str
    source: str = "graph_node"
    token_index: int | None = None
    placeholder: str | None = None
    semantic_type_hint: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_llm_view(self) -> dict[str, Any]:
        view = {
            "node_id": self.node_id,
            "text": self.display_text,
            "anchor_kind": self.anchor_kind,
        }
        if self.semantic_type_hint:
            view["semantic_type_hint"] = self.semantic_type_hint
        if self.reason:
            view["reason"] = self.reason
        return view


@dataclass
class AnchorSelectionResult:
    selected_anchors: list[SelectedAnchor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

    def __iter__(self):
        return iter(self.selected_anchors)

    def __len__(self) -> int:
        return len(self.selected_anchors)

    def __getitem__(self, index: int) -> SelectedAnchor:
        return self.selected_anchors[index]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnchorConnectedSubgraph:
    selected_anchor_node_ids: list[str] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    shortest_paths: list[dict[str, Any]] = field(default_factory=list)
    graph: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_anchor_node_ids": self.selected_anchor_node_ids,
            "nodes": self.nodes,
            "edges": self.edges,
            "shortest_paths": self.shortest_paths,
        }


@dataclass
class RestoredAnchorConnectedSubgraph:
    selected_anchor_node_ids: list[str] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    shortest_paths: list[dict[str, Any]] = field(default_factory=list)
    display_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTNode:
    id: str
    label: str
    kind: str
    semantic_type: str | None = None
    source: str = "derived"
    source_graph_nodes: list[str] = field(default_factory=list)
    source_token_indices: list[int] = field(default_factory=list)
    grounding_text: str = ""
    cue_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTEdge:
    source: str
    target: str
    edge_type: str = "attribute"
    relation_hint: str = ""
    support_path: list[str] = field(default_factory=list)
    support_dependency_relations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTPrimaryOperator:
    operator: str = "NONE"
    inputs: list[str] = field(default_factory=list)
    output: str = "answer"
    cue_text: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTResult:
    status: str
    primary_operator: SemanticASTPrimaryOperator
    nodes: list[SemanticASTNode] = field(default_factory=list)
    edges: list[SemanticASTEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_by_id(self) -> dict[str, SemanticASTNode]:
        return {node.id: node for node in self.nodes}


def _token_label(word: str, index: int) -> str:
    if index <= 0:
        return word
    return f"{word}[{index}]"


@dataclass
class AnchorEdge:
    source: str
    target: str
    weight: float
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
    weighted_graph: Any | None = None
    anchor_subgraph: Any | None = None


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
    type: str = "edge"
    source: str = "llm"
    ast_edge: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
