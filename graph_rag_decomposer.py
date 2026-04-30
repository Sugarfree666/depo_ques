#!/usr/bin/env python3
"""
Dependency-tree + Query-AST based complex question decomposer.

This script validates a Graph-RAG style decomposition pipeline:
1. Use an LLM to extract the ordered query entities / type variables.
2. Run dependency parsing with spaCy or stanza.
3. Collapse entity spans onto the dependency tree to build a Query AST.
4. Convert the token-level AST into an entity relation graph.
5. Traverse each one-hop edge and ask an LLM to generate an atomic sub-question.

Suggested installation:
    pip install openai networkx spacy stanza
    python -m spacy download zh_core_web_sm
    python -m spacy download en_core_web_sm
    python -c "import stanza; stanza.download('zh'); stanza.download('en')"

Examples:
    python graph_rag_decomposer.py --question "研发了 AlphaGo 的那家人工智能公司的 CEO 毕业于哪所大学，这所大学位于哪座城市？"
    python graph_rag_decomposer.py --question-file questions.json --index 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx


# OpenAI configuration placeholders. Replace them directly if you do not want
# to use environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ENTITY_MODEL = "gpt-4o-mini"
QUESTION_MODEL = "gpt-4o-mini"

DEFAULT_SAMPLE_QUESTION = (
    "研发了 AlphaGo 的那家人工智能公司的 CEO 毕业于哪所大学，这所大学位于哪座城市？"
)
DEFAULT_QUESTION_FILE = "questions.json"
PLACEHOLDER_POOL = ["X", "Y", "Z", "W", "V", "U", "T", "S", "R", "Q", "P"]


warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("stanza").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TokenInfo:
    index: int
    text: str
    lemma: str
    pos: str
    dep: str
    head: int
    start_char: int
    end_char: int


@dataclass
class ParseResult:
    language: str
    backend: str
    tokens: List[TokenInfo]
    root_index: int


@dataclass
class EntityNode:
    node_id: str
    text: str
    canonical: str
    kind: str
    mention_start: int = -1
    mention_end: int = -1
    token_start: int = -1
    token_end: int = -1
    head_token: int = -1


@dataclass
class AtomicQuestion:
    question: str
    input_var: Optional[str]
    output_var: Optional[str]


@dataclass
class DecompositionResult:
    question: str
    entities: List[EntityNode]
    parse_result: ParseResult
    query_ast: nx.Graph
    entity_graph: nx.DiGraph
    atomic_questions: List[AtomicQuestion]
    trace: Dict[str, Any]


def detect_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    return "zh" if zh_chars > 0 and zh_chars >= en_chars / 2 else "en"


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = clean_json_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return a JSON object.")
        return json.loads(match.group(0))


def normalize_for_match(text: str) -> Tuple[str, List[int]]:
    kept_chars: List[str] = []
    char_map: List[int] = []
    for index, char in enumerate(text):
        if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", char):
            kept_chars.append(char.lower())
            char_map.append(index)
    return "".join(kept_chars), char_map


def find_span(text: str, needle: str, cursor: int = 0) -> Optional[Tuple[int, int]]:
    needle = needle.strip()
    if not needle:
        return None

    direct = text.find(needle, cursor)
    if direct >= 0:
        return direct, direct + len(needle)

    lowered_text = text.lower()
    lowered_needle = needle.lower()
    lowered_direct = lowered_text.find(lowered_needle, cursor)
    if lowered_direct >= 0:
        return lowered_direct, lowered_direct + len(needle)

    normalized_text, char_map = normalize_for_match(text)
    normalized_needle, _ = normalize_for_match(needle)
    if not normalized_needle:
        return None

    normalized_index = normalized_text.find(normalized_needle)
    if normalized_index < 0:
        return None

    start_char = char_map[normalized_index]
    end_char = char_map[normalized_index + len(normalized_needle) - 1] + 1
    return start_char, end_char


def map_span_to_tokens(
    tokens: Sequence[TokenInfo], start_char: int, end_char: int
) -> Tuple[int, int, int]:
    covered = [
        token.index
        for token in tokens
        if not (token.end_char <= start_char or token.start_char >= end_char)
    ]
    if not covered:
        distances = [
            (
                min(
                    abs(token.start_char - start_char),
                    abs(token.end_char - end_char),
                ),
                token.index,
            )
            for token in tokens
        ]
        closest = min(distances)[1]
        return closest, closest, closest

    covered_set = set(covered)
    span_roots = [idx for idx in covered if tokens[idx].head not in covered_set]
    head_token = span_roots[0] if span_roots else covered[0]
    return min(covered), max(covered), head_token


def assign_entity_spans(question: str, parse_result: ParseResult, entities: List[EntityNode]) -> None:
    cursor = 0
    for entity in entities:
        span = find_span(question, entity.text, cursor)
        if span is None and entity.canonical != entity.text:
            span = find_span(question, entity.canonical, cursor)
        if span is None:
            span = find_span(question, entity.text, 0)
        if span is None and entity.canonical != entity.text:
            span = find_span(question, entity.canonical, 0)
        if span is None:
            raise ValueError(f"Failed to align entity span: {entity.text}")

        entity.mention_start, entity.mention_end = span
        entity.token_start, entity.token_end, entity.head_token = map_span_to_tokens(
            parse_result.tokens, entity.mention_start, entity.mention_end
        )
        cursor = entity.mention_end


def build_dependency_graph(parse_result: ParseResult) -> nx.Graph:
    graph = nx.Graph()
    for token in parse_result.tokens:
        graph.add_node(token.index)
        if token.index != token.head:
            graph.add_edge(token.index, token.head, dep=token.dep)
    return graph


def build_query_ast(parse_result: ParseResult, entities: Sequence[EntityNode]) -> nx.Graph:
    """
    Build the minimal token-level subtree that connects the parser root and the
    entity head tokens.

    The dependency parser returns a token tree. Each entity span is first
    collapsed to its syntactic head token. We then keep the union of the
    shortest dependency paths:
    - root -> entity head
    - consecutive entity head -> next entity head

    The result is a minimal connected subtree that still preserves the query
    chain. This subtree is the Query AST used as the bridge between syntax and
    the later entity graph.
    """

    full_graph = build_dependency_graph(parse_result)
    kept_nodes = {parse_result.root_index}

    for entity in entities:
        kept_nodes.update(nx.shortest_path(full_graph, parse_result.root_index, entity.head_token))

    for left, right in zip(entities, entities[1:]):
        kept_nodes.update(nx.shortest_path(full_graph, left.head_token, right.head_token))

    return full_graph.subgraph(kept_nodes).copy()


def extract_edge_evidence(
    question: str,
    parse_result: ParseResult,
    left: EntityNode,
    right: EntityNode,
    path: Sequence[int],
) -> Dict[str, Any]:
    left_span = set(range(left.token_start, left.token_end + 1))
    right_span = set(range(right.token_start, right.token_end + 1))
    skipped = left_span | right_span

    path_tokens = [parse_result.tokens[token_index].text for token_index in path]
    path_labels = [parse_result.tokens[token_index].dep for token_index in path]
    relation_tokens = [
        parse_result.tokens[token_index].text
        for token_index in path
        if token_index not in skipped and parse_result.tokens[token_index].pos != "PUNCT"
    ]

    surface_start = min(left.mention_start, right.mention_start)
    surface_end = max(left.mention_end, right.mention_end)
    surface_span = question[surface_start:surface_end].strip(" ，,。？！?;；")

    return {
        "path_tokens": path_tokens,
        "path_labels": path_labels,
        "relation_tokens": relation_tokens,
        "surface_span": surface_span,
    }


def build_entity_graph(
    question: str,
    parse_result: ParseResult,
    entities: Sequence[EntityNode],
    query_ast: nx.Graph,
) -> nx.DiGraph:
    """
    Convert the token-level Query AST into an entity relation graph.

    The AST still lives at token granularity, but Graph RAG needs semantic nodes.
    We therefore collapse each entity span into one graph node and connect
    consecutive entities in the semantic chain. The edge stores dependency-path
    evidence extracted from the AST, so later one-hop question generation can
    use real syntax instead of only surface order.
    """

    full_graph = build_dependency_graph(parse_result)
    graph = nx.DiGraph()

    for entity in entities:
        graph.add_node(
            entity.node_id,
            text=entity.text,
            canonical=entity.canonical,
            kind=entity.kind,
        )

    for edge_order, (left, right) in enumerate(zip(entities, entities[1:]), start=1):
        working_graph = query_ast
        try:
            path = nx.shortest_path(working_graph, left.head_token, right.head_token)
        except nx.NetworkXNoPath:
            path = nx.shortest_path(full_graph, left.head_token, right.head_token)

        evidence = extract_edge_evidence(question, parse_result, left, right, path)
        graph.add_edge(left.node_id, right.node_id, order=edge_order, **evidence)

    return graph


class DependencyParser:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Any] = {}

    def parse(self, text: str, download_models: bool = False) -> ParseResult:
        language = detect_language(text)
        if download_models:
            self.ensure_models(language)

        spacy_result = self._parse_with_spacy(text, language)
        if spacy_result is not None:
            return spacy_result

        stanza_result = self._parse_with_stanza(text, language)
        if stanza_result is not None:
            return stanza_result

        raise RuntimeError(self.build_setup_error(language))

    def ensure_models(self, language: str) -> None:
        if importlib.util.find_spec("spacy") is not None:
            try:
                import spacy
                from spacy.cli import download as spacy_download
            except Exception:
                pass
            else:
                if not any(spacy.util.is_package(name) for name in self._spacy_models_for_language(language)):
                    spacy_download(self._spacy_models_for_language(language)[0])

        if importlib.util.find_spec("stanza") is not None:
            try:
                import stanza
            except Exception:
                pass
            else:
                try:
                    stanza.Pipeline(
                        lang=language,
                        processors="tokenize,pos,lemma,depparse",
                        use_gpu=False,
                        verbose=False,
                    )
                except Exception:
                    stanza.download(language, processors="tokenize,pos,lemma,depparse", verbose=False)

    def build_setup_error(self, language: str) -> str:
        commands: List[str] = []
        spacy_installed = importlib.util.find_spec("spacy") is not None
        stanza_installed = importlib.util.find_spec("stanza") is not None

        if not spacy_installed and not stanza_installed:
            commands.append("pip install openai networkx spacy stanza")
        else:
            if not spacy_installed:
                commands.append("pip install spacy")
            if not stanza_installed:
                commands.append("pip install stanza")

        if spacy_installed and not self._has_any_spacy_model(language):
            commands.append(f"python -m spacy download {self._spacy_models_for_language(language)[0]}")

        if stanza_installed and not self._has_stanza_model(language):
            commands.append(f"python -c \"import stanza; stanza.download('{language}')\"")

        if not commands:
            commands = [
                "pip install -U spacy",
                f"python -m spacy download {self._spacy_models_for_language(language)[0]}",
                "pip install -U stanza",
                f"python -c \"import stanza; stanza.download('{language}')\"",
            ]

        message = [
            f"No dependency parser is available for language '{language}'.",
            "Install one parser stack and rerun:",
            *commands,
            "Or rerun this script with --download-models after the parser package is installed.",
        ]
        return "\n".join(message)

    @staticmethod
    def _spacy_models_for_language(language: str) -> List[str]:
        return ["zh_core_web_trf", "zh_core_web_sm"] if language == "zh" else ["en_core_web_trf", "en_core_web_sm"]

    def _has_any_spacy_model(self, language: str) -> bool:
        try:
            import spacy
        except Exception:
            return False
        return any(spacy.util.is_package(name) for name in self._spacy_models_for_language(language))

    @staticmethod
    def _has_stanza_model(language: str) -> bool:
        resource_dir = os.getenv("STANZA_RESOURCES_DIR")
        if not resource_dir:
            resource_dir = str(Path.home() / "stanza_resources")
        resources_path = Path(resource_dir) / "resources.json"
        language_path = Path(resource_dir) / language
        return resources_path.exists() and language_path.exists()

    def _parse_with_spacy(self, text: str, language: str) -> Optional[ParseResult]:
        try:
            import spacy
        except ImportError:
            return None

        model_candidates = (
            ["zh_core_web_trf", "zh_core_web_sm", "en_core_web_trf", "en_core_web_sm"]
            if language == "zh"
            else ["en_core_web_trf", "en_core_web_sm", "zh_core_web_trf", "zh_core_web_sm"]
        )

        for model_name in model_candidates:
            cache_key = ("spacy", model_name)
            try:
                if cache_key not in self._cache:
                    self._cache[cache_key] = spacy.load(model_name)
                nlp = self._cache[cache_key]
                doc = nlp(text)
            except OSError:
                continue

            tokens = [
                TokenInfo(
                    index=token.i,
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    dep=token.dep_,
                    head=token.head.i,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                )
                for token in doc
            ]
            root_index = next((token.i for token in doc if token.head == token), 0)
            detected_language = "zh" if model_name.startswith("zh_") else "en"
            return ParseResult(
                language=detected_language,
                backend=f"spacy:{model_name}",
                tokens=tokens,
                root_index=root_index,
            )

        return None

    def _parse_with_stanza(self, text: str, language: str) -> Optional[ParseResult]:
        try:
            import stanza
        except ImportError:
            return None

        cache_key = ("stanza", language)
        try:
            if cache_key not in self._cache:
                self._cache[cache_key] = stanza.Pipeline(
                    lang=language,
                    processors="tokenize,pos,lemma,depparse",
                    use_gpu=False,
                    verbose=False,
                )
            pipeline = self._cache[cache_key]
            doc = pipeline(text)
        except Exception:
            return None

        tokens: List[TokenInfo] = []
        roots: List[int] = []

        for sentence in doc.sentences:
            words = sentence.words
            base_index = len(tokens)
            spans = self._get_stanza_spans(text, words, base_index, tokens)

            for local_index, word in enumerate(words):
                global_index = base_index + local_index
                head_index = global_index if word.head == 0 else base_index + word.head - 1
                if word.head == 0:
                    roots.append(global_index)

                start_char, end_char = spans[local_index]
                tokens.append(
                    TokenInfo(
                        index=global_index,
                        text=word.text,
                        lemma=getattr(word, "lemma", word.text) or word.text,
                        pos=getattr(word, "upos", "") or "",
                        dep=getattr(word, "deprel", "") or "",
                        head=head_index,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )

        if not tokens:
            return None

        return ParseResult(
            language=language,
            backend=f"stanza:{language}",
            tokens=tokens,
            root_index=roots[0] if roots else 0,
        )

    @staticmethod
    def _get_stanza_spans(
        text: str,
        words: Sequence[Any],
        base_index: int,
        existing_tokens: Sequence[TokenInfo],
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        cursor = existing_tokens[-1].end_char if existing_tokens else 0

        for word in words:
            start_char = getattr(word, "start_char", None)
            end_char = getattr(word, "end_char", None)
            if start_char is None or end_char is None:
                found = text.find(word.text, cursor)
                if found < 0:
                    found = text.find(word.text)
                if found < 0:
                    found = cursor
                start_char = found
                end_char = found + len(word.text)
            spans.append((start_char, end_char))
            cursor = end_char

        return spans


class OpenAIJSONClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        if not api_key or api_key == "YOUR_API_KEY":
            raise RuntimeError("Please set OPENAI_API_KEY or edit OPENAI_API_KEY in the script.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The 'openai' package is not installed. Run: pip install openai") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call_json(self, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )
        return extract_first_json_object(response.output_text)


class GraphRAGDecomposer:
    def __init__(self, llm: OpenAIJSONClient, parser: DependencyParser) -> None:
        self.llm = llm
        self.parser = parser

    def extract_entities(self, question: str) -> Tuple[List[EntityNode], Dict[str, Any]]:
        system_prompt = (
            "You extract the semantic query chain for complex question decomposition.\n"
            "Return JSON only.\n"
            "Schema:\n"
            "{\n"
            '  "entities": [\n'
            '    {"text": "...", "canonical": "...", "kind": "entity|type|role"}\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Keep only the minimal set of nodes needed to answer the question.\n"
            "- Order the nodes from the grounded starting entity to the final answer type.\n"
            "- Preserve the original language of the question.\n"
            "- Use short surface forms such as AlphaGo, CEO, university, city, 大学, 城市.\n"
            "- 'entity' is a concrete named thing. 'role' is a role/title such as CEO. "
            "'type' is a generic target category such as company, university, city.\n"
            "- Do not include explanation."
        )
        user_prompt = f"Question:\n{question}"
        payload = self.llm.call_json(ENTITY_MODEL, system_prompt, user_prompt)

        raw_entities = payload.get("entities", [])
        if not isinstance(raw_entities, list):
            raise ValueError("Entity extraction returned an invalid JSON schema.")

        entities: List[EntityNode] = []
        seen: set[Tuple[str, str]] = set()
        for index, item in enumerate(raw_entities, start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            canonical = str(item.get("canonical", text)).strip() or text
            kind = str(item.get("kind", "type")).strip().lower() or "type"
            if not text:
                continue
            marker = (text, kind)
            if marker in seen:
                continue
            seen.add(marker)
            entities.append(
                EntityNode(
                    node_id=f"E{index}",
                    text=text,
                    canonical=canonical,
                    kind=kind,
                )
            )

        if len(entities) < 2:
            raise ValueError("At least two ordered query nodes are required.")

        return entities, payload

    def generate_atomic_question(
        self,
        original_question: str,
        source_entity: EntityNode,
        target_entity: EntityNode,
        edge_payload: Dict[str, Any],
        subject_text: str,
    ) -> Tuple[str, Dict[str, Any], bool]:
        system_prompt = (
            "You generate exactly one absolute atomic sub-question for one edge of a query graph.\n"
            "Return JSON only: {\"question\": \"...\"}\n"
            "Rules:\n"
            "- Same language as the original question.\n"
            "- Ask only for the next target node, never for any later hop.\n"
            "- Use subject_text literally as the subject mention. If it is X/Y/Z, keep the placeholder unchanged.\n"
            "- Keep the question short, direct, and answerable in one hop.\n"
            "- Do not include explanation or options.\n"
            "- Prefer natural phrasing such as '哪个人工智能公司研发了AlphaGo？', "
            "'X的CEO是谁？', 'Y毕业于哪所大学？', 'Z位于哪座城市？'."
        )
        user_prompt = json.dumps(
            {
                "original_question": original_question,
                "subject_text": subject_text,
                "source_node": {
                    "text": source_entity.text,
                    "kind": source_entity.kind,
                },
                "target_node": {
                    "text": target_entity.text,
                    "kind": target_entity.kind,
                },
                "ast_evidence": edge_payload,
            },
            ensure_ascii=False,
            indent=2,
        )

        result = self.llm.call_json(QUESTION_MODEL, system_prompt, user_prompt)
        question = str(result.get("question", "")).strip()
        if question:
            return question, result, False
        return self._heuristic_atomic_question(
            original_question, subject_text, target_entity.text, edge_payload
        ), result, True

    @staticmethod
    def _heuristic_atomic_question(
        original_question: str,
        subject_text: str,
        target_text: str,
        edge_payload: Dict[str, Any],
    ) -> str:
        language = detect_language(original_question)
        relation_text = "".join(edge_payload.get("relation_tokens", []))
        surface_span = str(edge_payload.get("surface_span", ""))
        evidence_text = f"{relation_text} {surface_span}".lower()

        if language == "zh":
            if "ceo" in evidence_text:
                return f"{subject_text}的{target_text}是谁？"
            if "毕业" in evidence_text:
                return f"{subject_text}毕业于哪所大学？"
            if "位于" in evidence_text or "城市" in target_text:
                return f"{subject_text}位于哪座城市？"
            if any(keyword in evidence_text for keyword in ("研发", "开发", "创造", "推出")):
                return f"哪个{target_text}研发了{subject_text}？"
            return f"与{subject_text}直接相关的{target_text}是什么？"

        if "ceo" in evidence_text:
            return f"Who is the CEO of {subject_text}?"
        if "graduat" in evidence_text or "university" in target_text.lower():
            return f"Which university did {subject_text} graduate from?"
        if "city" in target_text.lower() or "locat" in evidence_text:
            return f"Which city is {subject_text} located in?"
        if any(keyword in evidence_text for keyword in ("develop", "create", "build", "invent")):
            return f"Which {target_text} developed {subject_text}?"
        return f"What {target_text} is directly related to {subject_text}?"

    def decompose(
        self, question: str, download_models: bool = False
    ) -> DecompositionResult:
        entities, entity_payload = self.extract_entities(question)
        parse_result = self.parser.parse(question, download_models=download_models)
        assign_entity_spans(question, parse_result, entities)
        query_ast = build_query_ast(parse_result, entities)
        entity_graph = build_entity_graph(question, parse_result, entities, query_ast)

        atomic_questions: List[AtomicQuestion] = []
        atomic_trace: List[Dict[str, Any]] = []
        entity_by_id = {entity.node_id: entity for entity in entities}
        current_subject = entities[0].text
        current_input_var: Optional[str] = None
        ordered_edges = sorted(entity_graph.edges(data=True), key=lambda item: item[2]["order"])

        for edge_index, (source_id, target_id, edge_payload) in enumerate(ordered_edges):
            source_entity = entity_by_id[source_id]
            target_entity = entity_by_id[target_id]
            output_var = (
                PLACEHOLDER_POOL[edge_index] if edge_index < len(ordered_edges) - 1 else None
            )
            question_text, llm_output, used_heuristic = self.generate_atomic_question(
                original_question=question,
                source_entity=source_entity,
                target_entity=target_entity,
                edge_payload=edge_payload,
                subject_text=current_subject,
            )
            atomic_questions.append(
                AtomicQuestion(
                    question=question_text,
                    input_var=current_input_var,
                    output_var=output_var,
                )
            )
            atomic_trace.append(
                {
                    "step": edge_index + 1,
                    "source_node": source_entity.text,
                    "target_node": target_entity.text,
                    "subject_text": current_subject,
                    "input_var": current_input_var,
                    "output_var": output_var,
                    "edge_payload": edge_payload,
                    "llm_output": llm_output,
                    "used_heuristic_fallback": used_heuristic,
                    "final_question": question_text,
                }
            )
            if output_var is not None:
                current_subject = output_var
                current_input_var = output_var

        return DecompositionResult(
            question=question,
            entities=entities,
            parse_result=parse_result,
            query_ast=query_ast,
            entity_graph=entity_graph,
            atomic_questions=atomic_questions,
            trace={
                "entity_extraction": {
                    "model": ENTITY_MODEL,
                    "llm_output": entity_payload,
                },
                "atomic_generation": {
                    "model": QUESTION_MODEL,
                    "steps": atomic_trace,
                },
            },
        )


def load_question_from_json(path: Path, index: int) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Question file must contain a JSON array.")
    if not 0 <= index < len(data):
        raise IndexError(f"Question index {index} is out of range for {path}.")

    item = data[index]
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("question", "query", "text"):
            if key in item:
                return str(item[key])
    raise ValueError(f"Unsupported question record at index {index}: {item!r}")


def resolve_question(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    if args.question:
        return args.question.strip(), {"mode": "custom_input"}

    file_path = Path(args.question_file) if args.question_file else Path(DEFAULT_QUESTION_FILE)
    if file_path.exists():
        return load_question_from_json(file_path, args.index), {
            "mode": "question_file",
            "path": str(file_path.resolve()),
            "index": args.index,
        }

    return DEFAULT_SAMPLE_QUESTION, {"mode": "default_sample"}


def serialize_tokens(tokens: Sequence[TokenInfo]) -> List[Dict[str, Any]]:
    return [
        {
            "index": token.index,
            "text": token.text,
            "lemma": token.lemma,
            "pos": token.pos,
            "dep": token.dep,
            "head": token.head,
            "start_char": token.start_char,
            "end_char": token.end_char,
        }
        for token in tokens
    ]


def serialize_entities(entities: Sequence[EntityNode]) -> List[Dict[str, Any]]:
    return [
        {
            "node_id": entity.node_id,
            "text": entity.text,
            "canonical": entity.canonical,
            "kind": entity.kind,
            "mention_start": entity.mention_start,
            "mention_end": entity.mention_end,
            "token_start": entity.token_start,
            "token_end": entity.token_end,
            "head_token": entity.head_token,
        }
        for entity in entities
    ]


def serialize_query_ast(query_ast: nx.Graph, parse_result: ParseResult) -> Dict[str, Any]:
    token_map = {token.index: token for token in parse_result.tokens}
    ordered_nodes = sorted(query_ast.nodes())
    ordered_edges = sorted(query_ast.edges(data=True), key=lambda item: (min(item[0], item[1]), max(item[0], item[1])))
    return {
        "nodes": [
            {
                "token_index": token_index,
                "text": token_map[token_index].text,
                "dep": token_map[token_index].dep,
                "head": token_map[token_index].head,
            }
            for token_index in ordered_nodes
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "dep": edge_data.get("dep", ""),
            }
            for source, target, edge_data in ordered_edges
        ],
    }


def serialize_entity_graph(entity_graph: nx.DiGraph) -> Dict[str, Any]:
    ordered_edges = sorted(entity_graph.edges(data=True), key=lambda item: item[2]["order"])
    return {
        "nodes": [
            {
                "node_id": node_id,
                **node_data,
            }
            for node_id, node_data in entity_graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                **edge_data,
            }
            for source, target, edge_data in ordered_edges
        ],
    }


def serialize_atomic_questions(atomic_questions: Sequence[AtomicQuestion]) -> List[Dict[str, Any]]:
    return [
        {
            "step": index,
            "question": item.question,
            "input_var": item.input_var,
            "output_var": item.output_var,
        }
        for index, item in enumerate(atomic_questions, start=1)
    ]


def build_log_payload(
    args: argparse.Namespace,
    question: str,
    source_info: Dict[str, Any],
    result: Optional[DecompositionResult] = None,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success" if error is None else "error",
        "question": question,
        "source": source_info,
        "configuration": {
            "entity_model": ENTITY_MODEL,
            "question_model": QUESTION_MODEL,
            "base_url": args.base_url,
            "download_models": bool(args.download_models),
        },
    }

    if result is not None:
        payload["decomposition"] = {
            "parser": {
                "language": result.parse_result.language,
                "backend": result.parse_result.backend,
                "root_index": result.parse_result.root_index,
            },
            "tokens": serialize_tokens(result.parse_result.tokens),
            "entities": serialize_entities(result.entities),
            "query_ast": serialize_query_ast(result.query_ast, result.parse_result),
            "entity_graph": serialize_entity_graph(result.entity_graph),
            "atomic_questions": serialize_atomic_questions(result.atomic_questions),
            "trace": result.trace,
        }

    if error is not None:
        payload["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    return payload


def write_log_file(log_dir: Path, payload: Dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    status = payload.get("status", "success")
    file_path = log_dir / f"{timestamp}_{status}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def render_graph_text(graph: nx.DiGraph) -> str:
    ordered_edges = sorted(graph.edges(data=True), key=lambda item: item[2]["order"])
    parts = [f"{graph.nodes[source]['text']} -> {graph.nodes[target]['text']}" for source, target, _ in ordered_edges]
    return ", ".join(parts)


def render_atomic_question_line(index: int, item: AtomicQuestion) -> str:
    markers: List[str] = []
    if item.input_var:
        markers.append(f"输入={item.input_var}")
    if item.output_var:
        markers.append(f"输出={item.output_var}")
    marker_text = f" [{', '.join(markers)}]" if markers else ""
    return f"Q{index}{marker_text} {item.question}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dependency-tree + Graph-RAG question decomposer")
    parser.add_argument("--question", type=str, default=None, help="Custom input question.")
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help=f"JSON file path. Defaults to {DEFAULT_QUESTION_FILE} when the file exists.",
    )
    parser.add_argument("--index", type=int, default=0, help="Question index in the JSON array.")
    parser.add_argument("--api-key", type=str, default=OPENAI_API_KEY, help="OpenAI API key.")
    parser.add_argument("--base-url", type=str, default=OPENAI_BASE_URL, help="OpenAI base URL.")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Attempt to download missing spaCy/stanza parser models before running.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="decomposition_logs",
        help="Directory for per-run JSON logs.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log file generation.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    log_dir = Path(args.log_dir)
    question = ""
    source_info: Dict[str, Any] = {"mode": "unresolved"}
    result: Optional[DecompositionResult] = None

    try:
        question, source_info = resolve_question(args)
        llm = OpenAIJSONClient(api_key=args.api_key, base_url=args.base_url)
        parser = DependencyParser()
        decomposer = GraphRAGDecomposer(llm=llm, parser=parser)
        result = decomposer.decompose(question, download_models=args.download_models)
        if not args.no_log:
            write_log_file(log_dir, build_log_payload(args, question, source_info, result=result))

        print(f"1. 原问题：{question}")
        print(f"2. 查询实体关系图：{render_graph_text(result.entity_graph)}")
        print("3. 分解后的子问题：")
        for index, item in enumerate(result.atomic_questions, start=1):
            print(render_atomic_question_line(index, item))
    except Exception as exc:
        if not args.no_log:
            write_log_file(log_dir, build_log_payload(args, question, source_info, result=result, error=exc))
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
