from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import (
    AnchorSelectionResult,
    AtomicSubquestion,
    MaskReplacement,
    MaskSpanResult,
    QuestionRecord,
    RestoredAnchorConnectedSubgraph,
    RestoredGraphNodeCandidate,
    SemanticASTResult,
)

if TYPE_CHECKING:
    from anchor_selector import AnchorSelector
    from ast_builder import SemanticASTOptimizer
    from corenlp_parser import CoreNLPParser
    from graph_builder import GraphBuilder
    from mask_span_extractor import MaskSpanExtractor
    from subquestion_generator import SubquestionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEPO decomposition with mask-only parsing, restored anchor selection, semantic AST, and one-hop subquestions."
    )
    parser.add_argument("--question", help="Run one manually supplied question instead of questions.json.")
    parser.add_argument("--questions-file", default="questions.json", help="Path to questions.json.")
    parser.add_argument("--api-key", help="OpenAI API key. Used only if OPENAI_API_KEY is not set.")
    parser.add_argument("--base-url", help="OpenAI base URL. Used only if OPENAI_BASE_URL is not set.")
    parser.add_argument(
        "--corenlp-url",
        default="http://localhost:9000",
        help="Endpoint used by Stanza CoreNLPClient for the managed CoreNLP server.",
    )
    parser.add_argument("--corenlp-memory", default="4G", help="Java heap memory for managed CoreNLP.")
    parser.add_argument(
        "--corenlp-home",
        help="Path to a Stanford CoreNLP directory containing stanford-corenlp*.jar files.",
    )
    parser.add_argument(
        "--corenlp-timeout-ms",
        type=int,
        default=60000,
        help="CoreNLP annotation timeout in milliseconds.",
    )
    parser.add_argument("--debug", action="store_true", help="Print detailed intermediate structures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    records = [QuestionRecord(question=args.question)] if args.question else read_questions(args.questions_file)

    try:
        from anchor_selector import AnchorSelector
        from ast_builder import SemanticASTOptimizer
        from corenlp_parser import CoreNLPConnectionError, CoreNLPParser
        from graph_builder import GraphBuilder
        from llm_client import LLMClient
        from mask_span_extractor import MaskSpanExtractor
        from subquestion_generator import SubquestionGenerator

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        mask_span_extractor = MaskSpanExtractor(llm_client)
        graph_builder = GraphBuilder()
        anchor_selector = AnchorSelector(llm_client)
        semantic_ast_optimizer = SemanticASTOptimizer(llm_client)
        subquestion_generator = SubquestionGenerator(llm_client)

        with CoreNLPParser(
            args.corenlp_url,
            timeout_ms=args.corenlp_timeout_ms,
            memory=args.corenlp_memory,
            corenlp_home=args.corenlp_home,
        ) as parser:
            for index, record in enumerate(records, start=1):
                result = run_pipeline(
                    record=record,
                    index=index,
                    mask_span_extractor=mask_span_extractor,
                    parser=parser,
                    graph_builder=graph_builder,
                    anchor_selector=anchor_selector,
                    semantic_ast_optimizer=semantic_ast_optimizer,
                    subquestion_generator=subquestion_generator,
                    debug=args.debug,
                )
                print_result(index, record, result, debug=args.debug)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (CoreNLPConnectionError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def run_pipeline(
    record: QuestionRecord,
    index: int,
    mask_span_extractor: "MaskSpanExtractor",
    parser: "CoreNLPParser",
    graph_builder: "GraphBuilder",
    anchor_selector: "AnchorSelector",
    semantic_ast_optimizer: "SemanticASTOptimizer",
    subquestion_generator: "SubquestionGenerator",
    debug: bool = False,
) -> dict[str, Any]:
    del index, debug
    from placeholder import selective_entity_masking

    mask_spans = mask_span_extractor.extract(record.question)
    replacement = selective_entity_masking(
        original_question=record.question,
        extracted_nodes=mask_spans,
    )
    dependency_parse = parser.parse(replacement.masked_question)
    weighted_graph = graph_builder.build_weighted_dependency_graph(dependency_parse)
    graph_node_candidates = graph_builder.build_graph_node_candidates(
        dependency_parse=dependency_parse,
        replacement=replacement,
    )
    restored_graph_node_candidates = graph_builder.restore_graph_node_candidates(
        graph_node_candidates=graph_node_candidates,
        replacement=replacement,
    )
    anchor_selection = anchor_selector.select(
        original_question=record.question,
        masked_question=replacement.masked_question,
        replacement=replacement,
        dependency_parse=dependency_parse,
        weighted_graph=weighted_graph,
        restored_graph_node_candidates=restored_graph_node_candidates,
    )
    anchor_connected_subgraph = graph_builder.build_anchor_connected_subgraph(
        weighted_graph=weighted_graph,
        selected_anchors=anchor_selection.selected_anchors,
        graph_node_candidates=graph_node_candidates,
    )
    restored_anchor_connected_subgraph = graph_builder.restore_anchor_connected_subgraph(
        anchor_connected_subgraph=anchor_connected_subgraph,
        replacement=replacement,
    )
    semantic_ast = semantic_ast_optimizer.optimize(
        original_question=record.question,
        replacement=replacement,
        selected_anchors=anchor_selection.selected_anchors,
        restored_anchor_connected_subgraph=restored_anchor_connected_subgraph,
    )
    subquestions = subquestion_generator.generate(
        original_question=record.question,
        ast=semantic_ast,
    )
    return {
        "mask_spans": mask_spans,
        "replacement": replacement,
        "dependency_parse": dependency_parse,
        "weighted_graph": weighted_graph,
        "graph_node_candidates": graph_node_candidates,
        "restored_graph_node_candidates": restored_graph_node_candidates,
        "anchor_selection": anchor_selection,
        "anchor_connected_subgraph": anchor_connected_subgraph,
        "restored_anchor_connected_subgraph": restored_anchor_connected_subgraph,
        "semantic_ast": semantic_ast,
        "subquestions": subquestions,
    }


def print_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    from graph_builder import format_weighted_graph_edges

    mask_spans: MaskSpanResult = result["mask_spans"]
    replacement: MaskReplacement = result["replacement"]
    dependency_parse = result["dependency_parse"]
    weighted_graph = result["weighted_graph"]
    restored_graph_node_candidates: list[RestoredGraphNodeCandidate] = result["restored_graph_node_candidates"]
    anchor_selection: AnchorSelectionResult = result["anchor_selection"]
    restored_anchor_connected_subgraph: RestoredAnchorConnectedSubgraph = result["restored_anchor_connected_subgraph"]
    semantic_ast: SemanticASTResult = result["semantic_ast"]
    subquestions: list[AtomicSubquestion] = result["subquestions"]

    separator = "=" * 60
    print(separator)
    title = f"Question {index}"
    if record.qid:
        title += f" ({record.qid})"
    print(title)
    print(separator)
    print()

    print("[Original Question]")
    print(record.question)
    print()

    print("[1. Mask Spans]")
    if replacement.mask_mappings:
        for mapping in replacement.mask_mappings:
            print(f"  - {mapping.original_text} -> {mapping.placeholder}")
    else:
        print("  (none)")
    if debug:
        _print_warnings(mask_spans.warnings)
    print()

    print("[2. Selective Masked Question]")
    print(replacement.masked_question)
    print()

    print("[3. CoreNLP Dependency Parse]")
    print("Edges:")
    if dependency_parse.edges:
        for edge in dependency_parse.edges:
            print(f"  - {edge.display()}")
    else:
        print("  (no dependency edges)")
    print()

    print("[4. Weighted Undirected Dependency Graph]")
    weighted_lines = format_weighted_graph_edges(weighted_graph)
    if weighted_lines:
        for line in weighted_lines:
            print(line)
    else:
        print("  (no weighted edges)")
    print()

    print("[5. Restored Graph Node Candidates]")
    printed_candidate_texts: set[str] = set()
    for candidate in restored_graph_node_candidates:
        if candidate.kind_hint == "context":
            continue
        if candidate.display_text in printed_candidate_texts:
            continue
        printed_candidate_texts.add(candidate.display_text)
        print(f"  - {candidate.display_text}")
    if not printed_candidate_texts:
        print("  (none)")
    print()

    print("[6. Selected Explicit Anchors]")
    if anchor_selection.selected_anchors:
        for anchor in anchor_selection.selected_anchors:
            print(f"  - {anchor.display_text}")
    else:
        print("  (none)")
    if debug:
        _print_warnings(anchor_selection.warnings)
    print()

    print("[7. Anchor Connected Subgraph]")
    print("Edges:")
    subgraph_edge_lines = _format_restored_subgraph_edges(restored_anchor_connected_subgraph)
    if subgraph_edge_lines:
        for line in subgraph_edge_lines:
            print(line)
    else:
        print("  (no subgraph edges)")
    print()

    print("[8. Final Semantic AST]")
    _print_semantic_ast(semantic_ast)
    if debug:
        _print_warnings(semantic_ast.warnings)
    print()

    print("[9. Atomic Subquestions]")
    if not subquestions:
        print("  (no atomic subquestions generated)")
    for item in subquestions:
        print(f"  q{item.index}: {item.question}")
    print()


def _format_restored_subgraph_edges(
    restored_anchor_connected_subgraph: RestoredAnchorConnectedSubgraph,
) -> list[str]:
    lines: list[str] = []
    for edge in restored_anchor_connected_subgraph.edges:
        source = edge.get("source")
        target = edge.get("target")
        source_text = edge.get("source_text", source)
        target_text = edge.get("target_text", target)
        relation = edge.get("relation") or "|".join(edge.get("relations", []))
        relation_text = relation or "related"
        lines.append(f"  - {source_text}[{source}] --{relation_text}--> {target_text}[{target}]")
    return lines


def _print_semantic_ast(semantic_ast: SemanticASTResult) -> None:
    operator = semantic_ast.primary_operator
    if operator.operator != "NONE":
        inputs = ", ".join(operator.inputs)
        output = operator.output or "answer"
        cue = f" cue={operator.cue_text}" if operator.cue_text else ""
        print(f"Operator: {operator.operator}({inputs}) -> {output}{cue}")
    else:
        print("Operator: NONE")

    print("Nodes:")
    if semantic_ast.nodes:
        for node in semantic_ast.nodes:
            print(f"  - {node.id}: {node.label}")
    else:
        print("  (none)")
    print("Edges:")
    if semantic_ast.edges:
        for edge in semantic_ast.edges:
            hint = f" ({edge.relation_hint})" if edge.relation_hint else ""
            print(f"  - {edge.source} -> {edge.target}{hint}")
    else:
        print("  (none)")


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print(f"  - {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
