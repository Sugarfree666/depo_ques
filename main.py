from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import ASTResult, AtomicSubquestion, ExtractionResult, PlaceholderReplacement, QuestionRecord

if TYPE_CHECKING:
    from ast_builder import ASTBuilder
    from corenlp_parser import CoreNLPParser
    from entity_extractor import EntityExtractor
    from graph_builder import GraphBuilder
    from subquestion_generator import SubquestionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEPO decomposition from selective masking through weighted dependency graphs, AST, and one-hop subquestions."
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
    parser.add_argument("--debug", action="store_true", help="Print additional intermediate structures.")
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
        from ast_builder import ASTBuilder
        from corenlp_parser import CoreNLPConnectionError, CoreNLPParser
        from entity_extractor import EntityExtractor
        from graph_builder import AnchorGraphError, GraphBuilder
        from llm_client import LLMClient
        from subquestion_generator import SubquestionGenerator

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        extractor = EntityExtractor(llm_client)
        graph_builder = GraphBuilder()
        ast_builder = ASTBuilder(llm_client)
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
                    extractor=extractor,
                    parser=parser,
                    graph_builder=graph_builder,
                    ast_builder=ast_builder,
                    subquestion_generator=subquestion_generator,
                    debug=args.debug,
                )
                print_result(index, record, result, debug=args.debug)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (CoreNLPConnectionError, AnchorGraphError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def run_pipeline(
    record: QuestionRecord,
    index: int,
    extractor: "EntityExtractor",
    parser: "CoreNLPParser",
    graph_builder: "GraphBuilder",
    ast_builder: "ASTBuilder",
    subquestion_generator: "SubquestionGenerator",
    debug: bool = False,
) -> dict[str, Any]:
    from placeholder import selective_entity_masking

    extraction = extractor.extract(record.question)
    replacement = selective_entity_masking(record.question, extraction)
    anchor_extraction = replacement.anchor_extraction or extraction
    dependency_parse = parser.parse(replacement.masked_question)
    anchor_graph = graph_builder.build_anchor_graph(dependency_parse, anchor_extraction)
    ast = ast_builder.build(record.question, anchor_extraction, replacement, anchor_graph)
    subquestions = subquestion_generator.generate(record.question, ast, anchor_extraction)
    return {
        "extraction": extraction,
        "anchor_extraction": anchor_extraction,
        "replacement": replacement,
        "dependency_parse": dependency_parse,
        "anchor_graph": anchor_graph,
        "ast": ast,
        "subquestions": subquestions,
    }


def print_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    from graph_builder import format_graph_lines, format_weighted_graph_edges

    extraction: ExtractionResult = result["extraction"]
    anchor_extraction: ExtractionResult = result.get("anchor_extraction", extraction)
    replacement: PlaceholderReplacement = result["replacement"]
    dependency_parse = result["dependency_parse"]
    anchor_graph = result["anchor_graph"]
    ast: ASTResult = result["ast"]
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

    print("[1. Entities and Type Variables]")
    print("Entities:")
    _print_nodes(extraction.entities)
    print()
    print("Type Variables:")
    _print_nodes(extraction.type_variables)
    print()

    print("[2. Selective Masked Question]")
    print(replacement.question)
    print()
    print("Mask Mapping:")
    if replacement.mask_mapping:
        for placeholder, info in replacement.mask_mapping.items():
            print(f"  - {placeholder}: {info.get('text')} ({info.get('semantic_type')})")
    else:
        print("  (no complex entities masked)")
    print()
    print("Preserved Type Variables:")
    if replacement.preserved_type_variables:
        for item in replacement.preserved_type_variables:
            print(f"  - {item.get('placeholder')}: {item.get('text')}")
    else:
        print("  (none)")
    print()

    print("[3. Dependency Graph: Enhanced++]")
    print("Edges:")
    edge_lines = _format_dependency_edges(dependency_parse)
    if edge_lines:
        for line in edge_lines:
            print(line)
    else:
        print("  (no dependency edges)")
    print()

    print("[4. Weighted Undirected Dependency Graph]")
    print("Edges:")
    weighted_graph = anchor_graph.weighted_graph
    weighted_lines = format_weighted_graph_edges(weighted_graph) if weighted_graph is not None else []
    if weighted_lines:
        for line in weighted_lines:
            print(line)
    else:
        print("  (no weighted edges)")
    print()

    print("[5. Anchor Shortest-Path Subgraph]")
    print("Edges:")
    anchor_subgraph = anchor_graph.anchor_subgraph
    subgraph_lines = format_weighted_graph_edges(anchor_subgraph) if anchor_subgraph is not None else []
    if subgraph_lines:
        for line in subgraph_lines:
            print(line)
    else:
        print("  (no anchor subgraph edges)")
    print()

    print("[6. Anchor-Only Semantic Graph]")
    entity_nodes = [node.placeholder for node in anchor_extraction.entities]
    for line in format_graph_lines(
        anchor_graph.graph,
        label_func=_semantic_label(anchor_extraction, replacement),
        entity_nodes=entity_nodes,
    ):
        print(line)
    print()

    print("[7. Final AST]")
    operator_nodes = [
        node for node, attrs in ast.graph.nodes(data=True) if attrs.get("kind") == "operator"
    ]
    for line in format_graph_lines(
        ast.graph,
        label_func=ast.display_label,
        entity_nodes=entity_nodes,
        operator_nodes=operator_nodes,
    ):
        print(line)
    print()
    print("Operators:")
    if ast.operators:
        for operator in ast.operators:
            attach = f" (attach_to: {', '.join(operator.attach_to)})" if operator.attach_to else ""
            print(f"  - {operator.operator}{attach}")
    else:
        print("  - NONE")
    print()

    print("[8. Atomic Subquestions]")
    if not subquestions:
        print("  (no atomic subquestions generated)")
    for item in subquestions:
        print(f"  q{item.index}: {item.question}")
        if item.answer_variable:
            print(f"      answer: {item.answer_variable}")
        print()


def _print_nodes(nodes: list[Any]) -> None:
    if not nodes:
        print("  (none)")
        return
    for node in nodes:
        print(f"  - {node.placeholder}: {node.text}")


def _format_dependency_edges(dependency_parse: Any) -> list[str]:
    return [f"  - {edge.display()}" for edge in dependency_parse.edges]


def _semantic_label(extraction: ExtractionResult, replacement: PlaceholderReplacement) -> Any:
    labels = dict(replacement.mapping)
    for node in extraction.nodes:
        labels.setdefault(node.placeholder, node.text)

    def label(node: str) -> str:
        return labels.get(node, node)

    return label


if __name__ == "__main__":
    raise SystemExit(main())
