from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from io_utils import read_questions
from models import ASTResult, AtomicSubquestion, ExtractionResult, PlaceholderReplacement, QuestionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEPO one-hop atomic subquestion decomposition using entity/type-variable relation graphs."
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
    extractor: EntityExtractor,
    parser: CoreNLPParser,
    graph_builder: GraphBuilder,
    ast_builder: ASTBuilder,
    subquestion_generator: SubquestionGenerator,
    debug: bool = False,
) -> dict[str, Any]:
    from placeholder import replace_with_placeholders

    extraction = extractor.extract(record.question)
    replacement = replace_with_placeholders(record.question, extraction)
    dependency_parse = parser.parse(record.question)
    anchor_graph = graph_builder.build_anchor_graph(dependency_parse, extraction)
    ast = ast_builder.build(record.question, extraction, replacement, anchor_graph)
    subquestions = subquestion_generator.generate(record.question, ast, extraction)
    return {
        "extraction": extraction,
        "replacement": replacement,
        "dependency_parse": dependency_parse,
        "anchor_graph": anchor_graph,
        "ast": ast,
        "subquestions": subquestions,
    }


def print_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    from graph_builder import format_dependency_edges, format_graph_lines

    extraction: ExtractionResult = result["extraction"]
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

    print("[2. Placeholder Question]")
    print(replacement.question)
    print()
    print("Placeholder Mapping:")
    for placeholder, text in replacement.mapping.items():
        print(f"  - {placeholder}: {text}")
    print()

    print("[3. Dependency Graph: Enhanced++]")
    print("Edges:")
    edge_lines = format_dependency_edges(dependency_parse)
    if edge_lines:
        for line in edge_lines:
            print(line)
    else:
        print("  (no dependency edges)")
    print()

    print("[4. Anchor MST / Anchor Graph]")
    entity_nodes = [node.placeholder for node in extraction.entities]
    for line in format_graph_lines(anchor_graph.graph, entity_nodes=entity_nodes):
        print(line)
    print()

    print("[5. Final AST]")
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

    print("[6. Atomic Subquestions]")
    if not subquestions:
        print("  (no atomic subquestions generated)")
    for item in subquestions:
        print(f"  q{item.index}: {item.question}")
        if item.answer_variable:
            print(f"      answer: {item.answer_variable}")
        print()

    if debug:
        print("[Debug]")
        debug_payload = {
            "placeholder_replacements": replacement.replacements,
            "anchor_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "path_words": edge.path_words,
                    "relations": edge.relations,
                }
                for edge in anchor_graph.edges
            ],
        }
        print(json.dumps(debug_payload, ensure_ascii=False, indent=2))
        print()


def _print_nodes(nodes: list[Any]) -> None:
    if not nodes:
        print("  (none)")
        return
    for node in nodes:
        print(f"  - {node.placeholder}: {node.text}")


if __name__ == "__main__":
    raise SystemExit(main())
