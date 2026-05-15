from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import ExtractionResult, PlaceholderReplacement, QuestionRecord

if TYPE_CHECKING:
    from corenlp_parser import CoreNLPParser
    from entity_extractor import EntityExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect CoreNLP Enhanced++ dependency graphs after entity extraction and selective masking."
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
    parser.add_argument("--debug", action="store_true", help="Accepted for compatibility; output still stops at dependency edges.")
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
        from corenlp_parser import CoreNLPConnectionError, CoreNLPParser
        from entity_extractor import EntityExtractor
        from llm_client import LLMClient

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        extractor = EntityExtractor(llm_client)

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
    extractor: "EntityExtractor",
    parser: "CoreNLPParser",
    debug: bool = False,
) -> dict[str, Any]:
    from placeholder import selective_entity_masking

    extraction = extractor.extract(record.question)
    replacement = selective_entity_masking(record.question, extraction)
    dependency_parse = parser.parse(replacement.masked_question)
    return {
        "extraction": extraction,
        "replacement": replacement,
        "dependency_parse": dependency_parse,
    }


def print_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    extraction: ExtractionResult = result["extraction"]
    replacement: PlaceholderReplacement = result["replacement"]
    dependency_parse = result["dependency_parse"]

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


def _print_nodes(nodes: list[Any]) -> None:
    if not nodes:
        print("  (none)")
        return
    for node in nodes:
        print(f"  - {node.placeholder}: {node.text}")


def _format_dependency_edges(dependency_parse: Any) -> list[str]:
    return [f"  - {edge.display()}" for edge in dependency_parse.edges]


if __name__ == "__main__":
    raise SystemExit(main())
