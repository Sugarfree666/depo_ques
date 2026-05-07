from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from hypergraph_rag import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MockLLMClient,
    QuestionRecord,
    QueryDecomposer,
    build_client,
    load_question_records,
    render_console_result,
    result_to_dict,
)
from hypergraph_rag.io_utils import write_json
from hypergraph_rag.parsing import DEFAULT_SPACY_MODEL


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose complex questions into a typed query graph and atomic sub-questions."
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Process one custom question from the command line.",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        help="Optional question id when --question is used.",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="questions.json",
        help="Path to a JSON file containing batch questions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional batch limit when reading from --questions-file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to write full structured JSON output.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL or the OpenAI API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name. Defaults to gpt-4o-mini.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP timeout in seconds for the OpenAI-compatible client.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count for transient API failures.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default=DEFAULT_SPACY_MODEL,
        help="spaCy model name for auxiliary dependency parsing.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the built-in mock client for the two reference example questions.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.mock:
        client = MockLLMClient()
    else:
        client = build_client(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
        )

    decomposer = QueryDecomposer(client=client, spacy_model=args.spacy_model)

    records = _resolve_records(args)
    payload_results: list[dict] = []
    console_blocks: list[str] = []

    for record in records:
        result, graph = decomposer.decompose_question(
            record.question,
            question_id=record.question_id,
        )
        payload_results.append(result_to_dict(result, graph))
        console_blocks.append(render_console_result(result, graph))

    print("\n\n".join(console_blocks))

    if args.output:
        payload = {
            "model": getattr(client, "model", args.model),
            "base_url": "mock" if args.mock else args.base_url,
            "result_count": len(payload_results),
            "results": payload_results,
        }
        write_json(args.output, payload)
        print(f"\nStructured JSON written to: {Path(args.output).resolve()}")

    return 0


def _resolve_records(args: argparse.Namespace):
    if args.question:
        return [
            QuestionRecord(
                question=args.question.strip(),
                question_id=args.question_id or None,
            )
        ]

    records = load_question_records(args.questions_file)
    if args.limit is not None:
        return records[: args.limit]
    return records


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
