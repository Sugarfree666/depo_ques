#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ast_builder import build_query_ast
from dependency_parser import parse_dependencies
from graph_builder import ast_to_semantic_graph, build_variable_syntax_tree
from llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleClient
from mention_extractor import extract_mentions
from models import AtomicSubquestion, to_dict
from subquestion_planner import plan_atomic_subquestions
from subquestion_verbalizer import verbalize_atomic_subquestion
from typed_clause_normalizer import normalize_dependencies
from validators import programmatic_quality_checks, validate_ast, validate_subquestions


class ProcessLogger:
    def __init__(self, path: Path, append: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self.path.write_text("", encoding="utf-8")

    def section(self, title: str) -> None:
        self._write(f"\n{'=' * 80}\n[{datetime.now().isoformat(timespec='seconds')}] {title}\n{'=' * 80}\n")

    def json(self, title: str, value: Any) -> None:
        self._write(f"\n--- {title} ---\n{json.dumps(to_dict(value), ensure_ascii=False, indent=2)}\n")

    def text(self, title: str, value: str) -> None:
        self._write(f"\n--- {title} ---\n{value}\n")

    def _write(self, value: str) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(value)


def decompose_question(question: str, llm_client: Any, process_log: ProcessLogger | None = None) -> dict:
    if process_log:
        process_log.section("Complex Question")
        process_log.text("question", question)

    mentions = extract_mentions(question, llm_client)
    if process_log:
        process_log.json("step 1 - entity/type variable recognition", mentions)

    raw_dependency_parse = parse_dependencies(question)
    if process_log:
        process_log.json("step 2 - deterministic dependency parsing", raw_dependency_parse)

    typed_clauses = normalize_dependencies(raw_dependency_parse, mentions)
    if process_log:
        process_log.json("step 3 - rule-based typed dependency normalization", typed_clauses)

    query_ast = build_query_ast(question, mentions, typed_clauses)
    validate_ast(query_ast)
    if process_log:
        process_log.json("step 4 - rule-based query ast construction", query_ast)

    semantic_graph = ast_to_semantic_graph(query_ast)
    variable_syntax_tree = build_variable_syntax_tree(query_ast)
    if process_log:
        process_log.text("step 6 - entity-type variable syntax tree", variable_syntax_tree)
        process_log.json("step 6 - semantic graph", semantic_graph)

    subquestion_plans = plan_atomic_subquestions(query_ast)
    if process_log:
        process_log.json("step 7 - programmatic atomic subquestion planning", subquestion_plans)

    atomic_subquestions = [
        verbalize_atomic_subquestion(plan, query_ast, llm_client)
        for plan in subquestion_plans
    ]
    validate_subquestions(atomic_subquestions, semantic_graph)
    if process_log:
        process_log.json("step 8 - llm verbalized atomic subquestions", atomic_subquestions)

    quality_checks = programmatic_quality_checks(
        query_ast,
        semantic_graph,
        atomic_subquestions,
        getattr(llm_client, "call_stages", []),
    )
    if process_log:
        process_log.json("step 9 - programmatic quality checks", quality_checks)

    return {
        "mentions": to_dict(mentions),
        "raw_dependency_parse": to_dict(raw_dependency_parse),
        "typed_clauses": to_dict(typed_clauses),
        "query_ast": to_dict(query_ast),
        "variable_syntax_tree": variable_syntax_tree,
        "semantic_graph": semantic_graph,
        "subquestion_plans": to_dict(subquestion_plans),
        "execution_order": [plan.id for plan in subquestion_plans],
        "atomic_subquestions": to_dict(atomic_subquestions),
        "programmatic_quality_checks": quality_checks,
    }


def load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    questions: list[str] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            question = item
        elif isinstance(item, dict) and isinstance(item.get("question"), str):
            question = item["question"]
        else:
            raise ValueError(f"Question item #{index} must be a string or an object with a question field.")
        if question.strip():
            questions.append(question.strip())
    return questions


def select_questions(args: argparse.Namespace) -> list[str]:
    if args.question:
        return [args.question.strip()]

    questions = load_questions(args.questions_file)
    if args.all:
        selected = questions
    else:
        if args.index < 0 or args.index >= len(questions):
            raise IndexError(f"--index must be between 0 and {len(questions) - 1}.")
        selected = [questions[args.index]]

    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def write_output(results: list[dict], output_path: Path | None) -> None:
    payload: dict | list[dict] = results[0] if len(results) == 1 else results
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict English complex-question decomposition pipeline."
    )
    parser.add_argument("--question", help="Directly decompose one custom English question.")
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("questions.json"),
        help="JSON file containing English questions. Defaults to questions.json.",
    )
    parser.add_argument("--index", type=int, default=0, help="0-based question index from --questions-file.")
    parser.add_argument("--all", action="store_true", help="Process every question in --questions-file.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected questions to process.")
    parser.add_argument("--output", type=Path, help="Write JSON output to this file instead of stdout.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("decomposition.log"),
        help="Write a detailed process log to this file. Defaults to decomposition.log.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable process log file writing.")
    parser.add_argument("--append-log", action="store_true", help="Append to --log-file instead of overwriting it.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible base URL ending in /v1. Defaults to {DEFAULT_BASE_URL}.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help=f"Defaults to {DEFAULT_MODEL}.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries for transient failures.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_arg_parser()
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    process_log: ProcessLogger | None = None
    try:
        questions = select_questions(args)
        process_log = None if args.no_log else ProcessLogger(args.log_file, append=args.append_log)
        if process_log:
            process_log.section("Run started")
            process_log.json(
                "configuration",
                {
                    "model": args.model,
                    "base_url": args.base_url,
                    "questions_file": str(args.questions_file),
                    "question_count": len(questions),
                    "output": str(args.output) if args.output else None,
                },
            )

        llm_client = OpenAICompatibleClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
        )
        results = [decompose_question(question, llm_client, process_log=process_log) for question in questions]
        write_output(results, args.output)

        if process_log:
            process_log.section("Run completed")
        return 0
    except Exception as exc:
        if process_log:
            process_log.section("Run failed")
            process_log.text("error", str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
