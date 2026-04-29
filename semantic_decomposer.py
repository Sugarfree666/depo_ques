#!/usr/bin/env python3
"""Build a query AST and one-hop atomic subquestions for complex questions.

The pipeline is intentionally small and dependency-free:
1. Read questions from questions.json or a direct CLI argument.
2. Ask a GPT-4o-mini compatible chat-completions endpoint to extract
   entity/type variables and build a compiler-style query AST plus graph.
3. Ask the model again to generate one-hop atomic subquestions from the graph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


SYSTEM_PROMPT = """You are a rigorous semantic parser for hypergraph RAG question decomposition.

Your job is to transform a complex natural-language question into:
1. entity variables: concrete named entities or literal anchors from the question.
2. type variables: answerable typed variables, such as company, CEO, university, city.
3. dependency-style relation evidence.
4. a Query AST that organizes variables and relations like a compiler would.
5. a semantic graph whose one-hop edges can each be verbalized as exactly one atomic subquestion.

Rules:
- Preserve every constraint in the original question.
- Do not merge two relations into one graph edge.
- A graph edge must connect exactly two variables and represent one predicate/relation.
- Prefer a chain or small tree that supports step-by-step retrieval.
- Use stable ids: v1, v2, ... for variables; r1, r2, ... for relations/edges.
- If a relation is implicit, infer the most conservative relation phrase and mark evidence as "implicit".
- Keep surface text in the same language as the input question when possible.
- Output valid JSON only. Do not wrap the JSON in Markdown.
"""


EXTRACTION_PROMPT = """Parse the complex question into entity/type variables, a Query AST, and a one-hop semantic graph.

Return exactly this JSON shape:
{
  "question": "...",
  "variables": [
    {
      "id": "v1",
      "surface": "AlphaGo",
      "kind": "entity",
      "type_hint": "system/product",
      "is_anchor": true,
      "description": "concrete entity mentioned in the question"
    }
  ],
  "dependency_evidence": [
    {
      "id": "d1",
      "head": "v2",
      "dependent": "v1",
      "relation_phrase": "developed",
      "evidence_text": "developed AlphaGo"
    }
  ],
  "query_ast": {
    "node_type": "answer",
    "variable": "v5",
    "children": [
      {
        "node_type": "relation",
        "relation_id": "r4",
        "source": "v4",
        "target": "v5",
        "children": []
      }
    ]
  },
  "semantic_graph": {
    "nodes": [
      {
        "id": "v1",
        "label": "AlphaGo",
        "kind": "entity",
        "type_hint": "system/product"
      }
    ],
    "edges": [
      {
        "id": "r1",
        "source": "v2",
        "target": "v1",
        "relation": "developed",
        "source_role": "developer",
        "target_role": "developed object",
        "evidence": "developed AlphaGo"
      }
    ]
  },
  "root_answer_variable": "v5",
  "parse_notes": ["..."]
}

Question:
{question}
"""


SUBQUESTION_PROMPT = """Generate one-hop atomic subquestions from the semantic graph.

Input parse JSON:
{parse_json}

Return exactly this JSON shape:
{
  "atomic_subquestions": [
    {
      "id": "q1",
      "edge_id": "r1",
      "known_variables": ["v1"],
      "unknown_variable": "v2",
      "question": "Which artificial intelligence company developed AlphaGo?",
      "answer_placeholder": "X1",
      "depends_on": [],
      "reason": "one-hop relation between AlphaGo and artificial intelligence company"
    }
  ],
  "execution_order": ["q1"],
  "final_answer_variable": "v5",
  "quality_checks": {
    "all_edges_covered": true,
    "each_question_one_hop": true,
    "notes": []
  }
}

Rules:
- Each atomic subquestion must use exactly one semantic_graph edge.
- If neither endpoint is a concrete anchor, depend on the previous placeholder that makes it answerable.
- Use placeholders X1, X2, ... for intermediate answers.
- The output should form a retrieval plan from concrete anchors toward the root_answer_variable when possible.
- Keep subquestions concise and answerable.
- Output valid JSON only. Do not wrap the JSON in Markdown.
"""


class ProcessLogger:
    def __init__(self, path: Path, append: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self.path.write_text("", encoding="utf-8")

    def section(self, title: str) -> None:
        self._write(f"\n{'=' * 80}\n[{self._now()}] {title}\n{'=' * 80}\n")

    def text(self, title: str, content: str) -> None:
        self._write(f"\n--- {title} ---\n{content}\n")

    def json(self, title: str, value: Any) -> None:
        self.text(title, json.dumps(value, ensure_ascii=False, indent=2))

    def start_run(self, args: argparse.Namespace, question_count: int) -> None:
        self.section("Run started")
        self.json(
            "configuration",
            {
                "model": args.model,
                "base_url": args.base_url,
                "questions_file": str(args.questions_file),
                "question_count": question_count,
                "index": args.index,
                "all": args.all,
                "limit": args.limit,
                "output": str(args.output) if args.output else None,
                "log_file": str(args.log_file),
                "append_log": args.append_log,
                "custom_question": bool(args.question),
            },
        )

    def _write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(text)

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class OpenAICompatibleClient:
    api_key: str
    base_url: str
    model: str = DEFAULT_MODEL
    timeout: int = 60
    retries: int = 2

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        process_log: ProcessLogger | None = None,
        stage: str = "chat",
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if process_log:
            process_log.json(
                f"{stage} request",
                {
                    "model": self.model,
                    "temperature": temperature,
                    "response_format": payload["response_format"],
                    "messages": messages,
                },
            )

        content = self._post_chat_completions(payload)
        if process_log:
            process_log.text(f"{stage} raw response", content)

        parsed = parse_json_content(content)
        if process_log:
            process_log.json(f"{stage} parsed JSON", parsed)
        return parsed

    def _post_chat_completions(self, payload: dict[str, Any]) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                return data["choices"][0]["message"]["content"]
            except error.HTTPError as exc:
                last_error = exc
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code < 500 or attempt == self.retries:
                    raise RuntimeError(f"HTTP {exc.code} from chat endpoint: {detail}") from exc
            except (error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.retries:
                    raise RuntimeError(f"Failed to call chat endpoint: {exc}") from exc

            time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Failed to call chat endpoint: {last_error}")


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON: {content}") from exc
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model JSON response must be an object.")
    return parsed


def load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    questions: list[str] = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            question = item
        elif isinstance(item, dict) and isinstance(item.get("question"), str):
            question = item["question"]
        else:
            raise ValueError(f"Question item #{idx} must be a string or an object with a question field.")

        question = question.strip()
        if question:
            questions.append(question)
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


def build_parse(
    client: OpenAICompatibleClient,
    question: str,
    process_log: ProcessLogger | None = None,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXTRACTION_PROMPT.replace("{question}", question)},
    ]
    return client.chat_json(messages, process_log=process_log, stage="Stage 1: query AST and semantic graph")


def build_subquestions(
    client: OpenAICompatibleClient,
    parse_result: dict[str, Any],
    process_log: ProcessLogger | None = None,
) -> dict[str, Any]:
    parse_json = json.dumps(parse_result, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SUBQUESTION_PROMPT.replace("{parse_json}", parse_json)},
    ]
    return client.chat_json(messages, process_log=process_log, stage="Stage 2: atomic subquestions")


def decompose_question(
    client: OpenAICompatibleClient,
    question: str,
    process_log: ProcessLogger | None = None,
    question_index: int | None = None,
    question_count: int | None = None,
) -> dict[str, Any]:
    if process_log:
        if question_index is not None and question_count is not None:
            process_log.section(f"Question {question_index}/{question_count}")
        else:
            process_log.section("Question")
        process_log.text("original question", question)

    parse_result = build_parse(client, question, process_log=process_log)
    subquestion_result = build_subquestions(client, parse_result, process_log=process_log)
    result = {
        "question": question,
        "variables": parse_result.get("variables", []),
        "dependency_evidence": parse_result.get("dependency_evidence", []),
        "query_ast": parse_result.get("query_ast", {}),
        "semantic_graph": parse_result.get("semantic_graph", {}),
        "root_answer_variable": parse_result.get("root_answer_variable"),
        "parse_notes": parse_result.get("parse_notes", []),
        "atomic_subquestions": subquestion_result.get("atomic_subquestions", []),
        "execution_order": subquestion_result.get("execution_order", []),
        "final_answer_variable": subquestion_result.get(
            "final_answer_variable", parse_result.get("root_answer_variable")
        ),
        "quality_checks": subquestion_result.get("quality_checks", {}),
    }
    if process_log:
        process_log.text("compact semantic graph", format_compact_semantic_graph(result))
        process_log.text("console output block", format_console_output([result]))
        process_log.json("merged final result", result)
    return result


def variable_label_map(result: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}

    for variable in result.get("variables", []):
        if not isinstance(variable, dict):
            continue
        variable_id = variable.get("id")
        label = variable.get("surface") or variable.get("label")
        if isinstance(variable_id, str) and isinstance(label, str) and label.strip():
            labels[variable_id] = label.strip()

    graph = result.get("semantic_graph", {})
    for node in graph.get("nodes", []) if isinstance(graph, dict) else []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        label = node.get("label") or node.get("surface")
        if isinstance(node_id, str) and isinstance(label, str) and label.strip():
            labels[node_id] = label.strip()

    return labels


def is_anchor_variable(result: dict[str, Any], variable_id: str) -> bool:
    for variable in result.get("variables", []):
        if not isinstance(variable, dict) or variable.get("id") != variable_id:
            continue
        return bool(variable.get("is_anchor")) or variable.get("kind") == "entity"
    return False


def format_path_graph(edges: list[dict[str, Any]], labels: dict[str, str], result: dict[str, Any]) -> str | None:
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        adjacency.setdefault(source, []).append((target, edge))
        adjacency.setdefault(target, []).append((source, edge))

    if not adjacency or any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    if len(edges) != len(adjacency) - 1:
        return None

    leaves = [node_id for node_id, neighbors in adjacency.items() if len(neighbors) <= 1]
    anchors = [node_id for node_id in leaves if is_anchor_variable(result, node_id)]
    start = anchors[0] if anchors else (leaves[0] if leaves else next(iter(adjacency)))

    parts = [labels.get(start, start)]
    visited_edges: set[str] = set()
    previous: str | None = None
    current = start

    while True:
        next_items = [
            (neighbor, edge)
            for neighbor, edge in adjacency[current]
            if neighbor != previous and edge.get("id", f"{edge.get('source')}->{edge.get('target')}") not in visited_edges
        ]
        if not next_items:
            break

        neighbor, edge = next_items[0]
        edge_key = str(edge.get("id", f"{edge.get('source')}->{edge.get('target')}"))
        visited_edges.add(edge_key)

        if edge.get("source") == current and edge.get("target") == neighbor:
            arrow = "->"
        elif edge.get("target") == current and edge.get("source") == neighbor:
            arrow = "<-"
        else:
            arrow = "--"
        parts.append(f"{arrow}{labels.get(neighbor, neighbor)}")
        previous, current = current, neighbor

    return "".join(parts)


def format_edge_list_graph(edges: list[dict[str, Any]], labels: dict[str, str]) -> str:
    fragments: list[str] = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        fragments.append(f"{labels.get(source, source)}->{labels.get(target, target)}")
    return "；".join(fragments) if fragments else "未生成关系语义图"


def format_compact_semantic_graph(result: dict[str, Any]) -> str:
    graph = result.get("semantic_graph", {})
    if not isinstance(graph, dict):
        return "未生成关系语义图"

    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    labels = variable_label_map(result)
    path_graph = format_path_graph(edges, labels, result)
    if path_graph:
        return path_graph
    return format_edge_list_graph(edges, labels)


def format_console_output(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for result in results:
        lines = [
            f"原始问题：{result.get('question', '')}",
            f"关系语义图：{format_compact_semantic_graph(result)}",
            "分解后的子问题：",
        ]

        subquestions = result.get("atomic_subquestions", [])
        if isinstance(subquestions, list) and subquestions:
            for idx, item in enumerate(subquestions, start=1):
                if isinstance(item, dict):
                    question = item.get("question", "")
                else:
                    question = str(item)
                lines.append(f"{idx}. {question}")
        else:
            lines.append("未生成子问题")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def write_output(results: list[dict[str, Any]], output_path: Path | None) -> None:
    payload: dict[str, Any] | list[dict[str, Any]]
    payload = results[0] if len(results) == 1 else results

    if output_path:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(format_console_output(results))


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Query AST, semantic graph, and one-hop atomic subquestions."
    )
    parser.add_argument("--question", help="Directly decompose one custom question.")
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("questions.json"),
        help="JSON file containing questions. Defaults to questions.json.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="0-based question index from --questions-file. Ignored when --all or --question is used.",
    )
    parser.add_argument("--all", action="store_true", help="Process every question in --questions-file.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected questions to process.")
    parser.add_argument("--output", type=Path, help="Write JSON output to this file instead of stdout.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("decomposition.log"),
        help="Write a detailed decomposition process log to this file. Defaults to decomposition.log.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable process log file writing.")
    parser.add_argument("--append-log", action="store_true", help="Append to --log-file instead of overwriting it.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help=f"Base URL ending in /v1. Defaults to OPENAI_BASE_URL or {DEFAULT_BASE_URL}.",
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

    try:
        questions = select_questions(args)
        process_log = None if args.no_log else ProcessLogger(args.log_file, append=args.append_log)
        if process_log:
            process_log.start_run(args, len(questions))

        client = OpenAICompatibleClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
        )

        results = []
        for idx, question in enumerate(questions, start=1):
            results.append(
                decompose_question(
                    client,
                    question,
                    process_log=process_log,
                    question_index=idx,
                    question_count=len(questions),
                )
            )

        write_output(results, args.output)
        if process_log:
            process_log.section("Run completed")
        return 0
    except Exception as exc:
        process_log = locals().get("process_log")
        if isinstance(process_log, ProcessLogger):
            process_log.section("Run failed")
            process_log.text("error", str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
