from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import networkx as nx

from hypergraph_rag.execution import group_steps_by_level
from hypergraph_rag.models import DecompositionResult, QuestionRecord
from hypergraph_rag.query_ast import graph_to_dict, graph_to_edge_lines


def load_question_records(path: str | Path) -> list[QuestionRecord]:
    question_path = Path(path)
    payload = json.loads(question_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Question file must contain a JSON array.")

    records: list[QuestionRecord] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            question = item.strip()
            question_id = None
        elif isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            question_id = str(item["id"]).strip() if "id" in item and str(item["id"]).strip() else None
        else:
            raise ValueError(f"Unsupported question record at index {index}: {item!r}")

        if not question:
            raise ValueError(f"Missing question text at index {index}.")
        records.append(QuestionRecord(question=question, question_id=question_id))

    return records


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def result_to_dict(result: DecompositionResult, graph: nx.DiGraph) -> dict:
    return {
        "id": result.question_id,
        "question": result.question,
        "raw_llm_extraction_output": result.raw_extraction_output,
        "extracted_units": result.extracted_units.to_dict(),
        "dependency_parse": result.dependency_parse.to_dict(),
        "query_graph": graph_to_dict(graph),
        "execution_steps": [step.to_dict() for step in result.execution_steps],
    }


def render_console_result(result: DecompositionResult, graph: nx.DiGraph) -> str:
    lines: list[str] = []
    if result.question_id:
        lines.append(f"Question ID: {result.question_id}")
    lines.append(f"Question: {result.question}")
    lines.append("")
    lines.append("Query AST:")
    for line in graph_to_edge_lines(graph):
        lines.append(line)

    if result.dependency_parse.warnings:
        lines.append("")
        lines.append(
            f"Dependency parse note: {'; '.join(result.dependency_parse.warnings)}"
        )

    lines.append("")
    lines.append("Execution Steps:")

    counter = 1
    for level, steps in group_steps_by_level(result.execution_steps).items():
        lines.append(f"Level {level}:")
        for step in steps:
            label = step.natural_language_question
            if step.kind == "logical_operation":
                label = f"{label} [logical_operation]"
            lines.append(f"{counter}. {label} -> {step.output_variable}")
            counter += 1
        lines.append("")

    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
