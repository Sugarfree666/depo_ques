from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import QuestionRecord


def read_questions(path: str | Path) -> list[QuestionRecord]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Questions file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("questions.json must be a list of strings or objects.")

    records: list[QuestionRecord] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            question = item.strip()
            qid = None
        elif isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            qid_value: Any = item.get("id", item.get("qid"))
            qid = str(qid_value) if qid_value is not None else None
        else:
            raise ValueError(f"Unsupported question item at index {index}: {item!r}")
        if not question:
            raise ValueError(f"Question at index {index} is empty.")
        records.append(QuestionRecord(question=question, qid=qid))
    return records

