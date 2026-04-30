from __future__ import annotations

from typing import Any, Protocol

from models import Mention


class MentionLLM(Protocol):
    call_stages: list[str]

    def call_json(self, stage: str, messages: list[dict[str, str]], temperature: float = 0.0) -> Any:
        ...


SYSTEM_PROMPT = """You extract only entity/type mentions from English complex questions.

Allowed task:
- Identify constants, variables, answer variables, and rule-detectable coreference mentions.

Forbidden tasks:
- Do not produce dependency parses.
- Do not produce query ASTs.
- Do not produce semantic graphs.
- Do not produce relations, edge ids, execution order, quality checks, or subquestions.

Return JSON only.
"""


USER_PROMPT_TEMPLATE = """Extract mentions from this English question.

Return this JSON shape:
{
  "mentions": [
    {
      "id": "e1",
      "text": "AlphaGo",
      "kind": "constant",
      "type_hint": "AI_System"
    },
    {
      "id": "v1",
      "text": "artificial intelligence company",
      "kind": "variable",
      "type_hint": "Company"
    },
    {
      "id": "v4",
      "text": "city",
      "kind": "answer_variable",
      "type_hint": "City"
    }
  ]
}

Rules:
- All input questions are English.
- Named entities are constants.
- Noun phrases introduced by which/what/who are variables.
- The final asked variable is answer_variable.
- Pronouns or demonstratives such as "this university", "that company", "its CEO" should be represented as kind "coreference" if needed. Put the antecedent id or best antecedent text in type_hint.
- Use e1, e2, ... for constants and v1, v2, ... for variables/answer variables.
- Do not invent facts or relations.

Question:
{question}
"""


def extract_mentions(question: str, llm_client: MentionLLM) -> list[Mention]:
    response = llm_client.call_json(
        "mention_extraction",
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{question}", question)},
        ],
        temperature=0.0,
    )
    raw_mentions = response.get("mentions", response) if isinstance(response, dict) else response
    if not isinstance(raw_mentions, list):
        raise ValueError("Mention extraction must return a list or an object with a mentions list.")

    mentions: list[Mention] = []
    for item in raw_mentions:
        if not isinstance(item, dict):
            raise ValueError("Each mention must be an object.")
        mention = Mention(
            id=str(item["id"]).strip(),
            text=str(item["text"]).strip(),
            kind=str(item["kind"]).strip(),
            type_hint=str(item["type_hint"]).strip(),
        )
        if not mention.id or not mention.text or not mention.kind or not mention.type_hint:
            raise ValueError(f"Incomplete mention: {item}")
        mentions.append(mention)

    return mentions
