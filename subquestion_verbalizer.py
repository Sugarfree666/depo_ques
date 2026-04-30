from __future__ import annotations

from typing import Protocol

from models import AtomicSubquestion, QueryAST, RelationNode, SubquestionPlan


class VerbalizationLLM(Protocol):
    call_stages: list[str]

    def call_text(
        self,
        stage: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ) -> str:
        ...


SYSTEM_PROMPT = """You verbalize exactly one program-generated relation into one English atomic question.

Allowed task:
- Turn the provided SubquestionPlan and RelationNode into a single natural-language question.

Forbidden tasks:
- Do not create edge ids.
- Do not create variables.
- Do not create relations.
- Do not change the predicate.
- Do not add multi-hop constraints.
- Do not output JSON or Markdown.
"""


def verbalize_atomic_subquestion(
    plan: SubquestionPlan,
    ast: QueryAST,
    llm_client: VerbalizationLLM,
) -> AtomicSubquestion:
    relation = _relation_for_plan(plan, ast)
    mention_by_id = {mention.id: mention for mention in ast.mentions}
    known = mention_by_id[plan.known_arg]
    unknown = mention_by_id[plan.unknown_arg]
    prompt = f"""Create one English atomic question.

SubquestionPlan:
id = {plan.id}
edge_id = {plan.edge_id}
predicate = {plan.predicate}
known_arg_id = {plan.known_arg}
known_arg_text = {known.text}
unknown_arg_id = {plan.unknown_arg}
unknown_arg_text = {unknown.text}
unknown_arg_type = {plan.unknown_type}

RelationNode:
id = {relation.id}
predicate = {relation.predicate}
subject_id = {relation.subject}
subject_text = {mention_by_id[relation.subject].text}
object_id = {relation.object}
object_text = {mention_by_id[relation.object].text}
source_clause = {relation.source_clause}

Output only the question text.
"""
    text = llm_client.call_text(
        "subquestion_verbalization",
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return AtomicSubquestion(id=plan.id, edge_id=plan.edge_id, question=_clean_question(text))


def _relation_for_plan(plan: SubquestionPlan, ast: QueryAST) -> RelationNode:
    for relation in ast.relations:
        if relation.id == plan.edge_id:
            return relation
    raise ValueError(f"Plan {plan.id} references missing relation {plan.edge_id}.")


def _clean_question(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    if not cleaned.endswith("?"):
        cleaned += "?"
    return cleaned
