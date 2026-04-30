from __future__ import annotations

from models import Mention, QueryAST, RelationNode, SubquestionPlan


def plan_atomic_subquestions(ast: QueryAST) -> list[SubquestionPlan]:
    mention_by_id = {mention.id: mention for mention in ast.mentions}
    known = {mention.id for mention in ast.mentions if mention.kind == "constant"}
    remaining = list(ast.relations)
    plans: list[SubquestionPlan] = []

    while remaining:
        executable = _next_executable_relation(remaining, known)
        if executable is None:
            remaining_ids = ", ".join(relation.id for relation in remaining)
            raise ValueError(f"No executable relation remains. Unresolved relations: {remaining_ids}")

        subject_known = executable.subject in known
        known_arg = executable.subject if subject_known else executable.object
        unknown_arg = executable.object if subject_known else executable.subject
        unknown_mention = mention_by_id[unknown_arg]
        plans.append(
            SubquestionPlan(
                id=f"q{len(plans) + 1}",
                edge_id=executable.id,
                predicate=executable.predicate,
                known_arg=known_arg,
                unknown_arg=unknown_arg,
                unknown_type=unknown_mention.type_hint,
            )
        )
        known.add(unknown_arg)
        remaining.remove(executable)

    return plans


def answer_variable_reachable(ast: QueryAST) -> bool:
    known = {mention.id for mention in ast.mentions if mention.kind == "constant"}
    remaining = list(ast.relations)

    while remaining:
        executable = _next_executable_relation(remaining, known)
        if executable is None:
            break

        known.add(executable.subject)
        known.add(executable.object)
        remaining.remove(executable)
        if ast.root_answer_variable in known:
            return True

    return ast.root_answer_variable in known


def _next_executable_relation(relations: list[RelationNode], known: set[str]) -> RelationNode | None:
    for relation in relations:
        subject_known = relation.subject in known
        object_known = relation.object in known
        if subject_known ^ object_known:
            return relation
    return None
