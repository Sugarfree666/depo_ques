from __future__ import annotations

from models import Mention, QueryAST, RelationNode, TypedClause


PREDICATE_NORMALIZATION = {
    "develop": "developed",
    "developed": "developed",
    "graduate from": "graduated_from",
    "graduated_from": "graduated_from",
    "be located in": "located_in",
    "located_in": "located_in",
    "CEO of": "CEO_of",
    "CEO_of": "CEO_of",
    "director of": "director_of",
    "nationality of": "nationality_of",
    "birthplace of": "birthplace_of",
}


def build_query_ast(question: str, mentions: list[Mention], typed_clauses: list[TypedClause]) -> QueryAST:
    relation_clauses = [clause for clause in typed_clauses if clause.clause_type != "coreference"]
    relations = [
        RelationNode(
            id=f"r{index}",
            predicate=normalize_predicate(clause.predicate),
            subject=clause.subject,
            object=clause.object,
            source_clause=clause.text,
            clause_type=clause.clause_type,
        )
        for index, clause in enumerate(relation_clauses, start=1)
    ]
    return QueryAST(
        question=question,
        mentions=mentions,
        relations=relations,
        root_answer_variable=_root_answer_variable(mentions),
        dependencies=_relation_dependencies(relations, mentions),
    )


def normalize_predicate(predicate: str) -> str:
    return PREDICATE_NORMALIZATION.get(predicate, predicate.replace(" ", "_"))


def _root_answer_variable(mentions: list[Mention]) -> str:
    answer_variables = [mention.id for mention in mentions if mention.kind == "answer_variable"]
    if answer_variables:
        return answer_variables[-1]

    variables = [mention.id for mention in mentions if mention.kind == "variable"]
    if variables:
        return variables[-1]

    raise ValueError("No variable or answer_variable mention found.")


def _relation_dependencies(relations: list[RelationNode], mentions: list[Mention]) -> list[tuple[str, str]]:
    known = {mention.id for mention in mentions if mention.kind == "constant"}
    remaining = list(relations)
    executed: list[RelationNode] = []
    dependencies: list[tuple[str, str]] = []

    while remaining:
        executable = None
        for relation in remaining:
            subject_known = relation.subject in known
            object_known = relation.object in known
            if subject_known ^ object_known:
                executable = relation
                break
        if executable is None:
            break

        for previous in reversed(executed):
            if (
                previous.subject in {executable.subject, executable.object}
                or previous.object in {executable.subject, executable.object}
            ):
                dependencies.append((previous.id, executable.id))
                break

        known.add(executable.subject)
        known.add(executable.object)
        executed.append(executable)
        remaining.remove(executable)

    return dependencies
