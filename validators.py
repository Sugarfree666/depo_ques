from __future__ import annotations

from llm_client import ALLOWED_LLM_STAGES
from models import AtomicSubquestion, QueryAST
from subquestion_planner import answer_variable_reachable, plan_atomic_subquestions


class ValidationError(ValueError):
    pass


def validate_ast(ast: QueryAST) -> None:
    mention_by_id = {mention.id: mention for mention in ast.mentions}
    if ast.root_answer_variable not in mention_by_id:
        raise ValidationError("root_answer_variable does not reference an existing Mention.")

    relation_ids = [relation.id for relation in ast.relations]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValidationError("RelationNode ids must be unique.")

    for relation in ast.relations:
        if not relation.subject or not relation.object or relation.subject == relation.object:
            raise ValidationError(f"Relation {relation.id} must connect exactly two distinct arguments.")
        if relation.subject not in mention_by_id or relation.object not in mention_by_id:
            raise ValidationError(f"Relation {relation.id} references a missing Mention.")
        if mention_by_id[relation.subject].kind == "coreference" or mention_by_id[relation.object].kind == "coreference":
            raise ValidationError(f"Relation {relation.id} contains unresolved coreference.")

    if not any(mention.kind == "constant" for mention in ast.mentions):
        raise ValidationError("At least one constant is required for executable decomposition.")

    if not answer_variable_reachable(ast):
        raise ValidationError("The answer variable is not reachable from any constant.")

    # This simulates execution and fails if only two-unbound-variable relations remain.
    plan_atomic_subquestions(ast)


def validate_subquestions(subquestions: list[AtomicSubquestion], semantic_graph: dict) -> None:
    edge_by_id = {edge["id"]: edge for edge in semantic_graph.get("edges", [])}
    seen: set[str] = set()

    for subquestion in subquestions:
        if subquestion.edge_id not in edge_by_id:
            raise ValidationError(f"Subquestion {subquestion.id} references nonexistent edge_id {subquestion.edge_id}.")
        if subquestion.edge_id in seen:
            raise ValidationError(f"edge_id {subquestion.edge_id} appears more than once.")
        seen.add(subquestion.edge_id)

        edge = edge_by_id[subquestion.edge_id]
        predicate = str(edge.get("predicate", "")).lower()
        if predicate in {"comparison", "greater_than", "less_than", "more_than", "fewer_than"}:
            raise ValidationError(f"Comparison edge {subquestion.edge_id} cannot be a factual one-hop question.")


def programmatic_quality_checks(
    ast: QueryAST,
    semantic_graph: dict,
    subquestions: list[AtomicSubquestion],
    llm_call_stages: list[str],
) -> dict:
    ast_valid = _passes(lambda: validate_ast(ast))
    subquestions_valid = _passes(lambda: validate_subquestions(subquestions, semantic_graph))
    graph_edges_match_ast_relations = len(semantic_graph.get("edges", [])) == len(ast.relations)
    answer_reachable = answer_variable_reachable(ast)
    llm_usage_valid = all(stage in ALLOWED_LLM_STAGES for stage in llm_call_stages)

    return {
        "ast_valid": ast_valid,
        "graph_edges_match_ast_relations": graph_edges_match_ast_relations,
        "all_subquestions_reference_existing_edges": subquestions_valid,
        "answer_variable_reachable": answer_reachable,
        "llm_used_only_for_mentions_and_verbalization": llm_usage_valid,
    }


def _passes(check) -> bool:
    try:
        check()
    except Exception:
        return False
    return True
