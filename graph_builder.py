from __future__ import annotations

from models import Mention, QueryAST
from subquestion_planner import plan_atomic_subquestions


def ast_to_semantic_graph(ast: QueryAST) -> dict:
    nodes = [
        {
            "id": mention.id,
            "label": mention.text,
            "kind": mention.kind,
            "type_hint": mention.type_hint,
        }
        for mention in ast.mentions
        if mention.kind != "coreference"
    ]
    edges = [
        {
            "id": relation.id,
            "predicate": relation.predicate,
            "source": relation.subject,
            "target": relation.object,
            "source_clause": relation.source_clause,
        }
        for relation in ast.relations
    ]
    return {"nodes": nodes, "edges": edges}


def build_variable_syntax_tree(ast: QueryAST) -> str:
    mention_by_id = {mention.id: mention for mention in ast.mentions}
    relation_by_id = {relation.id: relation for relation in ast.relations}
    root = mention_by_id[ast.root_answer_variable]
    lines = [f"Ask({_fmt_mention(root)})"]

    plans = plan_atomic_subquestions(ast)
    for depth, plan in enumerate(reversed(plans)):
        relation = relation_by_id[plan.edge_id]
        subject = mention_by_id[relation.subject]
        obj = mention_by_id[relation.object]
        indent = "    " * depth
        lines.append(f"{indent}`-- {relation.predicate}({_fmt_mention(subject)}, {_fmt_mention(obj)})")

    return "\n".join(lines)


def _fmt_mention(mention: Mention) -> str:
    return f"{mention.id}: {mention.type_hint}"
