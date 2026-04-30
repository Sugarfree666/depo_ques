from __future__ import annotations

import re
import unittest

from ast_builder import build_query_ast
from dependency_parser import parse_dependencies
from graph_builder import ast_to_semantic_graph, build_variable_syntax_tree
from mention_extractor import extract_mentions
from models import AtomicSubquestion
from semantic_decomposer import decompose_question
from subquestion_planner import answer_variable_reachable, plan_atomic_subquestions
from subquestion_verbalizer import verbalize_atomic_subquestion
from typed_clause_normalizer import normalize_dependencies
from validators import ValidationError, validate_ast, validate_subquestions


QUESTION = (
    "Which university did the CEO of the artificial intelligence company that developed AlphaGo "
    "graduate from, and in which city is that university located?"
)


MENTIONS = [
    {"id": "e1", "text": "AlphaGo", "kind": "constant", "type_hint": "AI_System"},
    {"id": "v1", "text": "artificial intelligence company", "kind": "variable", "type_hint": "Company"},
    {"id": "v2", "text": "CEO", "kind": "variable", "type_hint": "Person"},
    {"id": "v3", "text": "university", "kind": "variable", "type_hint": "University"},
    {"id": "v4", "text": "city", "kind": "answer_variable", "type_hint": "City"},
]


class FakeLLMClient:
    def __init__(self) -> None:
        self.call_stages: list[str] = []

    def call_json(self, stage, messages, temperature=0.0):
        self.call_stages.append(stage)
        if stage != "mention_extraction":
            raise AssertionError(f"Unexpected JSON LLM stage: {stage}")
        return {"mentions": MENTIONS}

    def call_text(self, stage, messages, temperature=0.0, response_format=None):
        self.call_stages.append(stage)
        if stage != "subquestion_verbalization":
            raise AssertionError(f"Unexpected text LLM stage: {stage}")
        prompt = messages[-1]["content"]
        edge_id = re.search(r"edge_id = (r\d+)", prompt).group(1)
        return {
            "r1": "Which artificial intelligence company developed AlphaGo?",
            "r2": "Who is the CEO of that artificial intelligence company?",
            "r3": "Which university did that CEO graduate from?",
            "r4": "In which city is that university located?",
        }[edge_id]


class PipelineTests(unittest.TestCase):
    def test_alphago_example_produces_four_one_hop_plans(self) -> None:
        result = decompose_question(QUESTION, FakeLLMClient())
        plans = result["subquestion_plans"]
        self.assertEqual(4, len(plans))
        self.assertEqual(["r1", "r2", "r3", "r4"], [plan["edge_id"] for plan in plans])
        self.assertTrue(all(result["programmatic_quality_checks"].values()))

    def test_semantic_graph_edges_count_equals_ast_relations_count(self) -> None:
        result = decompose_question(QUESTION, FakeLLMClient())
        self.assertEqual(
            len(result["query_ast"]["relations"]),
            len(result["semantic_graph"]["edges"]),
        )

    def test_every_subquestion_edge_id_exists(self) -> None:
        result = decompose_question(QUESTION, FakeLLMClient())
        edge_ids = {edge["id"] for edge in result["semantic_graph"]["edges"]}
        for subquestion in result["atomic_subquestions"]:
            self.assertIn(subquestion["edge_id"], edge_ids)

    def test_answer_variable_is_reachable_from_constant(self) -> None:
        fake = FakeLLMClient()
        mentions = extract_mentions(QUESTION, fake)
        parser_output = parse_dependencies(QUESTION)
        typed_clauses = normalize_dependencies(parser_output, mentions)
        ast = build_query_ast(QUESTION, mentions, typed_clauses)
        self.assertTrue(answer_variable_reachable(ast))

    def test_nonexistent_edge_id_validation_fails(self) -> None:
        result = decompose_question(QUESTION, FakeLLMClient())
        bad = [AtomicSubquestion(id="q_bad", edge_id="r999", question="Bad?")]
        with self.assertRaises(ValidationError):
            validate_subquestions(bad, result["semantic_graph"])

    def test_llm_is_not_called_during_deterministic_middle_stages(self) -> None:
        fake = FakeLLMClient()
        mentions = extract_mentions(QUESTION, fake)
        self.assertEqual(["mention_extraction"], fake.call_stages)

        parser_output = parse_dependencies(QUESTION)
        typed_clauses = normalize_dependencies(parser_output, mentions)
        ast = build_query_ast(QUESTION, mentions, typed_clauses)
        validate_ast(ast)
        semantic_graph = ast_to_semantic_graph(ast)
        build_variable_syntax_tree(ast)
        plans = plan_atomic_subquestions(ast)
        self.assertEqual(["mention_extraction"], fake.call_stages)

        subquestions = [verbalize_atomic_subquestion(plan, ast, fake) for plan in plans]
        validate_subquestions(subquestions, semantic_graph)
        self.assertEqual(
            ["mention_extraction"] + ["subquestion_verbalization"] * 4,
            fake.call_stages,
        )


if __name__ == "__main__":
    unittest.main()
