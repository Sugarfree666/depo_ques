from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hypergraph_rag import MockLLMClient, QueryDecomposer, graph_to_edge_lines, load_question_records


ALPHAGO_QUESTION = (
    "Which university did the CEO of the artificial intelligence company that "
    "developed AlphaGo graduate from and in which city is this university located?"
)

PARALLEL_QUESTION = (
    "Do director of film Ten9Eight: Shoot For The Moon and director of film "
    "Sabotage (1936 Film) share the same nationality?"
)


class QueryDecomposerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decomposer = QueryDecomposer(client=MockLLMClient())

    def test_multihop_reference_example(self) -> None:
        result, graph = self.decomposer.decompose_question(ALPHAGO_QUESTION, question_id="q1")

        self.assertEqual(
            graph_to_edge_lines(graph),
            [
                "AlphaGo --[developed_by]--> x_company:AI_Company",
                "x_company:AI_Company --[has_ceo]--> x_ceo:Person",
                "x_ceo:Person --[graduated_from]--> x_university:University",
                "x_university:University --[located_in]--> x_city:City",
            ],
        )
        self.assertEqual([step.execution_level for step in result.execution_steps], [0, 1, 2, 3])
        self.assertEqual(
            [step.output_variable for step in result.execution_steps],
            ["x_company", "x_ceo", "x_university", "x_city"],
        )
        self.assertTrue(all(step.kind == "atomic_question" for step in result.execution_steps))

    def test_parallel_reference_example(self) -> None:
        result, graph = self.decomposer.decompose_question(PARALLEL_QUESTION, question_id="q2")

        self.assertEqual(
            [step.execution_level for step in result.execution_steps],
            [0, 0, 1, 1, 2],
        )
        self.assertEqual(result.execution_steps[-1].kind, "logical_operation")
        self.assertEqual(result.execution_steps[-1].relation_or_operator, "compare_eq")
        self.assertEqual(
            len(result.execution_steps[-1].dependencies),
            2,
        )
        self.assertIn(
            "x_nationality_1 and x_nationality_2 --[compare_eq]--> x_same_nationality:Boolean",
            graph_to_edge_lines(graph),
        )

    def test_load_question_records_supports_both_input_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            question_path = Path(temp_dir) / "questions.json"
            question_path.write_text(
                json.dumps(
                    [
                        {"id": "q1", "question": "Question A"},
                        "Question B",
                    ]
                ),
                encoding="utf-8",
            )
            records = load_question_records(question_path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].question_id, "q1")
        self.assertEqual(records[0].question, "Question A")
        self.assertIsNone(records[1].question_id)
        self.assertEqual(records[1].question, "Question B")


if __name__ == "__main__":
    unittest.main()
