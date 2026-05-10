from __future__ import annotations

import re
import unittest

from entity_extractor import _repair_duplicate_surface_spans
from ast_builder import ASTBuilder
from graph_builder import GraphBuilder
from models import CoreNLPToken, DependencyEdge, DependencyParse, ExtractionResult, ExtractedNode
from placeholder import replace_with_placeholders
from subquestion_generator import SubquestionGenerator


class FakeLLM:
    def __init__(self, operator: str = "NONE", attach_to: list[str] | None = None) -> None:
        self.operator = operator
        self.attach_to = attach_to or []
        self.one_hop_calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        if "operator" in user_prompt and "operators" in user_prompt:
            return {
                "operators": [
                    {
                        "operator": self.operator,
                        "attach_to": self.attach_to,
                        "explanation": "test operator",
                    }
                ]
            }
        self.one_hop_calls += 1
        return {"question": f"mock one-hop question {self.one_hop_calls}?"}


def make_tokens(question: str) -> list[CoreNLPToken]:
    tokens: list[CoreNLPToken] = []
    for index, match in enumerate(re.finditer(r"[A-Za-z0-9]+|[^\w\s]", question), start=1):
        tokens.append(
            CoreNLPToken(
                index=index,
                word=match.group(0),
                character_offset_begin=match.start(),
                character_offset_end=match.end(),
            )
        )
    return tokens


def token_index(tokens: list[CoreNLPToken], word: str, occurrence: int = 1) -> int:
    matches = [token.index for token in tokens if token.word == word]
    return matches[occurrence - 1]


def span(question: str, text: str, occurrence: int = 1) -> tuple[int, int]:
    matches = list(re.finditer(re.escape(text), question))
    match = matches[occurrence - 1]
    return match.start(), match.end()


class LateBindingGraphTests(unittest.TestCase):
    def test_duplicate_surface_span_repair_assigns_second_director(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
        first_start, first_end = span(question, "director", 1)
        nodes = [
            ExtractedNode("PersonAlpha", "director", "type_variable", "Person", first_start, first_end, occurrence=1),
            ExtractedNode("PersonBeta", "director", "type_variable", "Person", first_start, first_end, occurrence=1),
        ]

        _repair_duplicate_surface_spans(question, nodes)

        second_start, second_end = span(question, "director", 2)
        self.assertEqual((nodes[0].start, nodes[0].end, nodes[0].occurrence), (first_start, first_end, 1))
        self.assertEqual((nodes[1].start, nodes[1].end, nodes[1].occurrence), (second_start, second_end, 2))

    def test_compare_question_folds_fragmented_films_and_keeps_repeated_directors(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
        tokens = make_tokens(question)
        edges = [
            DependencyEdge("director", "nmod:of", "film", token_index(tokens, "director", 1), token_index(tokens, "film", 1)),
            DependencyEdge("film", "appos", "Ten9Eight", token_index(tokens, "film", 1), token_index(tokens, "Ten9Eight", 1)),
            DependencyEdge("director", "obj", "nationality", token_index(tokens, "director", 1), token_index(tokens, "nationality", 1)),
            DependencyEdge("director", "nmod:of", "film", token_index(tokens, "director", 2), token_index(tokens, "film", 2)),
            DependencyEdge("film", "appos", "Sabotage", token_index(tokens, "film", 2), token_index(tokens, "Sabotage", 1)),
            DependencyEdge("director", "obj", "nationality", token_index(tokens, "director", 2), token_index(tokens, "nationality", 1)),
        ]
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("FilmAlpha", "Ten9Eight: Shoot For The Moon", "entity", "Film", *span(question, "Ten9Eight: Shoot For The Moon")),
                ExtractedNode("FilmBeta", "Sabotage (1936 Film)", "entity", "Film", *span(question, "Sabotage (1936 Film)")),
            ],
            type_variables=[
                ExtractedNode("PersonAlpha", "director", "type_variable", "Person", *span(question, "director", 1), occurrence=1),
                ExtractedNode("PersonBeta", "director", "type_variable", "Person", *span(question, "director", 2), occurrence=2),
                ExtractedNode("NationalityAlpha", "nationality", "type_variable", "Nationality", *span(question, "nationality")),
            ],
        )

        anchor_graph = GraphBuilder().build_anchor_graph(DependencyParse(tokens, edges), extraction)

        self.assertEqual(set(anchor_graph.graph.nodes), {"FilmAlpha", "FilmBeta", "PersonAlpha", "PersonBeta", "NationalityAlpha"})
        self.assertNotIn(token_index(tokens, "Ten9Eight"), anchor_graph.folded_graph.nodes)
        self.assertNotIn(token_index(tokens, "Sabotage"), anchor_graph.folded_graph.nodes)
        self.assertEqual(anchor_graph.anchor_positions["PersonAlpha"], [token_index(tokens, "director", 1)])
        self.assertEqual(anchor_graph.anchor_positions["PersonBeta"], [token_index(tokens, "director", 2)])

        mst_edges = {frozenset((source, target)) for source, target in anchor_graph.graph.edges}
        self.assertIn(frozenset(("FilmAlpha", "PersonAlpha")), mst_edges)
        self.assertIn(frozenset(("FilmBeta", "PersonBeta")), mst_edges)
        self.assertIn(frozenset(("PersonAlpha", "NationalityAlpha")), mst_edges)
        self.assertIn(frozenset(("PersonBeta", "NationalityAlpha")), mst_edges)

        ast = ASTBuilder(FakeLLM("COMPARE_SAME", ["NationalityAlpha"])).build(
            question,
            extraction,
            replace_with_placeholders(question, extraction),
            anchor_graph,
        )
        self.assertTrue(ast.graph.has_edge("NationalityAlpha", "COMPARE_SAME"))

    def test_serial_question_uses_original_parse_and_generates_university_city_hops(self) -> None:
        question = (
            "Which university did the CEO of the artificial intelligence company that developed "
            "AlphaGo graduate from and in which city is this university located?"
        )
        tokens = make_tokens(question)
        edges = [
            DependencyEdge("company", "acl", "AlphaGo", token_index(tokens, "company"), token_index(tokens, "AlphaGo")),
            DependencyEdge("CEO", "nmod:of", "company", token_index(tokens, "CEO"), token_index(tokens, "company")),
            DependencyEdge("graduate", "nsubj", "CEO", token_index(tokens, "graduate"), token_index(tokens, "CEO")),
            DependencyEdge("graduate", "obl:from", "university", token_index(tokens, "graduate"), token_index(tokens, "university", 1)),
            DependencyEdge("university", "coref", "university", token_index(tokens, "university", 1), token_index(tokens, "university", 2)),
            DependencyEdge("located", "nsubj", "university", token_index(tokens, "located"), token_index(tokens, "university", 2)),
            DependencyEdge("located", "obl:in", "city", token_index(tokens, "located"), token_index(tokens, "city")),
        ]
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("EntityAlpha", "AlphaGo", "entity", "Entity", *span(question, "AlphaGo")),
            ],
            type_variables=[
                ExtractedNode("UniversityAlpha", "university", "type_variable", "University", *span(question, "university", 1), occurrence=1),
                ExtractedNode("PersonAlpha", "CEO", "type_variable", "Person", *span(question, "CEO")),
                ExtractedNode("CompanyAlpha", "the artificial intelligence company", "type_variable", "Company", *span(question, "the artificial intelligence company")),
                ExtractedNode("CityAlpha", "city", "type_variable", "City", *span(question, "city")),
            ],
        )

        anchor_graph = GraphBuilder().build_anchor_graph(DependencyParse(tokens, edges), extraction)
        self.assertTrue(all(isinstance(node, str) for node in anchor_graph.graph.nodes))
        self.assertEqual(set(anchor_graph.graph.nodes), {"EntityAlpha", "CompanyAlpha", "PersonAlpha", "UniversityAlpha", "CityAlpha"})

        ast = ASTBuilder(FakeLLM("BRIDGE")).build(
            question,
            extraction,
            replace_with_placeholders(question, extraction),
            anchor_graph,
        )
        subquestions = SubquestionGenerator(FakeLLM()).generate(question, ast, extraction)

        self.assertGreaterEqual(len(subquestions), 4)
        self.assertIn("UniversityAlpha", {item.target_node for item in subquestions})
        self.assertIn("CityAlpha", {item.target_node for item in subquestions})
        self.assertEqual([item.answer_variable for item in subquestions[:4]], ["X1", "X2", "X3", "X4"])


if __name__ == "__main__":
    unittest.main()
