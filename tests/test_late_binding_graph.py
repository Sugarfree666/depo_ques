from __future__ import annotations

import re
import unittest

import networkx as nx

from entity_extractor import EntityExtractor, _repair_duplicate_surface_spans
from ast_builder import ASTBuilder
from graph_builder import GraphBuilder, relation_weight
from models import AnchorGraph, CoreNLPToken, DependencyEdge, DependencyParse, ExtractionResult, ExtractedNode
from placeholder import replace_with_placeholders, selective_entity_masking
from subquestion_generator import SubquestionGenerator, _enforce_source_variable_binding


class FakeLLM:
    def __init__(self, operator: str = "NONE", attach_to: list[str] | None = None) -> None:
        self.operator = operator
        self.attach_to = attach_to or []
        self.one_hop_calls = 0
        self.operator_prompts: list[str] = []
        self.one_hop_prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        if "operator" in user_prompt and "operators" in user_prompt:
            self.operator_prompts.append(user_prompt)
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
        self.one_hop_prompts.append(user_prompt)
        return {"question": f"mock one-hop question {self.one_hop_calls}?"}


class FakeExtractorLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        return self.payload


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
    def test_dependency_edge_display_includes_token_indices(self) -> None:
        edge = DependencyEdge("share", "nsubj", "director", 4, 2)

        self.assertEqual(edge.display(), "share[4] --nsubj--> director[2]")

    def test_dependency_relation_weight_table(self) -> None:
        self.assertEqual(relation_weight("nsubj"), 1)
        self.assertEqual(relation_weight("obj"), 1)
        self.assertEqual(relation_weight("nmod:of"), 3)
        self.assertEqual(relation_weight("obl:in"), 3)
        self.assertEqual(relation_weight("compound"), 3)
        self.assertEqual(relation_weight("conj:and"), 10)
        self.assertEqual(relation_weight("det"), 5)
        self.assertEqual(relation_weight("unknown_relation"), 3)

    def test_one_hop_question_enforces_previous_answer_variable(self) -> None:
        self.assertEqual(
            _enforce_source_variable_binding(
                "Which CEO is associated with the artificial intelligence company?",
                "X1",
                "artificial intelligence company",
            ),
            "Which CEO is associated with X1?",
        )
        self.assertEqual(
            _enforce_source_variable_binding(
                "Which university did the CEO graduate from?",
                "X2",
                "CEO",
            ),
            "Which university did X2 graduate from?",
        )

    def test_extractor_infers_implicit_age_anchor_from_comparative_cue(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"
        payload = {
            "entities": [
                {"text": "Ryan Tubridy", "semantic_type": "Person", "placeholder": "PersonAlpha"},
                {"text": "Mauro Massironi", "semantic_type": "Person", "placeholder": "PersonBeta"},
            ],
            "type_variables": [],
        }

        extraction = EntityExtractor(FakeExtractorLLM(payload)).extract(question)

        age_nodes = [node for node in extraction.type_variables if node.semantic_type == "Age"]
        self.assertEqual(len(age_nodes), 1)
        self.assertEqual(age_nodes[0].text, "age")
        self.assertEqual(question[age_nodes[0].start : age_nodes[0].end], "older")
        self.assertEqual(age_nodes[0].occurrence, 0)

    def test_extractor_maps_existing_implicit_type_variable_to_cue_span(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"
        payload = {
            "entities": [
                {"text": "Ryan Tubridy", "semantic_type": "Person", "placeholder": "PersonAlpha"},
                {"text": "Mauro Massironi", "semantic_type": "Person", "placeholder": "PersonBeta"},
            ],
            "type_variables": [
                {"text": "age", "semantic_type": "Age", "placeholder": "AgeAlpha"},
            ],
        }

        extraction = EntityExtractor(FakeExtractorLLM(payload)).extract(question)

        self.assertEqual(len(extraction.type_variables), 1)
        age = extraction.type_variables[0]
        self.assertEqual(age.placeholder, "AgeAlpha")
        self.assertEqual(question[age.start : age.end], "older")
        self.assertEqual(age.occurrence, 0)

    def test_extractor_normalizes_comparative_surface_variable_to_attribute(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"
        payload = {
            "entities": [
                {"text": "Ryan Tubridy", "semantic_type": "Person", "placeholder": "PersonAlpha"},
                {"text": "Mauro Massironi", "semantic_type": "Person", "placeholder": "PersonBeta"},
            ],
            "type_variables": [
                {"text": "older", "semantic_type": "Age", "placeholder": "AgeAlpha"},
            ],
        }

        extraction = EntityExtractor(FakeExtractorLLM(payload)).extract(question)

        self.assertEqual(len(extraction.type_variables), 1)
        age = extraction.type_variables[0]
        self.assertEqual(age.text, "age")
        self.assertEqual(age.semantic_type, "Age")
        self.assertEqual(question[age.start : age.end], "older")
        self.assertEqual(age.occurrence, 0)

    def test_implicit_age_anchor_aligns_to_comparative_token_in_anchor_graph(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"
        tokens = make_tokens(question)
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("PersonAlpha", "Ryan Tubridy", "entity", "Person", *span(question, "Ryan Tubridy")),
                ExtractedNode("PersonBeta", "Mauro Massironi", "entity", "Person", *span(question, "Mauro Massironi")),
            ],
            type_variables=[
                ExtractedNode("AgeAlpha", "age", "type_variable", "Age", *span(question, "older"), occurrence=0),
            ],
        )
        edges = [
            DependencyEdge("Who", "nsubj", "older", token_index(tokens, "Who"), token_index(tokens, "older")),
            DependencyEdge("older", "punct", ",", token_index(tokens, "older"), token_index(tokens, ",")),
            DependencyEdge(",", "dep", "Tubridy", token_index(tokens, ","), token_index(tokens, "Tubridy")),
            DependencyEdge(",", "dep", "Massironi", token_index(tokens, ","), token_index(tokens, "Massironi")),
            DependencyEdge("Tubridy", "compound", "Ryan", token_index(tokens, "Tubridy"), token_index(tokens, "Ryan")),
            DependencyEdge("Massironi", "compound", "Mauro", token_index(tokens, "Massironi"), token_index(tokens, "Mauro")),
            DependencyEdge("Tubridy", "conj:or", "Massironi", token_index(tokens, "Tubridy"), token_index(tokens, "Massironi")),
        ]

        anchor_graph = GraphBuilder().build_anchor_graph(DependencyParse(tokens, edges), extraction)

        self.assertEqual(anchor_graph.anchor_positions["AgeAlpha"], [token_index(tokens, "older")])
        self.assertEqual(set(anchor_graph.graph.nodes), {"PersonAlpha", "PersonBeta", "AgeAlpha"})
        self.assertTrue(nx.has_path(anchor_graph.graph, "PersonAlpha", "AgeAlpha"))
        self.assertTrue(nx.has_path(anchor_graph.graph, "PersonBeta", "AgeAlpha"))
        semantic_edges = {frozenset(edge) for edge in anchor_graph.graph.edges}
        self.assertEqual(
            semantic_edges,
            {frozenset(("PersonAlpha", "AgeAlpha")), frozenset(("PersonBeta", "AgeAlpha"))},
        )

    def test_selective_masking_preserves_implicit_attribute_cue_span(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("PersonAlpha", "Ryan Tubridy", "entity", "Person", *span(question, "Ryan Tubridy")),
                ExtractedNode("PersonBeta", "Mauro Massironi", "entity", "Person", *span(question, "Mauro Massironi")),
            ],
            type_variables=[
                ExtractedNode("AgeAlpha", "age", "type_variable", "Age", *span(question, "older"), occurrence=0),
            ],
        )

        replacement = selective_entity_masking(question, extraction)
        age = replacement.anchor_extraction.type_variables[0] if replacement.anchor_extraction else None

        self.assertEqual(replacement.masked_question, question)
        self.assertIsNotNone(age)
        assert age is not None
        self.assertEqual(question[age.start : age.end], "older")
        self.assertEqual(replacement.preserved_type_variables[0]["text"], "age")

    def test_compare_operator_prefers_implicit_attribute_anchor_over_entities(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("PersonAlpha", "Ryan Tubridy", "entity", "Person", *span(question, "Ryan Tubridy")),
                ExtractedNode("PersonBeta", "Mauro Massironi", "entity", "Person", *span(question, "Mauro Massironi")),
            ],
            type_variables=[
                ExtractedNode("AgeAlpha", "age", "type_variable", "Age", *span(question, "older"), occurrence=0),
            ],
        )
        graph = nx.Graph()
        graph.add_node("PersonAlpha", kind="entity", text="Ryan Tubridy", semantic_type="Person", order=6)
        graph.add_node("PersonBeta", kind="entity", text="Mauro Massironi", semantic_type="Person", order=9)
        graph.add_node("AgeAlpha", kind="type_variable", text="age", semantic_type="Age", order=3)
        graph.add_edge("PersonAlpha", "AgeAlpha", weight=10)
        graph.add_edge("PersonBeta", "AgeAlpha", weight=10)
        anchor_graph = AnchorGraph(graph=graph, edges=[], anchor_positions={})

        ast = ASTBuilder(FakeLLM("COMPARE_GREATER", ["PersonAlpha", "PersonBeta"])).build(
            question,
            extraction,
            replace_with_placeholders(question, extraction),
            anchor_graph,
        )

        self.assertEqual(ast.operators[0].attach_to, ["AgeAlpha"])
        self.assertTrue(ast.graph.has_edge("AgeAlpha", "COMPARE_GREATER"))

    def test_compare_subquestions_use_direct_entity_to_implicit_attribute_hops(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("PersonAlpha", "Ryan Tubridy", "entity", "Person", *span(question, "Ryan Tubridy")),
                ExtractedNode("PersonBeta", "Mauro Massironi", "entity", "Person", *span(question, "Mauro Massironi")),
            ],
            type_variables=[
                ExtractedNode("AgeAlpha", "age", "type_variable", "Age", *span(question, "older"), occurrence=0),
            ],
        )
        graph = nx.Graph()
        graph.add_node("AgeAlpha", kind="type_variable", text="age", semantic_type="Age", order=3)
        graph.add_node("PersonAlpha", kind="entity", text="Ryan Tubridy", semantic_type="Person", order=6)
        graph.add_node("PersonBeta", kind="entity", text="Mauro Massironi", semantic_type="Person", order=9)
        graph.add_edge("PersonAlpha", "PersonBeta", weight=5, relations=["conj:or"])
        graph.add_edge("PersonAlpha", "AgeAlpha", weight=10, relations=["punct/dep"])
        anchor_graph = AnchorGraph(graph=graph, edges=[], anchor_positions={})
        ast = ASTBuilder(FakeLLM("COMPARE_GREATER", ["AgeAlpha"])).build(
            question,
            extraction,
            replace_with_placeholders(question, extraction),
            anchor_graph,
        )

        subquestions = SubquestionGenerator(FakeLLM()).generate(question, ast, extraction)

        self.assertEqual(
            [(item.source_node, item.target_node) for item in subquestions[:2]],
            [("PersonAlpha", "AgeAlpha"), ("PersonBeta", "AgeAlpha")],
        )
        self.assertEqual([item.answer_variable for item in subquestions[:2]], ["X1", "X2"])
        self.assertEqual(subquestions[-1].question, "Which is greater, X1 or X2?")

    def test_selective_masking_masks_complex_films_and_preserves_type_variables(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("FilmAlpha", "Ten9Eight: Shoot For The Moon", "entity", "Film", *span(question, "Ten9Eight: Shoot For The Moon")),
                ExtractedNode("FilmBeta", "Sabotage (1936 Film)", "entity", "Film", *span(question, "Sabotage (1936 Film)")),
            ],
            type_variables=[
                ExtractedNode("PersonAlpha", "director", "type_variable", "Person", *span(question, "director", 1), occurrence=1),
                ExtractedNode("FilmVarAlpha", "film", "type_variable", "Film", *span(question, "film", 1), occurrence=1),
                ExtractedNode("PersonBeta", "director", "type_variable", "Person", *span(question, "director", 2), occurrence=2),
                ExtractedNode("FilmVarBeta", "film", "type_variable", "Film", *span(question, "film", 2), occurrence=2),
                ExtractedNode("NationalityAlpha", "nationality", "type_variable", "Nationality", *span(question, "nationality")),
            ],
        )

        replacement = selective_entity_masking(question, extraction)

        self.assertEqual(
            replacement.masked_question,
            "Do director of film MovieA and director of film MovieB share the same nationality?",
        )
        self.assertEqual(set(replacement.mask_mapping), {"MovieA", "MovieB"})
        self.assertEqual(replacement.mask_mapping["MovieA"]["text"], "Ten9Eight: Shoot For The Moon")
        self.assertEqual(replacement.mask_mapping["MovieB"]["text"], "Sabotage (1936 Film)")

        preserved_texts = [item["text"] for item in replacement.preserved_type_variables]
        self.assertEqual(preserved_texts, ["director", "film", "director", "film", "nationality"])

        anchor_extraction = replacement.anchor_extraction
        self.assertIsNotNone(anchor_extraction)
        assert anchor_extraction is not None
        self.assertEqual([node.placeholder for node in anchor_extraction.entities], ["MovieA", "MovieB"])
        for node in anchor_extraction.entities:
            self.assertEqual(replacement.masked_question[node.start : node.end], node.placeholder)
        for node in anchor_extraction.type_variables:
            self.assertEqual(replacement.masked_question[node.start : node.end].lower(), node.text.lower())

    def test_selective_masking_masks_complex_type_phrase_with_pos_hint(self) -> None:
        question = (
            "Which university did the CEO of the artificial intelligence company that developed "
            "AlphaGo graduate from?"
        )
        extraction = ExtractionResult(
            entities=[
                ExtractedNode("EntityAlpha", "AlphaGo", "entity", "Entity", *span(question, "AlphaGo")),
            ],
            type_variables=[
                ExtractedNode("UniversityAlpha", "university", "type_variable", "University", *span(question, "university")),
                ExtractedNode("PersonAlpha", "CEO", "type_variable", "Person", *span(question, "CEO")),
                ExtractedNode("CompanyAlpha", "company", "type_variable", "Company", *span(question, "company")),
            ],
        )

        replacement = selective_entity_masking(question, extraction)

        self.assertEqual(
            replacement.masked_question,
            "Which university did the CEO of the CompanyA that developed AlphaGo graduate from?",
        )
        self.assertEqual(replacement.mask_mapping["CompanyA"]["text"], "artificial intelligence company")
        preserved_texts = [item["text"] for item in replacement.preserved_type_variables]
        self.assertEqual(preserved_texts, ["university", "CEO"])

    def test_selective_masking_masks_multi_token_type_phrase_with_pos_hint(self) -> None:
        question = (
            "What region is known for its robust distribution network for local food and has "
            "a college operating a farm for over a hundred years?"
        )
        extraction = ExtractionResult(
            type_variables=[
                ExtractedNode("RegionAlpha", "region", "type_variable", "Region", *span(question, "region")),
                ExtractedNode("NetworkAlpha", "distribution network", "type_variable", "Network", *span(question, "distribution network")),
                ExtractedNode("CollegeAlpha", "college", "type_variable", "College", *span(question, "college")),
                ExtractedNode("FarmAlpha", "farm", "type_variable", "Farm", *span(question, "farm")),
            ],
        )

        replacement = selective_entity_masking(question, extraction)

        self.assertEqual(
            replacement.masked_question,
            "What region is known for its robust NetworkA for local food and has a college operating a farm for over a hundred years?",
        )
        self.assertEqual(replacement.mask_mapping["NetworkA"]["text"], "distribution network")
        preserved_texts = [item["text"] for item in replacement.preserved_type_variables]
        self.assertEqual(preserved_texts, ["region", "college", "farm"])

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

    def test_selective_masked_compare_graph_maps_ast_labels_back_to_original_entities(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
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
        replacement = selective_entity_masking(question, extraction)
        anchor_extraction = replacement.anchor_extraction
        self.assertIsNotNone(anchor_extraction)
        assert anchor_extraction is not None

        tokens = make_tokens(replacement.masked_question)
        edges = [
            DependencyEdge("director", "nmod:of", "film", token_index(tokens, "director", 1), token_index(tokens, "film", 1)),
            DependencyEdge("film", "appos", "MovieA", token_index(tokens, "film", 1), token_index(tokens, "MovieA")),
            DependencyEdge("director", "obj", "nationality", token_index(tokens, "director", 1), token_index(tokens, "nationality")),
            DependencyEdge("director", "nmod:of", "film", token_index(tokens, "director", 2), token_index(tokens, "film", 2)),
            DependencyEdge("film", "appos", "MovieB", token_index(tokens, "film", 2), token_index(tokens, "MovieB")),
            DependencyEdge("director", "obj", "nationality", token_index(tokens, "director", 2), token_index(tokens, "nationality")),
        ]

        anchor_graph = GraphBuilder().build_anchor_graph(
            DependencyParse(tokens, edges),
            anchor_extraction,
        )
        self.assertEqual(set(anchor_graph.graph.nodes), {"MovieA", "MovieB", "PersonAlpha", "PersonBeta", "NationalityAlpha"})

        fake_llm = FakeLLM("COMPARE_SAME", ["NationalityAlpha"])
        ast = ASTBuilder(fake_llm).build(question, anchor_extraction, replacement, anchor_graph)

        self.assertEqual(ast.display_label("MovieA"), "Ten9Eight: Shoot For The Moon")
        self.assertEqual(ast.display_label("MovieB"), "Sabotage (1936 Film)")
        self.assertTrue(ast.graph.has_edge("NationalityAlpha", "COMPARE_SAME"))

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
        self.assertTrue(all(isinstance(node, str) for node in anchor_graph.graph.nodes))
        self.assertIn(token_index(tokens, "Ten9Eight"), anchor_graph.anchor_subgraph.nodes)
        self.assertIn(token_index(tokens, "Sabotage"), anchor_graph.anchor_subgraph.nodes)
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

    def test_selective_masking_preserves_serial_question_types_and_ast_path(self) -> None:
        question = (
            "Which university did the CEO of the artificial intelligence company that developed "
            "AlphaGo graduate from and in which city is this university located?"
        )
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

        replacement = selective_entity_masking(question, extraction)
        self.assertEqual(
            replacement.masked_question,
            "Which university did the CEO of the CompanyA that developed AlphaGo graduate from and in which city is this university located?",
        )
        self.assertEqual(set(replacement.mask_mapping), {"CompanyA"})
        preserved_texts = {item["text"] for item in replacement.preserved_type_variables}
        self.assertEqual(preserved_texts, {"university", "CEO", "city"})

        anchor_extraction = replacement.anchor_extraction
        self.assertIsNotNone(anchor_extraction)
        assert anchor_extraction is not None
        tokens = make_tokens(replacement.masked_question)
        edges = [
            DependencyEdge("CompanyA", "acl", "AlphaGo", token_index(tokens, "CompanyA"), token_index(tokens, "AlphaGo")),
            DependencyEdge("CEO", "nmod:of", "CompanyA", token_index(tokens, "CEO"), token_index(tokens, "CompanyA")),
            DependencyEdge("graduate", "nsubj", "CEO", token_index(tokens, "graduate"), token_index(tokens, "CEO")),
            DependencyEdge("graduate", "obl:from", "university", token_index(tokens, "graduate"), token_index(tokens, "university", 1)),
            DependencyEdge("university", "coref", "university", token_index(tokens, "university", 1), token_index(tokens, "university", 2)),
            DependencyEdge("located", "nsubj", "university", token_index(tokens, "located"), token_index(tokens, "university", 2)),
            DependencyEdge("located", "obl:in", "city", token_index(tokens, "located"), token_index(tokens, "city")),
        ]

        anchor_graph = GraphBuilder().build_anchor_graph(DependencyParse(tokens, edges), anchor_extraction)
        ast = ASTBuilder(FakeLLM("BRIDGE")).build(question, anchor_extraction, replacement, anchor_graph)
        path = nx.shortest_path(ast.graph, "EntityAlpha", "CityAlpha")

        self.assertEqual(
            [ast.display_label(node) for node in path],
            ["AlphaGo", "artificial intelligence company", "CEO", "university", "city"],
        )

        one_hop_llm = FakeLLM()
        SubquestionGenerator(one_hop_llm).generate(question, ast, anchor_extraction)
        joined_prompts = "\n".join(one_hop_llm.one_hop_prompts)
        self.assertIn("AlphaGo", joined_prompts)
        self.assertIn("the artificial intelligence company", joined_prompts)

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
