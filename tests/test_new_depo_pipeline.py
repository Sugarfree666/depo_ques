from __future__ import annotations

import re
import unittest

import networkx as nx

from anchor_selector import AnchorSelector
from ast_builder import SemanticASTOptimizer
from graph_builder import GraphBuilder, relation_weight
from mask_span_extractor import MaskSpanExtractor
from models import (
    CoreNLPToken,
    DependencyEdge,
    DependencyParse,
    RestoredAnchorConnectedSubgraph,
    RestoredGraphNodeCandidate,
    SemanticASTEdge,
    SemanticASTNode,
    SemanticASTPrimaryOperator,
    SemanticASTResult,
)
from placeholder import selective_entity_masking
from subquestion_generator import SubquestionGenerator, compile_execution_plan


class FakeLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_prompt)
        return self.payload


class AtomicFakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        self.prompts.append(user_prompt)
        if '"step_type": "operator"' in user_prompt:
            return {"question": "Which is older, X1?"}
        if '"target_node": "age_1"' in user_prompt or '"ask": "age"' in user_prompt:
            return {"question": "What is the age of the actor?"}
        return {"question": "Who is the director of Ten9Eight: Shoot For The Moon?"}


class DAGFakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        self.prompts.append(user_prompt)
        if '"step_type": "operator"' in user_prompt:
            return {"question": "Are X2 and X4 the same?"}
        if '"known": "X1"' in user_prompt:
            return {"question": "What is the nationality of X1?"}
        if '"known": "X3"' in user_prompt:
            return {"question": "What is the nationality of X3?"}
        if '"target_node": "director_2"' in user_prompt:
            return {"question": "Who is the director of Sabotage (1936 Film)?"}
        return {"question": "Who is the director of Ten9Eight: Shoot For The Moon?"}


class BadBindingFakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        self.prompts.append(user_prompt)
        if '"step_type": "operator"' in user_prompt:
            return {"question": "What is the nationality of the director of Ten9Eight: Shoot For The Moon?"}
        if '"known": "X1"' in user_prompt:
            return {"question": "For X1, what is the nationality of director of Ten9Eight: Shoot For The Moon?"}
        return {"question": "Who is the director of Ten9Eight: Shoot For The Moon?"}


def make_tokens(question: str) -> list[CoreNLPToken]:
    pos_by_word = {
        "MovieA": "PROPN",
        "MovieB": "PROPN",
        "NetworkA": "NOUN",
        "actor": "NOUN",
        "country": "NOUN",
        "director": "NOUN",
        "film": "NOUN",
        "largest": "ADJ",
        "nationality": "NOUN",
        "older": "ADJ",
        "population": "NOUN",
        "same": "ADJ",
    }
    tokens: list[CoreNLPToken] = []
    for index, match in enumerate(re.finditer(r"[A-Za-z0-9]+|[^\w\s]", question), start=1):
        word = match.group(0)
        tokens.append(
            CoreNLPToken(
                index=index,
                word=word,
                character_offset_begin=match.start(),
                character_offset_end=match.end(),
                pos=pos_by_word.get(word),
            )
        )
    return tokens


def token_index(tokens: list[CoreNLPToken], word: str, occurrence: int = 1) -> int:
    matches = [token.index for token in tokens if token.word == word]
    return matches[occurrence - 1]


def restored_candidates_for(question: str) -> tuple[object, DependencyParse, nx.Graph, list[RestoredGraphNodeCandidate]]:
    mask_result = MaskSpanExtractor().extract(question)
    replacement = selective_entity_masking(question, mask_result)
    tokens = make_tokens(replacement.masked_question)
    parse = DependencyParse(tokens=tokens, edges=[])
    graph_builder = GraphBuilder()
    weighted_graph = graph_builder.build_weighted_dependency_graph(parse)
    graph_candidates = graph_builder.build_graph_node_candidates(parse, replacement)
    restored = graph_builder.restore_graph_node_candidates(graph_candidates, replacement)
    return replacement, parse, weighted_graph, restored


def ids_by_text(candidates: list[RestoredGraphNodeCandidate], text: str) -> list[str]:
    return [candidate.node_id for candidate in candidates if candidate.display_text == text]


class NewDEPOPipelineTests(unittest.TestCase):
    def test_complex_film_titles_restore_before_anchor_selection_and_same_is_operator(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
        mask_result = MaskSpanExtractor().extract(question)
        self.assertEqual([span.text for span in mask_result.mask_spans], [
            "Ten9Eight: Shoot For The Moon",
            "Sabotage (1936 Film)",
        ])
        replacement = selective_entity_masking(question, mask_result)
        self.assertEqual(
            replacement.masked_question,
            "Do director of film MovieA and director of film MovieB share the same nationality?",
        )

        tokens = make_tokens(replacement.masked_question)
        edges = [
            DependencyEdge("director", "nmod:of", "film", token_index(tokens, "director", 1), token_index(tokens, "film", 1)),
            DependencyEdge("film", "appos", "MovieA", token_index(tokens, "film", 1), token_index(tokens, "MovieA")),
            DependencyEdge("director", "nmod:of", "film", token_index(tokens, "director", 2), token_index(tokens, "film", 2)),
            DependencyEdge("film", "appos", "MovieB", token_index(tokens, "film", 2), token_index(tokens, "MovieB")),
            DependencyEdge("share", "nsubj", "director", token_index(tokens, "share"), token_index(tokens, "director", 1)),
            DependencyEdge("share", "nsubj", "director", token_index(tokens, "share"), token_index(tokens, "director", 2)),
            DependencyEdge("share", "obj", "nationality", token_index(tokens, "share"), token_index(tokens, "nationality")),
            DependencyEdge("nationality", "amod", "same", token_index(tokens, "nationality"), token_index(tokens, "same")),
        ]
        dependency_parse = DependencyParse(tokens=tokens, edges=edges)
        graph_builder = GraphBuilder()
        weighted_graph = graph_builder.build_weighted_dependency_graph(dependency_parse)
        graph_candidates = graph_builder.build_graph_node_candidates(dependency_parse, replacement)
        restored = graph_builder.restore_graph_node_candidates(graph_candidates, replacement)

        candidate_texts = [candidate.to_llm_view()["text"] for candidate in restored]
        self.assertIn("Ten9Eight: Shoot For The Moon", candidate_texts)
        self.assertIn("Sabotage (1936 Film)", candidate_texts)
        self.assertNotIn("MovieA [Ten9Eight: Shoot For The Moon]", candidate_texts)

        payload = {
            "selected_anchors": [
                {"node_id": ids_by_text(restored, "Ten9Eight: Shoot For The Moon")[0], "anchor_kind": "entity", "text": "Ten9Eight: Shoot For The Moon"},
                {"node_id": ids_by_text(restored, "Sabotage (1936 Film)")[0], "anchor_kind": "entity", "text": "Sabotage (1936 Film)"},
                {"node_id": ids_by_text(restored, "nationality")[0], "anchor_kind": "type_variable", "text": "nationality"},
                {"node_id": ids_by_text(restored, "same")[0], "anchor_kind": "type_variable", "text": "same"},
            ]
        }
        fake_anchor_llm = FakeLLM(payload)
        anchor_selection = AnchorSelector(fake_anchor_llm).select(
            original_question=question,
            masked_question=replacement.masked_question,
            replacement=replacement,
            dependency_parse=dependency_parse,
            weighted_graph=weighted_graph,
            restored_graph_node_candidates=restored,
        )

        selected_texts = [anchor.display_text for anchor in anchor_selection.selected_anchors]
        self.assertIn("Ten9Eight: Shoot For The Moon", selected_texts)
        self.assertIn("Sabotage (1936 Film)", selected_texts)
        self.assertIn("director", selected_texts)
        self.assertEqual(selected_texts.count("director"), 2)
        self.assertIn("nationality", selected_texts)
        self.assertNotIn("same", selected_texts)
        self.assertNotIn("Masked question", fake_anchor_llm.prompts[0])
        self.assertNotIn(replacement.masked_question, fake_anchor_llm.prompts[0])
        self.assertNotIn("Restored dependency tokens", fake_anchor_llm.prompts[0])
        self.assertNotIn("Restored dependency edges", fake_anchor_llm.prompts[0])
        self.assertNotIn("Do not select relation words", fake_anchor_llm.prompts[0])
        self.assertIn("select director in", fake_anchor_llm.prompts[0])
        self.assertIn('"text": "Do"', fake_anchor_llm.prompts[0])
        self.assertNotIn("MovieA [Ten9Eight: Shoot For The Moon]", fake_anchor_llm.prompts[0])

        connected = graph_builder.build_anchor_connected_subgraph(
            weighted_graph,
            anchor_selection.selected_anchors,
            graph_candidates,
        )
        restored_connected = graph_builder.restore_anchor_connected_subgraph(connected, replacement)
        self.assertTrue(any("Ten9Eight: Shoot For The Moon" in line for line in restored_connected.display_lines))

        ast_payload = {
            "status": "ok",
            "primary_operator": {
                "operator": "COMPARE_SAME",
                "cue_text": "same",
                "inputs": ["nationality_1", "nationality_2"],
                "output": "answer",
                "explanation": "same nationality comparison",
            },
            "nodes": [
                {"id": "movie_1", "label": "Ten9Eight: Shoot For The Moon", "kind": "entity", "semantic_type": "Film", "source": "selected_anchor", "source_graph_nodes": [ids_by_text(restored, "Ten9Eight: Shoot For The Moon")[0]], "grounding_text": "Ten9Eight: Shoot For The Moon"},
                {"id": "movie_2", "label": "Sabotage (1936 Film)", "kind": "entity", "semantic_type": "Film", "source": "selected_anchor", "source_graph_nodes": [ids_by_text(restored, "Sabotage (1936 Film)")[0]], "grounding_text": "Sabotage (1936 Film)"},
                {"id": "director_1", "label": "director", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": [ids_by_text(restored, "director")[0]], "grounding_text": "director"},
                {"id": "director_2", "label": "director", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": [ids_by_text(restored, "director")[1]], "grounding_text": "director"},
                {"id": "nationality_1", "label": "nationality", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": [ids_by_text(restored, "nationality")[0]], "grounding_text": "nationality"},
                {"id": "nationality_2", "label": "nationality", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": [ids_by_text(restored, "nationality")[0]], "grounding_text": "nationality"},
            ],
            "edges": [
                {"source": "movie_1", "target": "director_1", "edge_type": "attribute", "relation_hint": "director of film", "support_path": ["Ten9Eight: Shoot For The Moon", "film", "director"]},
                {"source": "movie_2", "target": "director_2", "edge_type": "attribute", "relation_hint": "director of film", "support_path": ["Sabotage (1936 Film)", "film", "director"]},
                {"source": "director_1", "target": "nationality_1", "edge_type": "attribute", "relation_hint": "nationality of director", "support_path": ["director", "nationality"]},
                {"source": "director_2", "target": "nationality_2", "edge_type": "attribute", "relation_hint": "nationality of director", "support_path": ["director", "nationality"]},
            ],
        }
        semantic_ast = SemanticASTOptimizer(FakeLLM(ast_payload)).optimize(
            question,
            replacement,
            anchor_selection.selected_anchors,
            restored_connected,
        )
        self.assertEqual(semantic_ast.primary_operator.operator, "COMPARE_SAME")

    def test_implicit_variable_not_selected_in_step4_but_added_in_step6(self) -> None:
        question = "Which actor is older?"
        replacement, dependency_parse, weighted_graph, restored = restored_candidates_for(question)
        payload = {
            "selected_anchors": [
                {"node_id": ids_by_text(restored, "actor")[0], "anchor_kind": "type_variable", "text": "actor"},
                {"node_id": ids_by_text(restored, "older")[0], "anchor_kind": "type_variable", "text": "older"},
                {"text": "age", "anchor_kind": "implicit_type_variable"},
            ]
        }
        selection = AnchorSelector(FakeLLM(payload)).select(
            question,
            replacement.masked_question,
            replacement,
            dependency_parse,
            weighted_graph,
            restored,
        )
        self.assertEqual([anchor.display_text for anchor in selection.selected_anchors], ["actor"])

        connected = GraphBuilder().build_anchor_connected_subgraph(weighted_graph, selection.selected_anchors, [])
        restored_connected = GraphBuilder().restore_anchor_connected_subgraph(connected, replacement)
        ast_payload = {
            "status": "ok",
            "primary_operator": {"operator": "COMPARE_GREATER", "cue_text": "older", "inputs": ["age_1"], "output": "answer", "explanation": "older compares age"},
            "nodes": [
                {"id": "actor_1", "label": "actor", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": [selection.selected_anchors[0].node_id], "grounding_text": "actor"},
                {"id": "age_1", "label": "age", "kind": "implicit_type_variable", "semantic_type": "Age", "source": "implicit_from_cue", "cue_text": "older", "grounding_text": "older"},
            ],
            "edges": [{"source": "actor_1", "target": "age_1", "edge_type": "attribute", "relation_hint": "age of actor", "support_path": ["actor", "older"]}],
        }
        semantic_ast = SemanticASTOptimizer(FakeLLM(ast_payload)).optimize(
            question,
            replacement,
            selection.selected_anchors,
            restored_connected,
        )
        self.assertEqual(semantic_ast.primary_operator.operator, "COMPARE_GREATER")
        self.assertEqual([node.label for node in semantic_ast.nodes if node.kind == "implicit_type_variable"], ["age"])

    def test_largest_population_selects_population_anchor_and_argmax_later(self) -> None:
        question = "Which country has the largest population?"
        replacement, dependency_parse, weighted_graph, restored = restored_candidates_for(question)
        payload = {
            "selected_anchors": [
                {"node_id": ids_by_text(restored, "country")[0], "anchor_kind": "type_variable", "text": "country"},
                {"node_id": ids_by_text(restored, "population")[0], "anchor_kind": "type_variable", "text": "population"},
                {"node_id": ids_by_text(restored, "largest")[0], "anchor_kind": "type_variable", "text": "largest"},
            ]
        }
        selection = AnchorSelector(FakeLLM(payload)).select(
            question,
            replacement.masked_question,
            replacement,
            dependency_parse,
            weighted_graph,
            restored,
        )
        self.assertEqual([anchor.display_text for anchor in selection.selected_anchors], ["country", "population"])

        ast_payload = {
            "status": "ok",
            "primary_operator": {"operator": "ARGMAX", "cue_text": "largest", "inputs": ["population_1"], "output": "country_1", "explanation": "largest population"},
            "nodes": [
                {"id": "country_1", "label": "country", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": [selection.selected_anchors[0].node_id], "grounding_text": "country"},
                {"id": "population_1", "label": "population", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": [selection.selected_anchors[1].node_id], "grounding_text": "population"},
            ],
            "edges": [{"source": "country_1", "target": "population_1", "edge_type": "attribute", "relation_hint": "population of country"}],
        }
        semantic_ast = SemanticASTOptimizer(FakeLLM(ast_payload)).optimize(
            question,
            replacement,
            selection.selected_anchors,
            GraphBuilder().restore_anchor_connected_subgraph(
                GraphBuilder().build_anchor_connected_subgraph(weighted_graph, selection.selected_anchors, []),
                replacement,
            ),
        )
        self.assertEqual(semantic_ast.primary_operator.operator, "ARGMAX")

    def test_multi_word_type_variable_restore_display_is_original_text(self) -> None:
        question = "What region is known for its distribution network?"
        mask_result = MaskSpanExtractor().extract(question)
        self.assertEqual([span.text for span in mask_result.mask_spans], ["distribution network"])
        replacement = selective_entity_masking(question, mask_result)
        self.assertIn("NetworkA", replacement.masked_question)

        tokens = make_tokens(replacement.masked_question)
        parse = DependencyParse(tokens=tokens, edges=[])
        graph_builder = GraphBuilder()
        candidates = graph_builder.build_graph_node_candidates(parse, replacement)
        restored = graph_builder.restore_graph_node_candidates(candidates, replacement)
        network = [candidate for candidate in restored if candidate.graph_text == "NetworkA"][0]
        self.assertEqual(network.display_text, "distribution network")
        self.assertNotEqual(network.to_llm_view()["text"], "NetworkA [distribution network]")

        payload = {"selected_anchors": [{"node_id": network.node_id, "anchor_kind": "type_variable", "text": "distribution network"}]}
        selection = AnchorSelector(FakeLLM(payload)).select(
            question,
            replacement.masked_question,
            replacement,
            parse,
            graph_builder.build_weighted_dependency_graph(parse),
            restored,
        )
        self.assertEqual(selection.selected_anchors[0].display_text, "distribution network")

    def test_mask_span_llm_uses_returned_span_without_regex_trim(self) -> None:
        question = (
            "Which university did the CEO of the artificial intelligence company that developed "
            "AlphaGo graduate from?"
        )
        span_start = question.index("the artificial intelligence company")
        payload = {
            "mask_spans": [
                {
                    "text": "the artificial intelligence company",
                    "start_char": span_start,
                    "end_char": span_start + len("the artificial intelligence company"),
                    "kind_hint": "type_variable",
                    "semantic_type_hint": "Company",
                    "reason": "multi-word type variable",
                }
            ]
        }

        mask_result = MaskSpanExtractor(FakeLLM(payload)).extract(question)
        replacement = selective_entity_masking(question, mask_result)

        self.assertEqual([span.text for span in mask_result.mask_spans], ["the artificial intelligence company"])
        self.assertEqual(
            replacement.masked_question,
            "Which university did the CEO of CompanyA that developed AlphaGo graduate from?",
        )

    def test_mask_span_llm_success_does_not_merge_heuristic_spans(self) -> None:
        question = "Who was born earlier, Elmer W. Conti or Seth Joshua?"
        span_start = question.index("Seth Joshua")
        payload = {
            "mask_spans": [
                {
                    "text": "Seth Joshua",
                    "start_char": span_start,
                    "end_char": span_start + len("Seth Joshua"),
                    "kind_hint": "entity",
                    "semantic_type_hint": "Person",
                    "reason": "LLM returned only this person name",
                }
            ]
        }

        mask_result = MaskSpanExtractor(FakeLLM(payload)).extract(question)
        replacement = selective_entity_masking(question, mask_result)

        self.assertEqual([span.text for span in mask_result.mask_spans], ["Seth Joshua"])
        self.assertEqual(
            replacement.masked_question,
            "Who was born earlier, Elmer W. Conti or PersonA?",
        )

    def test_mask_span_llm_accepts_parser_fragile_single_token_entities(self) -> None:
        question = "Which company developed AlphaGo with IBM?"
        payload = {
            "mask_spans": [
                {
                    "text": "AlphaGo",
                    "start_char": question.index("AlphaGo"),
                    "end_char": question.index("AlphaGo") + len("AlphaGo"),
                    "kind_hint": "entity",
                    "semantic_type_hint": "Product",
                    "reason": "mixed-case product name",
                },
                {
                    "text": "IBM",
                    "start_char": question.index("IBM"),
                    "end_char": question.index("IBM") + len("IBM"),
                    "kind_hint": "entity",
                    "semantic_type_hint": "Organization",
                    "reason": "acronym organization name",
                },
            ]
        }

        mask_result = MaskSpanExtractor(FakeLLM(payload)).extract(question)

        self.assertEqual([span.text for span in mask_result.mask_spans], ["AlphaGo", "IBM"])
        self.assertEqual([span.semantic_type_hint for span in mask_result.mask_spans], ["Product", "Organization"])

    def test_simple_type_variables_are_not_over_masked(self) -> None:
        question = "Which director is CEO of the university in the city and has nationality?"
        mask_result = MaskSpanExtractor().extract(question)
        self.assertEqual(mask_result.mask_spans, [])

    def test_person_name_masks_use_person_placeholders_from_human_context(self) -> None:
        question = "Who is older, Ryan Tubridy or Mauro Massironi?"

        mask_result = MaskSpanExtractor().extract(question)
        replacement = selective_entity_masking(question, mask_result)

        self.assertEqual(
            [(span.text, span.semantic_type_hint) for span in mask_result.mask_spans],
            [("Ryan Tubridy", "Person"), ("Mauro Massironi", "Person")],
        )
        self.assertEqual(replacement.masked_question, "Who is older, PersonA or PersonB?")

    def test_person_name_with_middle_initial_is_masked(self) -> None:
        question = "Who was born earlier, Elmer W. Conti or Seth Joshua?"

        mask_result = MaskSpanExtractor().extract(question)
        replacement = selective_entity_masking(question, mask_result)

        self.assertEqual(
            [(span.text, span.semantic_type_hint) for span in mask_result.mask_spans],
            [("Elmer W. Conti", "Person"), ("Seth Joshua", "Person")],
        )
        self.assertEqual(
            replacement.masked_question,
            "Who was born earlier, PersonA or PersonB?",
        )

    def test_capitalized_location_context_is_not_forced_to_person(self) -> None:
        question = "Which city is larger, New York City or Los Angeles?"

        mask_result = MaskSpanExtractor().extract(question)

        semantic_types = {span.text: span.semantic_type_hint for span in mask_result.mask_spans}
        self.assertNotEqual(semantic_types.get("Los Angeles"), "Person")

    def test_step4_validation_filters_illegal_cue_anchors(self) -> None:
        cues = ["same", "older", "largest", "and", "before"]
        candidates = [
            RestoredGraphNodeCandidate(
                node_id=str(index),
                token_index=index,
                graph_text=cue,
                restored_text=cue,
                display_text=cue,
                kind_hint="cue_candidate",
                text=cue,
            )
            for index, cue in enumerate(cues, start=1)
        ]
        payload = {
            "selected_anchors": [
                {"node_id": candidate.node_id, "anchor_kind": "type_variable", "text": candidate.display_text}
                for candidate in candidates
            ]
        }
        selection = AnchorSelector(FakeLLM(payload)).select(
            "same older largest and before",
            "same older largest and before",
            selective_entity_masking("same older largest and before", MaskSpanExtractor().extract("same older largest and before")),
            DependencyParse(tokens=[], edges=[]),
            nx.Graph(),
            candidates,
        )
        self.assertEqual(selection.selected_anchors, [])

    def test_step4_allows_relation_bearing_endpoint_nouns_but_filters_predicate_verbs(self) -> None:
        candidates = [
            RestoredGraphNodeCandidate(
                node_id="1",
                token_index=1,
                graph_text="director",
                restored_text="director",
                display_text="director",
                pos="NOUN",
                kind_hint="type_variable_candidate",
                text="director",
            ),
            RestoredGraphNodeCandidate(
                node_id="2",
                token_index=2,
                graph_text="share",
                restored_text="share",
                display_text="share",
                pos="VERB",
                kind_hint="context",
                text="share",
            ),
        ]
        payload = {
            "selected_anchors": [
                {"node_id": "1", "anchor_kind": "type_variable", "text": "director"},
                {"node_id": "2", "anchor_kind": "type_variable", "text": "share"},
            ]
        }

        question = "Do director of film A and director of film B share the same nationality?"
        selection = AnchorSelector(FakeLLM(payload)).select(
            question,
            "Do director of film MovieA and director of film MovieB share the same nationality?",
            selective_entity_masking(question, MaskSpanExtractor().extract(question)),
            DependencyParse(tokens=[], edges=[]),
            nx.Graph(),
            candidates,
        )

        self.assertEqual([anchor.display_text for anchor in selection.selected_anchors], ["director"])

    def test_mask_restore_keeps_internal_graph_text_and_llm_prompt_uses_restored_text(self) -> None:
        question = "Do director of film Ten9Eight: Shoot For The Moon share nationality?"
        mask_result = MaskSpanExtractor().extract(question)
        replacement = selective_entity_masking(question, mask_result)
        tokens = make_tokens(replacement.masked_question)
        parse = DependencyParse(tokens=tokens, edges=[])
        graph_builder = GraphBuilder()
        weighted_graph = graph_builder.build_weighted_dependency_graph(parse)
        graph_candidates = graph_builder.build_graph_node_candidates(parse, replacement)
        restored = graph_builder.restore_graph_node_candidates(graph_candidates, replacement)
        movie = [candidate for candidate in restored if candidate.placeholder == "MovieA"][0]

        self.assertEqual(movie.graph_text, "MovieA")
        self.assertEqual(movie.restored_text, "Ten9Eight: Shoot For The Moon")
        self.assertEqual(movie.display_text, "Ten9Eight: Shoot For The Moon")

        fake_llm = FakeLLM({"selected_anchors": [{"node_id": movie.node_id, "anchor_kind": "entity", "text": movie.display_text}]})
        selection = AnchorSelector(fake_llm).select(
            question,
            replacement.masked_question,
            replacement,
            parse,
            weighted_graph,
            restored,
        )
        self.assertNotIn("MovieA [Ten9Eight: Shoot For The Moon]", fake_llm.prompts[0])
        self.assertEqual(selection.selected_anchors[0].graph_text, "MovieA")
        self.assertEqual(selection.selected_anchors[0].restored_text, "Ten9Eight: Shoot For The Moon")

        connected = graph_builder.build_anchor_connected_subgraph(weighted_graph, selection.selected_anchors, graph_candidates)
        self.assertEqual(connected.selected_anchor_node_ids, [movie.node_id])

    def test_graph_weight_scheme_unchanged(self) -> None:
        self.assertEqual(relation_weight("nsubj"), 1)
        self.assertEqual(relation_weight("obj"), 1)
        self.assertEqual(relation_weight("nmod:of"), 3)
        self.assertEqual(relation_weight("compound"), 3)
        self.assertEqual(relation_weight("conj:and"), 10)
        self.assertEqual(relation_weight("det"), 5)

    def test_llm_atomic_subquestions_are_generated_per_semantic_one_hop_edge(self) -> None:
        semantic_ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(operator="NONE"),
            nodes=[
                SemanticASTNode(id="movie_1", label="Ten9Eight: Shoot For The Moon", kind="entity"),
                SemanticASTNode(id="director_1", label="director", kind="type_variable"),
            ],
            edges=[
                SemanticASTEdge(
                    source="movie_1",
                    target="director_1",
                    edge_type="attribute",
                    relation_hint="director of film",
                    support_path=["Ten9Eight: Shoot For The Moon", "director"],
                )
            ],
        )
        fake_llm = AtomicFakeLLM()
        subquestions = SubquestionGenerator(fake_llm).generate(
            "Who is the director of Ten9Eight: Shoot For The Moon?",
            semantic_ast,
        )
        self.assertEqual(len(subquestions), 1)
        self.assertEqual(subquestions[0].question, "Who is the director of Ten9Eight: Shoot For The Moon?")
        self.assertIn("Execution-plan step", fake_llm.prompts[0])
        self.assertIn('"source_node": "movie_1"', fake_llm.prompts[0])
        self.assertIn('"target_node": "director_1"', fake_llm.prompts[0])

        compare_ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(
                operator="COMPARE_GREATER",
                inputs=["age_1"],
                output="answer",
                cue_text="older",
            ),
            nodes=[
                SemanticASTNode(id="actor_1", label="actor", kind="type_variable"),
                SemanticASTNode(id="age_1", label="age", kind="implicit_type_variable", cue_text="older"),
            ],
            edges=[SemanticASTEdge(source="actor_1", target="age_1", edge_type="attribute", relation_hint="age of actor")],
        )
        fake_llm = AtomicFakeLLM()
        subquestions = SubquestionGenerator(fake_llm).generate("Which actor is older?", compare_ast)
        self.assertEqual(subquestions[0].question, "What is the age of the actor?")
        self.assertNotIn("older", subquestions[0].question.lower())
        self.assertEqual(subquestions[-1].operator, "COMPARE_GREATER")
        self.assertIn("older", subquestions[-1].question.lower())

    def test_semantic_ast_optimizer_reorients_edges_from_known_entity_to_answer_variable(self) -> None:
        question = (
            "Which university did the CEO of the artificial intelligence company that developed "
            "AlphaGo graduate from?"
        )
        replacement = selective_entity_masking(question, MaskSpanExtractor().extract(question))
        restored_subgraph = RestoredAnchorConnectedSubgraph(
            nodes=[
                {"node_id": "2", "text": "university"},
                {"node_id": "5", "text": "CEO"},
                {"node_id": "8", "text": "artificial intelligence company"},
                {"node_id": "11", "text": "AlphaGo"},
            ],
        )
        reversed_payload = {
            "status": "ok",
            "primary_operator": {"operator": "NONE", "inputs": [], "output": "university"},
            "nodes": [
                {"id": "university", "label": "university", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": ["2"]},
                {"id": "CEO", "label": "CEO", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": ["5"]},
                {"id": "CompanyA", "label": "artificial intelligence company", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": ["8"]},
                {"id": "AlphaGo", "label": "AlphaGo", "kind": "entity", "source": "selected_anchor", "source_graph_nodes": ["11"]},
            ],
            "edges": [
                {"source": "university", "target": "CEO", "edge_type": "attribute", "relation_hint": "graduated from"},
                {"source": "CEO", "target": "CompanyA", "edge_type": "attribute", "relation_hint": "CEO of"},
                {"source": "CompanyA", "target": "AlphaGo", "edge_type": "attribute", "relation_hint": "developed"},
            ],
        }

        semantic_ast = SemanticASTOptimizer(FakeLLM(reversed_payload)).optimize(
            question,
            replacement,
            [],
            restored_subgraph,
        )

        self.assertEqual(
            [(edge.source, edge.target) for edge in semantic_ast.edges],
            [("AlphaGo", "CompanyA"), ("CompanyA", "CEO"), ("CEO", "university")],
        )

    def test_semantic_ast_normalizes_node_labels_and_materializes_operator_node(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
        replacement = selective_entity_masking(question, MaskSpanExtractor().extract(question))
        restored_subgraph = RestoredAnchorConnectedSubgraph(
            nodes=[
                {"node_id": "8", "text": "Ten9Eight: Shoot For The Moon"},
                {"node_id": "9", "text": "director"},
                {"node_id": "13", "text": "nationality"},
                {"node_id": "18", "text": "Sabotage (1936 Film)"},
                {"node_id": "19", "text": "director"},
            ],
        )
        payload = {
            "status": "ok",
            "primary_operator": {
                "operator": "COMPARE_SAME",
                "cue_text": "same",
                "inputs": ["nationality_1", "nationality_2"],
                "output": "answer",
                "explanation": "same nationality comparison",
            },
            "nodes": [
                {"id": "movie_1", "label": "Ten9Eight: Shoot For The Moon", "kind": "entity", "source": "selected_anchor", "source_graph_nodes": ["8"]},
                {"id": "director_1", "label": "director_1", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": ["9"]},
                {"id": "nationality_1", "label": "nationality of director", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": ["13"]},
                {"id": "movie_2", "label": "Sabotage (1936 Film)", "kind": "entity", "source": "selected_anchor", "source_graph_nodes": ["18"]},
                {"id": "director_2", "label": "director_2", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": ["19"]},
                {"id": "nationality_2", "label": "nationality_2", "kind": "type_variable", "source": "selected_anchor", "source_graph_nodes": ["13"]},
            ],
            "edges": [
                {"source": "movie_1", "target": "director_1", "edge_type": "attribute", "relation_hint": "director of film"},
                {"source": "director_1", "target": "nationality_1", "edge_type": "attribute", "relation_hint": "nationality of director"},
                {"source": "movie_2", "target": "director_2", "edge_type": "attribute", "relation_hint": "director of film"},
                {"source": "director_2", "target": "nationality_2", "edge_type": "attribute", "relation_hint": "nationality of director"},
            ],
        }

        semantic_ast = SemanticASTOptimizer(FakeLLM(payload)).optimize(
            question,
            replacement,
            [],
            restored_subgraph,
        )

        labels = {node.id: node.label for node in semantic_ast.nodes}
        self.assertEqual(labels["director_1"], "director")
        self.assertEqual(labels["nationality_1"], "nationality")
        self.assertEqual(labels["director_2"], "director")
        self.assertEqual(labels["nationality_2"], "nationality")
        self.assertIn("operator_compare_same", labels)
        self.assertEqual(labels["operator_compare_same"], "COMPARE_SAME")
        self.assertIn(
            ("nationality_1", "operator_compare_same", "operator"),
            {(edge.source, edge.target, edge.edge_type) for edge in semantic_ast.edges},
        )
        self.assertIn(
            ("nationality_2", "operator_compare_same", "operator"),
            {(edge.source, edge.target, edge.edge_type) for edge in semantic_ast.edges},
        )

    def test_subquestion_generation_uses_dag_order_and_bound_variables(self) -> None:
        semantic_ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(
                operator="COMPARE_SAME",
                inputs=["nationality_1", "nationality_2"],
                output="answer",
                cue_text="same",
            ),
            nodes=[
                SemanticASTNode(id="movie_1", label="Ten9Eight: Shoot For The Moon", kind="entity"),
                SemanticASTNode(id="director_1", label="director", kind="type_variable"),
                SemanticASTNode(id="nationality_1", label="nationality", kind="type_variable"),
                SemanticASTNode(id="movie_2", label="Sabotage (1936 Film)", kind="entity"),
                SemanticASTNode(id="director_2", label="director", kind="type_variable"),
                SemanticASTNode(id="nationality_2", label="nationality", kind="type_variable"),
            ],
            edges=[
                SemanticASTEdge(source="movie_1", target="director_1", edge_type="attribute", relation_hint="director of film"),
                SemanticASTEdge(source="director_1", target="nationality_1", edge_type="attribute", relation_hint="nationality of director"),
                SemanticASTEdge(source="movie_2", target="director_2", edge_type="attribute", relation_hint="director of film"),
                SemanticASTEdge(source="director_2", target="nationality_2", edge_type="attribute", relation_hint="nationality of director"),
            ],
        )

        execution_plan = compile_execution_plan(semantic_ast)
        self.assertEqual(
            [(step.known, step.ask, step.answer_variable) for step in execution_plan.steps],
            [
                ("Ten9Eight: Shoot For The Moon", "director", "X1"),
                ("X1", "nationality", "X2"),
                ("Sabotage (1936 Film)", "director", "X3"),
                ("X3", "nationality", "X4"),
                ("", "", None),
            ],
        )
        self.assertEqual(execution_plan.steps[-1].step_type, "operator")
        self.assertEqual(execution_plan.steps[-1].inputs, ["X2", "X4"])

        subquestions = SubquestionGenerator(DAGFakeLLM()).generate(
            "Do the directors of Ten9Eight: Shoot For The Moon and Sabotage (1936 Film) have the same nationality?",
            semantic_ast,
        )

        self.assertEqual([item.answer_variable for item in subquestions[:4]], ["X1", "X2", "X3", "X4"])
        self.assertEqual(subquestions[1].question, "What is the nationality of X1?")
        self.assertEqual(subquestions[3].question, "What is the nationality of X3?")
        self.assertEqual(subquestions[4].question, "Are X2 and X4 the same?")

    def test_execution_plan_ignores_materialized_operator_edges_as_ordinary_steps(self) -> None:
        semantic_ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(
                operator="COMPARE_SAME",
                inputs=["nationality_1", "nationality_2"],
                output="answer",
                cue_text="same",
            ),
            nodes=[
                SemanticASTNode(id="movie_1", label="Ten9Eight: Shoot For The Moon", kind="entity"),
                SemanticASTNode(id="director_1", label="director", kind="type_variable"),
                SemanticASTNode(id="nationality_1", label="nationality", kind="type_variable"),
                SemanticASTNode(id="movie_2", label="Sabotage (1936 Film)", kind="entity"),
                SemanticASTNode(id="director_2", label="director", kind="type_variable"),
                SemanticASTNode(id="nationality_2", label="nationality", kind="type_variable"),
                SemanticASTNode(id="operator_compare_same", label="COMPARE_SAME", kind="operator"),
            ],
            edges=[
                SemanticASTEdge(source="movie_1", target="director_1", edge_type="attribute", relation_hint="director of film"),
                SemanticASTEdge(source="director_1", target="nationality_1", edge_type="attribute", relation_hint="nationality of director"),
                SemanticASTEdge(source="movie_2", target="director_2", edge_type="attribute", relation_hint="director of film"),
                SemanticASTEdge(source="director_2", target="nationality_2", edge_type="attribute", relation_hint="nationality of director"),
                SemanticASTEdge(source="nationality_1", target="operator_compare_same", edge_type="operator", relation_hint="COMPARE_SAME"),
                SemanticASTEdge(source="nationality_2", target="operator_compare_same", edge_type="operator", relation_hint="COMPARE_SAME"),
            ],
        )

        execution_plan = compile_execution_plan(semantic_ast)

        self.assertEqual([step.step_type for step in execution_plan.steps], ["edge", "edge", "edge", "edge", "operator"])
        self.assertEqual(
            [(step.source_node, step.target_node) for step in execution_plan.steps if step.step_type == "edge"],
            [
                ("movie_1", "director_1"),
                ("director_1", "nationality_1"),
                ("movie_2", "director_2"),
                ("director_2", "nationality_2"),
            ],
        )
        self.assertEqual(execution_plan.steps[-1].inputs, ["X2", "X4"])

    def test_subquestion_generation_rejects_expanded_bound_source_and_bad_operator_step(self) -> None:
        semantic_ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(
                operator="COMPARE_SAME",
                inputs=["nationality_1"],
                output="answer",
                cue_text="same",
            ),
            nodes=[
                SemanticASTNode(id="movie_1", label="Ten9Eight: Shoot For The Moon", kind="entity"),
                SemanticASTNode(id="director_1", label="director", kind="type_variable"),
                SemanticASTNode(id="nationality_1", label="nationality", kind="type_variable"),
                SemanticASTNode(id="operator_compare_same", label="COMPARE_SAME", kind="operator"),
            ],
            edges=[
                SemanticASTEdge(source="movie_1", target="director_1", edge_type="attribute", relation_hint="director of film"),
                SemanticASTEdge(source="director_1", target="nationality_1", edge_type="attribute", relation_hint="nationality of director"),
                SemanticASTEdge(source="nationality_1", target="operator_compare_same", edge_type="operator", relation_hint="COMPARE_SAME"),
            ],
        )

        subquestions = SubquestionGenerator(BadBindingFakeLLM()).generate(
            "Do the directors of Ten9Eight: Shoot For The Moon and Sabotage (1936 Film) have the same nationality?",
            semantic_ast,
        )

        self.assertEqual(subquestions[0].question, "Who is the director of Ten9Eight: Shoot For The Moon?")
        self.assertEqual(subquestions[1].question, "What is the nationality of X1?")
        self.assertEqual(subquestions[1].source, "fallback_template")
        self.assertEqual(subquestions[2].question, "Are X2 the same?")
        self.assertEqual(subquestions[2].source, "fallback_template")


if __name__ == "__main__":
    unittest.main()
