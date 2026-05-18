from __future__ import annotations

import json

ALLOWED_OPERATORS = [
    "NONE",
    "COMPARE_SAME",
    "COMPARE_DIFF",
    "COMPARE_GREATER",
    "COMPARE_LESS",
    "ARGMAX",
    "ARGMIN",
    "INTERSECTION",
    "UNION",
    "DIFFERENCE",
    "LOGICAL_AND",
    "LOGICAL_OR",
]

MASK_SPAN_EXTRACTION_SYSTEM = """
You are implementing DEPO Step 1: selective complex-span masking only.
Your job is to identify spans that should be replaced by a POS-hint placeholder before CoreNLP parsing.
This step protects parser-fragile surface spans only.
Do not perform anchor extraction.
Do not output selected anchors, implicit variables, operators, relations, AST nodes, or subquestions.
Extract complex named entities, parser-fragile proper names, titles, abbreviations/acronyms, multi-word type variables, and multi-word functional noun phrases that are likely to be fragmented or mistagged by a dependency parser.
Keep simple type variables such as director, CEO, university, city, nationality, age, population, country, and actor unmasked unless they are part of a larger multi-word phrase.
Return valid JSON only.
""".strip()


def build_mask_span_extraction_prompt(question: str) -> str:
    schema = {
        "mask_spans": [
            {
                "text": "Ten9Eight: Shoot For The Moon",
                "start_char": 20,
                "end_char": 51,
                "kind_hint": "entity",
                "semantic_type_hint": "Film",
                "reason": "complex title with digit and colon",
            }
        ]
    }
    return f"""
Identify only spans that should be masked before CoreNLP parsing.

This is not anchor extraction and not question decomposition. The output is only a list of surface spans in the original question that need parser protection.

Mask these span types:
- Complex named entities with colon, parentheses, quotes, digits, hyphens, apostrophes, periods, slashes, or other special punctuation.
- Parser-fragile proper names, including person names, organization names, institution names, place names, product names, work titles, event names, abbreviations, and acronyms.
- Continuous multi-word named entities such as Ryan Tubridy, New York City, University of Southern California, or Bank of America.
- Titles and named works such as films, books, songs, albums, series, papers, statutes, and artworks; keep the whole title as one span.
- Multi-word type variables or functional noun phrases such as distribution network, artificial intelligence company, mixed-use space, local food distribution network, public research university, or chief executive officer.

Span boundary rules:
- Use exact original question character offsets, start inclusive and end exclusive.
- Return the minimal contiguous span that should become one placeholder token.
- For named entities and titles, keep the full official/name-like surface form.
- For type variables and functional noun phrases, exclude leading determiners such as the/a/an unless they are part of a named title.
- Do not include relative clauses, prepositional complements, comparison words, or coordination words unless they are part of a proper name/title.
- If two candidate spans overlap, prefer the larger coherent entity/title, or the compact functional noun phrase for type variables.

Semantic type hints:
- Choose a semantic_type_hint that preserves the original POS/semantic role for placeholder generation.
- Person names in human contexts such as who/whom/whose, older/younger, actor, CEO, director, author, player, or president should use semantic_type_hint: Person.
- Location names should use City/Country/Region/Location when the question context asks for places.
- Organizations and institutions should use Company/Organization/University/Institution when supported by the span or context.
- Named works should use Film/Book/Song/Album/Series/Work when supported by local wording.
- Multi-word type variables should use kind_hint: type_variable and a semantic_type_hint for their head class.

Do not mask simple one-word type variables by default:
director, CEO, university, city, nationality, age, population, country, actor.

Forbidden outputs:
- selected anchors
- implicit type variables
- operators
- final AST
- subquestions
- decomposition of coordination

Use exact original question character offsets, start inclusive and end exclusive.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Question:
{question}
""".strip()


ANCHOR_SELECTION_SYSTEM = """
You are implementing DEPO Step 4: explicit anchor selection.
You must select anchors only from the provided restored graph node candidates.
You will see only the original question and the restored candidate set; do not require masked text, dependency tokens, or dependency edges.
The candidate text already shows the original question text; do not ask for or use placeholder/original mixed labels.
Allowed anchor kinds are exactly: entity, type_variable.
Relation-bearing nouns can be valid type_variable anchors when they are explicit entities, roles, answer types, or attributes to solve, such as director, CEO, university, country, nationality, population, or distribution network.
For a phrase like "director of film X", the relation hint is "of film X" but the node "director" is still a valid type_variable anchor.
Do not select implicit_type_variable, operator, cue, comparative cue, superlative cue, coordination cue, logical cue, function word, or predicate-only verb.
Do not select words such as same, different, older, younger, larger, smaller, largest, highest, first, last, before, after, and, or, both, either.
Return valid JSON only.
""".strip()


def build_anchor_selection_prompt(
    original_question: str,
    restored_graph_node_candidates: list[dict[str, object]],
) -> str:
    schema = {
        "selected_anchors": [
            {
                "node_id": "8",
                "anchor_kind": "entity",
                "text": "Ten9Eight: Shoot For The Moon",
                "reason": "Film entity explicitly mentioned in the question",
            },
            {
                "node_id": "13",
                "anchor_kind": "type_variable",
                "text": "nationality",
                "reason": "Explicit attribute being compared",
            },
        ]
    }
    return f"""
Select explicit anchors for the question from restored graph node candidates.

Original question:
{original_question}

Restored graph node candidates:
{json.dumps(restored_graph_node_candidates, ensure_ascii=False, indent=2)}

Rules:
- Output node_id values from the candidate list.
- Select only explicit entity anchors and explicit type_variable anchors.
- Select explicit relation-bearing endpoint nouns when they are values to solve or compare. For example, select director in "director of film X", CEO in "CEO of company", and nationality in "same nationality".
- Do not select implicit variables. For "Which actor is older?", select actor only; do not create age here.
- Do not select operators or cues. For "same nationality", select nationality, not same.
- Do not select predicate-only verbs, function words, comparative/superlative words, or coordination words.
- Relation phrases belong later in semantic AST edge relation_hint; Step 4 selects the endpoint nodes, not relation text.
- If a candidate is a restored placeholder, use its restored text exactly in the text field.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


SEMANTIC_AST_OPTIMIZATION_SYSTEM = """
You are implementing DEPO Step 6: semantic AST optimization.
The input subgraph is an undirected syntactic/evidence subgraph with restored node text for display.
Use the original question and selected explicit anchors to build a directed semantic AST suitable for one-hop atomic subquestion generation.
The directed AST is a reasoning DAG: each edge must point from an already-known or already-bound node to the next node that should be solved.
This is the only step that may create implicit type variables and choose a primary operator.
Choose exactly one primary_operator from the allowed operator set. Use NONE when there is no comparison, superlative, set, or logical cue.
Do not invent entities that are not present in the original question or mask mapping.
Do not generate subquestions.
Return valid JSON only.
""".strip()


def build_semantic_ast_optimization_prompt(
    original_question: str,
    replacement: dict[str, object],
    selected_anchors: list[dict[str, object]],
    restored_anchor_connected_subgraph: dict[str, object],
    allowed_operators: list[str],
) -> str:
    schema = {
        "status": "ok",
        "primary_operator": {
            "operator": "COMPARE_SAME",
            "cue_text": "same",
            "inputs": ["nationality_1", "nationality_2"],
            "output": "answer",
            "explanation": "The question asks whether two nationalities are the same.",
        },
        "nodes": [
            {
                "id": "movie_1",
                "label": "Ten9Eight: Shoot For The Moon",
                "kind": "entity",
                "semantic_type": "Film",
                "source": "selected_anchor",
                "source_graph_nodes": ["8"],
                "source_token_indices": [8],
                "grounding_text": "Ten9Eight: Shoot For The Moon",
                "cue_text": "",
            }
        ],
        "edges": [
            {
                "source": "movie_1",
                "target": "director_1",
                "edge_type": "attribute",
                "relation_hint": "director of film",
                "support_path": ["Ten9Eight: Shoot For The Moon", "film", "director"],
                "support_dependency_relations": ["appos", "nmod:of"],
            }
        ],
    }
    return f"""
Optimize the restored anchor connected subgraph into a directed semantic AST.

Original question:
{original_question}

Mask restore information:
{json.dumps(replacement, ensure_ascii=False, indent=2)}

Selected explicit anchors:
{json.dumps(selected_anchors, ensure_ascii=False, indent=2)}

Restored anchor connected subgraph:
{json.dumps(restored_anchor_connected_subgraph, ensure_ascii=False, indent=2)}

Allowed primary operators:
{json.dumps(allowed_operators, ensure_ascii=False)}

Rules:
- Choose exactly one primary_operator.operator from the allowed set.
- Use NONE when no operator cue is present.
- Operator choice must be grounded in the original question cue, e.g. same -> COMPARE_SAME, different -> COMPARE_DIFF, older -> COMPARE_GREATER on age, largest/highest/most -> ARGMAX.
- You may add implicit type variables only when grounded by a cue in the original question.
- Implicit variables must include cue_text and grounding_text.
- You may split parallel branches and copy shared variables, e.g. nationality -> nationality_1 and nationality_2.
- Node id may carry branch suffixes such as director_1 or nationality_2, but node label must be clean natural-language text such as director or nationality.
- Do not put edge/relation phrases into node labels. For example, label the node nationality, not nationality of director.
- Put phrases such as director of film or nationality of director only in edge.relation_hint.
- Convert the undirected evidence graph into directed semantic edges whose direction follows inference, not surface syntax.
- Direction rule: source is a known constant/entity or a previously solved variable; target is the next variable/value to solve.
- If explicit named entities exist, start each branch from those entities and move toward answer variables or operator inputs.
- If no explicit entity exists, start from the answer candidate type or comparison subject, then move toward attributes used for constraints/operators.
- For "Which university did the CEO of the company that developed AlphaGo graduate from?", the reasoning direction is AlphaGo -> company -> CEO -> university, not university -> CEO -> company -> AlphaGo.
- For "Do film A and film B share the same nationality?", use film A -> director_1 -> nationality_1 and film B -> director_2 -> nationality_2.
- For "Which country has the largest population?", use country -> population, with ARGMAX over population.
- The primary operator will be represented as an operator node by the system; primary_operator.inputs must name the branch endpoint node ids consumed by the operator.
- Keep selected anchors unless you provide an explicit reason in the node/edge choices.
- Do not create entities that are absent from selected anchors or mask mappings.
- Do not generate atomic subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


ATOMIC_SUBQUESTION_GENERATION_SYSTEM = """
You are implementing DEPO Step 8: LLM-based atomic subquestion generation.
Generate exactly one atomic subquestion for the provided one-hop semantic AST edge, or exactly one operator step for the provided primary operator.
Use the original question and semantic AST context, but do not combine multiple AST edges into a multi-hop question.
The input edge is already oriented as source/bound node -> target node to solve.
If the source is bound to an answer variable such as X1, use that variable in the question instead of expanding the original source label.
For ordinary attribute edges, do not include operator cue words such as same, older, largest, before, or after.
Return valid JSON only.
""".strip()


ATOMIC_PLAN_STEP_SURFACE_SYSTEM = """
You are implementing DEPO Step 8 surface realization for one deterministic execution-plan step.
The semantic AST has already been compiled into a variable-bound execution DAG by code.
Do not re-plan, reorder, infer hidden hops, merge steps, or use any node not present in this single plan step.
For an edge step, generate one question whose answer is answer_variable.
Use step.known as the known subject exactly; if it is X1, X2, or another variable, the exact variable must appear in the question.
For an operator step, generate one question that applies step.operator to step.inputs exactly.
Do not include comparative/superlative/operator cue words in ordinary edge questions.
Return valid JSON only.
""".strip()


def build_atomic_plan_step_surface_prompt(
    original_question: str,
    plan_step: dict[str, object],
) -> str:
    schema = {
        "question": "What is the nationality of X1?",
        "answer_variable": "X2",
        "explanation": "This surfaces only the provided execution step.",
    }
    return f"""
Generate one atomic subquestion from this already-compiled execution-plan step.

Original question:
{original_question}

Execution-plan step:
{json.dumps(plan_step, ensure_ascii=False, indent=2)}

Rules:
- Do not infer a different step from the original question.
- Do not use the full AST or any unstated path.
- For step_type=edge, ask only for step.ask of step.known using step.relation_hint as wording guidance.
- The answer to an edge step will be step.answer_variable.
- If step.known is an answer variable such as X1, X2, or X1_nationality, that exact variable must appear in the question.
- If step.known is a variable, do not expand it back into step.known_node_label or the original entity/path.
- For ordinary edge steps, do not include operator cue words such as same, different, older, younger, largest, highest, first, before, or after.
- For step_type=operator, mention the variables in step.inputs directly and apply step.operator. Do not ask another attribute question.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


def build_atomic_subquestion_generation_prompt(
    original_question: str,
    semantic_ast: dict[str, object],
    current_edge: dict[str, object],
    source_node: dict[str, object] | None,
    target_node: dict[str, object] | None,
    primary_operator: dict[str, object],
) -> str:
    schema = {
        "question": "Who is the director of Ten9Eight: Shoot For The Moon?",
        "answer_variable": "X1",
        "explanation": "This asks only for the target node of the one-hop edge.",
    }
    return f"""
Generate one atomic subquestion for the current semantic item.

Original question:
{original_question}

Final semantic AST:
{json.dumps(semantic_ast, ensure_ascii=False, indent=2)}

Current one-hop edge or operator step:
{json.dumps(current_edge, ensure_ascii=False, indent=2)}

Source node:
{json.dumps(source_node, ensure_ascii=False, indent=2)}

Target node:
{json.dumps(target_node, ensure_ascii=False, indent=2)}

Primary operator:
{json.dumps(primary_operator, ensure_ascii=False, indent=2)}

Rules:
- For a directed one-hop edge, generate exactly one question for that edge only.
- Treat current_edge.source_display as the known subject. Treat current_edge.target_label as the value to ask for.
- If current_edge.source_display is X1, X2, or another variable, that exact variable must appear in the generated question.
- When current_edge.source_display is a variable, do not also expand it back into the original path or source label. For example, ask "What is the nationality of X1?", not "For X1, what is the nationality of the director of FilmA?"
- The answer to this subquestion will be current_edge.answer_variable.
- Do not merge this edge with another edge.
- Do not include same/older/largest/comparative/superlative cue words in ordinary attribute questions.
- For an implicit variable edge such as actor -> age, ask a normal attribute question such as "What is the age of the actor?"
- For an operator step, generate a question that applies the operator to current_edge.inputs. Mention those input variables directly.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

GLOBAL_METHOD_GUARD = """
You are implementing the DEPO method from depo.md.
CoreNLP parses a selectively masked question: complex noun phrases may be replaced by POS-hinting placeholders such as MovieA, CompanyA, or NetworkA.
Type variables and syntactic scaffold words stay in natural language.
After parsing, entity/type-variable token spans are folded into anchor supernodes on the dependency graph.
The MST is an anchor-only MST over entity/type-variable anchor nodes.
Do not do ordinary end-to-end subquestion decomposition.
Do only the current pipeline step.
Do not introduce unsupported entities or type variables; implicit attribute variables are allowed only when grounded by an explicit comparative, superlative, ordinal, or predicate cue in the original question.
Do not merge repeated variables when they have different semantic roles.
The output must be valid parseable JSON and nothing else.
"""

ENTITY_EXTRACTION_SYSTEM = (
    GLOBAL_METHOD_GUARD
    + """
Current step: identify entity nodes and type-variable nodes only.
Do not generate subquestions.
Extract minimal relation-bearing anchor nodes for dependency-graph inspection, not full descriptive noun phrases.
Include implicit type variables when the question asks about an attribute through a comparative, superlative, ordinal, or predicate word even if the attribute noun is not literally present.
When a comparative/superlative cue modifies an event or predicate, use the predicate word as the anchor text and keep the cue only in cue_text/cue_start/cue_end.
Assign each node a natural-language CamelCase placeholder in the format SemanticType + GreekOrdinal.
Use Greek ordinals in this order: Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta, Theta.
Examples: CompanyAlpha, PersonAlpha, PersonBeta, FilmAlpha, FilmBeta, NationalityAlpha.
Use EntityAlpha only when no more specific semantic type is natural.
For repeated mentions with different roles, keep separate nodes with separate placeholders.
For type variables, use the shortest surface span that still names the relation endpoint correctly.
Prefer the head role/category for organization, institution, place, person, and title endpoints.
Keep pre-head words only when they form an essential functional/common-noun term; remove field, topic, quality, domain, purpose, or scope modifiers.
Do not omit a functional/structural nominal endpoint just because it is introduced as an attributed property or predicate complement.
Return exact character spans in the original question whenever possible, using Python-style start inclusive and end exclusive.
For an implicit type variable whose text is not present in the question, set text to the semantic attribute name and use cue_text/cue_start/cue_end for the word that expresses it in the question.
These original spans will be shifted after selective masking and then aligned to CoreNLP tokens, so span accuracy is critical.
"""
)


def build_entity_extraction_prompt(question: str) -> str:
    schema = {
        "entities": [
            {
                "text": "NamedEntity",
                "semantic_type": "Entity",
                "placeholder": "EntityAlpha",
                "start": 0,
                "end": 11,
                "occurrence": 1,
            }
        ],
        "type_variables": [
            {
                "text": "company",
                "semantic_type": "Company",
                "placeholder": "CompanyAlpha",
                "start": 0,
                "end": 7,
                "occurrence": 1,
                "cue_text": "",
                "cue_start": None,
                "cue_end": None,
            }
        ],
    }
    return f"""
Extract only the core entity and type-variable nodes for this question.

Definitions:
- entity: a concrete named entity or named artifact explicitly named in the question.
- type_variable: a minimal role, title, office, answer type, object type, institution type, system, artifact, place type, or other common-noun concept that acts as an endpoint in the question's relation chain.
- implicit type_variable: an attribute endpoint that is asked through a comparative/superlative/predicate cue rather than by an explicit noun in the question. Use the semantic attribute as text and the cue word span for alignment.

Rules:
- Do not generate atomic subquestions.
- Do not output relations.
- Do not invent nodes not explicitly supported by the question text.
- Extract only relation-bearing graph anchors: named entities, answer types, roles/titles/offices, institutions, places, systems, artifacts, and object/category concepts that are endpoints of predicates, possessives, clauses, or prepositional relations.
- Always include explicit role/title/office mentions when they participate in a relation, including abbreviations and uppercase titles. Do not drop a role just because it is attached to another node by a possessive, "of", or relative-clause relation.
- Always include implicit compared or ranked attributes. For example, comparative/superlative words imply an attribute node; output that attribute as a type_variable and provide the cue word offsets.
- For event or predicate comparisons, the anchor is the predicate token, not the predicate plus comparative phrase. Keep the comparative word only as the cue.
- For each type_variable, choose the shortest contiguous span that still names the endpoint correctly. Remove determiners and nonessential adjectives.
- Include nominal predicate complements and attributed/possessed things when they are themselves functional, structural, institutional, artifact, system, place, or role endpoints in the relation chain. They remain anchors even if the surrounding clause describes a property of another node.
- For organization, institution, place, person, and role/title endpoints, the head category or title is normally the node. Remove preceding field, industry, topic, domain, quality, purpose, scope, and descriptive modifiers unless the whole phrase is a proper named entity.
- For functional or structural common-noun endpoints, keep a compact compound span only when the pre-head word changes the endpoint class and the head alone would be too vague for the relation chain. Keep only essential compound words; remove determiners, quality adjectives, clauses, and prepositional complements.
- If a word or phrase only describes, restricts, classifies, quantifies, dates, measures, or gives the topic/domain/purpose of another anchor, do not extract it as a standalone node unless the question directly asks for that value. This pruning applies to the modifiers and complements around an anchor, not to the anchor noun phrase itself.
- Do not extract objects inside modifier/complement phrases as separate nodes when they are only topical restrictions or purposes of another endpoint.
- Do not extract quantities, durations, dates, ordinals, comparative words, or measurement phrases as standalone nodes unless the question directly asks for that value as the answer. A duration or quantity that only modifies how long an action lasted is not an anchor.
- Before returning, prune every multi-word type_variable span: if removing a pre-head word leaves a valid role/category endpoint of the same relation, output the shorter span; if removing it changes a functional/structural endpoint into a vague generic noun, keep the compact compound.
- Keep duplicate role variables separate when they belong to different branches or distinct mentions with different roles.
- Preserve exact surface text from the original question.
- Return accurate start/end character offsets for each surface span in the original question.
- If the type variable is implicit and its text does not occur in the question, start/end may point to the cue word and cue_text/cue_start/cue_end must identify that same cue.
- Prefer semantic placeholders like PersonAlpha for CEO/director, FilmAlpha for films, CompanyAlpha for companies.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}

Question:
{question}
""".strip()


OPERATOR_SELECTION_SYSTEM = (
    GLOBAL_METHOD_GUARD
    + f"""
Current step: choose operators and shared-node attachments for the final AST.
The input graph is already an anchor-only semantic graph built from weighted dependency shortest paths.
You must not rewrite the anchor graph.
You must not add, remove, or reorder anchor-anchor edges.
You must not generate subquestions.
You may only choose from this fixed operator set: {", ".join(ALLOWED_OPERATORS)}.
Return JSON only.
"""
)


def build_operator_prompt(
    question: str,
    anchor_nodes: list[dict[str, str]],
    anchor_edges: list[dict[str, object]],
) -> str:
    schema = {
        "operators": [
            {
                "operator": "COMPARE_SAME",
                "attach_to": ["NationalityAlpha"],
                "explanation": "The question asks whether two branch results share the same nationality.",
            }
        ]
    }
    return f"""
Given the original question and the anchor-only semantic graph, choose only the needed operator(s) and the existing anchor node(s) they attach to.

Original question:
{question}

Anchor nodes:
{json.dumps(anchor_nodes, ensure_ascii=False, indent=2)}

Anchor semantic graph edges:
{json.dumps(anchor_edges, ensure_ascii=False, indent=2)}

Rules:
- Keep the anchor graph unchanged.
- If the graph is a simple serial bridge with no comparison, set, extremum, or logical operator, use NONE.
- If the question asks whether two branch results are the same, use COMPARE_SAME and attach it to the shared result node.
- If the question asks whether two branch results are different, use COMPARE_DIFF and attach it to the shared result node.
- Use INTERSECTION for common/shared results, UNION for either/all alternatives, and DIFFERENCE for results present in one branch but not another.
- Use COMPARE_GREATER or COMPARE_LESS for numeric/ordered comparisons.
- Use ARGMAX or ARGMIN for superlative maximum/minimum selection.
- Use LOGICAL_AND or LOGICAL_OR for explicit boolean combination conditions.
- Do not create new anchor nodes.
- Do not generate subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}
""".strip()


ONE_HOP_SUBQUESTION_SYSTEM = (
    GLOBAL_METHOD_GUARD
    + """
Current step: rewrite exactly one adjacent AST edge as one atomic subquestion.
Use only the two provided adjacent nodes and the original question.
Respect variable binding exactly: if an endpoint is an intermediate answer variable such as X1 or X2, use that variable verbatim in the generated question.
Do not use any other AST nodes.
Do not use multi-hop information.
Do not generate a sequence of subquestions.
Do not infer the complete decomposition; generate only the one subquestion for this one adjacent edge.
Return JSON only.
"""
)


def build_one_hop_prompt(
    original_question: str,
    source_display: str,
    target_display: str,
    source_original: str,
    target_original: str,
    answer_variable: str,
    edge_hint: str | None = None,
) -> str:
    schema = {"question": "Which company developed AlphaGo?"}
    hint_text = f"\nDependency/AST edge hint: {edge_hint}" if edge_hint else ""
    return f"""
Generate one atomic subquestion for exactly this one-hop AST edge.

Original question:
{original_question}

Adjacent AST edge endpoints:
- Source endpoint text to use in the subquestion, verbatim: {source_display}
- Target endpoint to ask for: {target_display}

Endpoint meanings:
- Source original node meaning only, not replacement text when source endpoint is a variable: {source_original}
- Target original node: {target_original}

The answer variable assigned by the program will be: {answer_variable}
{hint_text}

Rules:
- Only use the two endpoints above and the original question.
- If the source endpoint is an answer variable such as X1, X2, or X1_nationality, the exact variable string must appear in the question.
- When the source endpoint is a variable, do not expand it back to the source original node text.
- Do not include comparative cue words such as earlier, later, older, younger, larger, or smaller in one-hop attribute questions; those cue words belong to the final operator question.
- Ask for the target endpoint as the answer.
- Do not mention any other node from the full AST.
- Do not generate additional subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}
""".strip()
