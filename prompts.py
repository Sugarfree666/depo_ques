from __future__ import annotations

import json

ALLOWED_OPERATORS = [
    "COMPARE_SAME",
    "COMPARE_DIFFERENT",
    "AND",
    "OR",
    "FILTER",
    "COUNT",
    "BRIDGE",
    "NONE",
]

GLOBAL_METHOD_GUARD = """
You are implementing the DEPO method from depo.md.
CoreNLP parses the original unmodified question; placeholders are display/anchor labels only.
After parsing, entity/type-variable token spans are folded into placeholder supernodes on the dependency graph.
The MST is an anchor-only MST over entity/type-variable placeholder nodes.
Do not do ordinary end-to-end subquestion decomposition.
Do only the current pipeline step.
Do not introduce entities or type variables that are not present in the original question.
Do not merge repeated variables when they have different semantic roles.
The output must be valid parseable JSON and nothing else.
"""

ENTITY_EXTRACTION_SYSTEM = (
    GLOBAL_METHOD_GUARD
    + """
Current step: identify entity nodes and type-variable nodes only.
Do not generate subquestions.
Assign each node a natural-language CamelCase placeholder in the format SemanticType + GreekOrdinal.
Use Greek ordinals in this order: Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta, Theta.
Examples: CompanyAlpha, PersonAlpha, PersonBeta, FilmAlpha, FilmBeta, NationalityAlpha.
Use EntityAlpha only when no more specific semantic type is natural.
For repeated mentions with different roles, keep separate nodes with separate placeholders.
Return exact character spans in the original question whenever possible, using Python-style start inclusive and end exclusive.
These spans will be aligned to CoreNLP tokens after the original question is parsed, so span accuracy is critical.
"""
)


def build_entity_extraction_prompt(question: str) -> str:
    schema = {
        "entities": [
            {
                "text": "AlphaGo",
                "semantic_type": "Entity",
                "placeholder": "EntityAlpha",
                "start": 0,
                "end": 7,
                "occurrence": 1,
            }
        ],
        "type_variables": [
            {
                "text": "the artificial intelligence company",
                "semantic_type": "Company",
                "placeholder": "CompanyAlpha",
                "start": 0,
                "end": 35,
                "occurrence": 1,
            }
        ],
    }
    return f"""
Extract only the core entity and type-variable nodes for this question.

Definitions:
- entity: a concrete named entity or named artifact, such as AlphaGo, Ten9Eight: Shoot For The Moon, Sabotage (1936 Film).
- type_variable: a category, role, concept, or answer type in the question, such as CEO, director, university, city, nationality.

Rules:
- Do not generate atomic subquestions.
- Do not output relations.
- Do not invent nodes not explicitly supported by the question text.
- Keep duplicate role variables separate when they belong to different branches, such as two directors in a comparison.
- Preserve exact surface text from the original question.
- Return accurate start/end character offsets for each surface span in the original question.
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
The input graph is already an anchor-only MST produced after dependency-graph subtree folding.
You must not rewrite the anchor MST.
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
Given the original question and the anchor MST, choose only the needed operator(s) and the existing anchor node(s) they attach to.

Original question:
{question}

Anchor nodes:
{json.dumps(anchor_nodes, ensure_ascii=False, indent=2)}

Anchor MST edges:
{json.dumps(anchor_edges, ensure_ascii=False, indent=2)}

Rules:
- Keep the anchor MST unchanged.
- If the graph is a simple serial bridge, use BRIDGE or NONE.
- If the question asks whether two branch results are the same, use COMPARE_SAME and attach it to the shared result node.
- If the question asks whether two branch results are different, use COMPARE_DIFFERENT and attach it to the shared result node.
- For count questions, use COUNT and attach it to the counted result node.
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
- Source endpoint to mention in the subquestion: {source_display}
- Target endpoint to ask for: {target_display}

Endpoint meanings:
- Source original node: {source_original}
- Target original node: {target_original}

The answer variable assigned by the program will be: {answer_variable}
{hint_text}

Rules:
- Only use the two endpoints above and the original question.
- If the source endpoint is an answer variable such as X1, use that variable in the question.
- Ask for the target endpoint as the answer.
- Do not mention any other node from the full AST.
- Do not generate additional subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}
""".strip()
