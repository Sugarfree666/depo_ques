# English Complex Question Decomposer

This repository implements a strict decomposition pipeline for English complex questions:

```text
Complex Question
-> Entity / Type Variable Recognition
-> Dependency Parsing
-> Query AST Construction
-> Entity-Type Variable Syntax Tree
-> One-hop Relation-based Atomic Subquestion Generation
```

LLM usage is restricted to two stages:

- `mention_extraction`: extract constants, variables, and answer variables.
- `subquestion_verbalization`: verbalize one program-generated `SubquestionPlan` into one English atomic question.

Dependency parsing, typed-clause normalization, Query AST construction, semantic graph construction, syntax tree construction, execution planning, validation, and quality checks are all deterministic Python code.

## Setup

The CLI uses an OpenAI-compatible `/chat/completions` endpoint:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

The parser uses spaCy with `en_core_web_sm` when available. If spaCy or the model is not installed, it falls back to a deterministic rule-based English parser so the pipeline remains testable without network access.

## Usage

Run a custom question:

```powershell
python semantic_decomposer.py --question "Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from, and in which city is that university located?"
```

Run a question from `questions.json`:

```powershell
python semantic_decomposer.py --index 0
python semantic_decomposer.py --all --limit 5 --output results.json
```

Write or disable the process log:

```powershell
python semantic_decomposer.py --index 0 --log-file logs\decomposition_0.log
python semantic_decomposer.py --index 0 --no-log
```

## Output

The CLI prints JSON with these top-level keys:

- `mentions`
- `raw_dependency_parse`
- `typed_clauses`
- `query_ast`
- `variable_syntax_tree`
- `semantic_graph`
- `subquestion_plans`
- `execution_order`
- `atomic_subquestions`
- `programmatic_quality_checks`

The quality checks are computed by Python:

```json
{
  "ast_valid": true,
  "graph_edges_match_ast_relations": true,
  "all_subquestions_reference_existing_edges": true,
  "answer_variable_reachable": true,
  "llm_used_only_for_mentions_and_verbalization": true
}
```

## Tests

Run the test suite with:

```powershell
python -m unittest discover -s tests
```
