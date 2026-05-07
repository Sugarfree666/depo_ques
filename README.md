# Hypergraph RAG Query Decomposer

Python framework for decomposing complex questions into a typed query graph and strictly atomic one-hop sub-questions.

## What it does

1. Calls an OpenAI-compatible API to extract:
   - concrete entities
   - typed variables
   - one-hop relation units
   - logical operators
   - branch and merge hints
2. Builds an auxiliary dependency parse with spaCy.
3. Constructs a typed query graph with `networkx.DiGraph`.
4. Calls the LLM once per relation edge to generate exactly one atomic sub-question.
5. Produces an execution plan with dependency levels for both retrieval and logical steps.

## Project layout

- `main.py`: runnable CLI demo
- `hypergraph_rag/clients.py`: OpenAI-compatible client and mock client
- `hypergraph_rag/parsing.py`: spaCy dependency parsing
- `hypergraph_rag/query_ast.py`: typed query graph construction
- `hypergraph_rag/execution.py`: atomic question generation and execution plan
- `tests/test_pipeline.py`: simple reference tests

## Setup

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Environment variables:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

## CLI examples

Single question:

```bash
python main.py --question "Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?"
```

Batch mode:

```bash
python main.py --questions-file questions.json --output decomposed_questions.json
```

Mock demo for the two reference examples:

```bash
python main.py --mock --question "Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?"
```

## Notes

- The dependency parse is auxiliary evidence. If spaCy or the requested spaCy model is unavailable, the code falls back to a heuristic parse and records warnings in the result.
- Logical operations are emitted as `logical_operation` execution steps instead of atomic retrieval questions.
