# Hypergraph RAG Semantic Decomposer

This repository contains a small CLI prototype for validating entity/type-variable based decomposition of complex questions.

It reads a complex question, asks `gpt-4o-mini` to build a compiler-style Query AST and one-hop semantic graph, then asks the model to generate atomic subquestions from graph edges.

## Configuration

Set the API key and optional OpenAI-compatible base URL:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

You can also pass them directly:

```powershell
python semantic_decomposer.py --api-key "your_api_key" --base-url "https://api.openai.com/v1"
```

## Usage

Process the first question in `questions.json`:

```powershell
python semantic_decomposer.py
```

Process a specific question by 0-based index:

```powershell
python semantic_decomposer.py --index 3
```

Process every question, or a limited batch:

```powershell
python semantic_decomposer.py --all --output results.json
python semantic_decomposer.py --all --limit 5 --output sample_results.json
```

Process a custom question:

```powershell
python semantic_decomposer.py --question "研发了 AlphaGo 的那家人工智能公司的 CEO 毕业于哪所大学，这所大学位于哪座城市？"
```

Write the detailed process log to a custom file:

```powershell
python semantic_decomposer.py --index 0 --log-file logs\decomposition_0.log
```

Disable the process log:

```powershell
python semantic_decomposer.py --index 0 --no-log
```

## Output

Console output is concise:

```text
原始问题：研发了 AlphaGo 的那家人工智能公司的 CEO 毕业于哪所大学，这所大学位于哪座城市？
关系语义图：AlphaGo<-人工智能公司->CEO->大学->城市
分解后的子问题：
1. 哪家人工智能公司研发了 AlphaGo？
2. 这家人工智能公司的 CEO 是谁？
3. 这位 CEO 毕业于哪所大学？
4. 这所大学位于哪座城市？
```

When `--output` is provided, the file still receives the full JSON result, including:

- `variables`: extracted entities and type variables.
- `dependency_evidence`: relation evidence from dependency-style parsing.
- `query_ast`: the generated query syntax tree.
- `semantic_graph`: nodes and one-hop relation edges.
- `atomic_subquestions`: one-hop subquestions with placeholders and dependencies.
- `execution_order`: suggested retrieval order.

By default, `decomposition.log` records the full decomposition process:

- run configuration without the API key.
- original question.
- stage 1 request and response for entity/type variables, Query AST, and semantic graph.
- stage 2 request and response for one-hop atomic subquestions.
- compact semantic graph shown in the console.
- merged final JSON result.
