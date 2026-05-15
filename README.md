# DEPO Dependency Graph Inspection

This project currently runs the front half of the DEPO pipeline so different
questions can be inspected through Stanford CoreNLP Enhanced++ dependency edges.

Current experimental architecture:

1. extract entities and type variables with placeholders and character spans
2. selectively mask only complex long entities, such as film/book/work titles, as EntityA/EntityB
3. parse the masked natural-language question with Stanford CoreNLP Enhanced++

Only selective masks are sent to CoreNLP. Type variables such as `director`, `film`, `CEO`,
`university`, `company`, and `city` remain in the parsed question so the dependency
parser keeps the natural syntactic scaffold.

The console output intentionally stops at `[3. Dependency Graph: Enhanced++]`.
Anchor graph, MST, AST, and atomic subquestion generation are not executed in
this inspection mode.

## Install

```powershell
pip install -r requirements.txt
```

Install Stanford CoreNLP for Stanza once:

```powershell
python -c "import stanza; stanza.install_corenlp()"
```

If Stanza cannot find CoreNLP, pass the directory containing `stanford-corenlp-*.jar`:

```powershell
python main.py --corenlp-home "C:\Users\sugarfree\AppData\Local\StanfordNLP\stanza\Cache\1.11.0\corenlp"
```

## Run

Run `questions.json`:

```powershell
python main.py
```

Run one question:

```powershell
python main.py --question "Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?"
```

Run with the compatibility `--debug` flag. Output still stops at dependency edges:

```powershell
python main.py --debug --question "Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?"
```

The program uses `stanza.server.CoreNLPClient` to start and stop the CoreNLP server automatically. You do not need to manually launch `StanfordCoreNLPServer`.

If a managed port is occupied, choose another endpoint:

```powershell
python main.py --corenlp-url "http://localhost:9007"
```

## Tests

The minimal tests use mocked `DependencyParse` objects and do not require a live CoreNLP server:

```powershell
python -m unittest tests.test_late_binding_graph
```
