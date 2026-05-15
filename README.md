# DEPO One-Hop Atomic Subquestion Decomposition

This project implements a DEPO-style decomposition pipeline for complex questions.

Current architecture:

1. extract entities and type variables with placeholders and character spans
2. selectively mask complex noun phrases with POS-hinting placeholders such as `MovieA`, `CompanyA`, or `NetworkA`
3. parse the masked natural-language question with Stanford CoreNLP Enhanced++
4. convert the dependency graph to a weighted undirected graph
5. build an anchor shortest-path subgraph with Dijkstra paths
6. collapse to an anchor-only semantic graph
7. choose AST operators with the LLM
8. generate adjacent one-hop atomic subquestions with the LLM

Only selective masks are sent to CoreNLP. Simple anchors such as `director`,
`CEO`, `university`, `city`, and `nationality` remain in the parsed question so
the dependency parser keeps the natural syntactic scaffold.

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

Run with debug output:

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
