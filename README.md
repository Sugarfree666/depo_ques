# DEPO One-Hop Atomic Subquestion Decomposition

This project implements the DEPO pipeline described in `depo.md`.

Current architecture:

1. extract entities and type variables with placeholders and character spans
2. selectively mask only complex long entities, such as film/book/work titles, as EntityA/EntityB
3. parse the masked natural-language question with Stanford CoreNLP Enhanced++
4. align masked entity spans and preserved type-variable spans to CoreNLP tokens
5. fold each entity/type-variable span into an anchor supernode
6. build an anchor-only MST over entity/type-variable anchors
7. add only allowed AST operators
8. generate atomic subquestions from adjacent one-hop AST edges

Only selective masks are sent to CoreNLP. Type variables such as `director`, `film`, `CEO`,
`university`, `company`, and `city` remain in the parsed question so the dependency
parser keeps the natural syntactic scaffold. AST labels and generated subquestions map
EntityA/EntityB back to the original entity names.

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
