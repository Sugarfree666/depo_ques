# DEPO One-Hop Atomic Subquestion Decomposition

This project implements the method in `depo.md`:

1. identify entity and type-variable anchors with `gpt-4o-mini`
2. replace anchors with natural CamelCase placeholders
3. parse the placeholder question with Stanford CoreNLP Enhanced++ dependencies
4. build an anchor MST from dependency-graph shortest paths
5. build the final AST by adding only allowed operators
6. generate atomic subquestions from adjacent one-hop AST edges

The CLI prints human-readable sections and does not write complex JSON by default.

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Configure OpenAI API

Environment variables have priority over command-line values.

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

You can also pass values on the command line when the environment variables are not set:

```powershell
python main.py --api-key "your_api_key" --base-url "https://api.openai.com/v1"
```

## Install Stanford CoreNLP For Stanza

The program uses `stanza.server.CoreNLPClient` to start and stop CoreNLP automatically.
You do not need to manually run `StanfordCoreNLPServer`.

Java is still required. After installing Python dependencies, install the CoreNLP package once:

```powershell
python -c "import stanza; stanza.install_corenlp()"
```

Recent Stanza versions may install CoreNLP under a versioned cache directory such as
`C:\Users\<you>\AppData\Local\StanfordNLP\stanza\Cache\1.11.0\corenlp`.
The CLI tries to detect that cache automatically.

If you install CoreNLP manually or into a custom directory, pass `--corenlp-home` or set
`CORENLP_HOME`:

```powershell
$env:CORENLP_HOME="D:\tools\stanford-corenlp"
```

The default managed endpoint is `http://localhost:9000`. Use `--corenlp-url` only if that
port is already occupied. The server is started once when the program begins processing
questions and is closed automatically when the program exits.

## Run `questions.json`

`questions.json` may be either:

```json
["question1", "question2"]
```

or:

```json
[{"id": "q1", "question": "question1"}]
```

Run:

```powershell
python main.py
```

## Run One Manual Question

```powershell
python main.py --question "Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?"
```

## Debug Output

Default output is concise and human-readable. To include extra intermediate structures:

```powershell
python main.py --debug
```

Optional CoreNLP runtime settings:

```powershell
python main.py --corenlp-url "http://localhost:9007" --corenlp-memory 6G --corenlp-timeout-ms 120000
```

If Stanza says CoreNLP is installed but `CoreNLPClient` cannot find it, point the CLI to
the directory that contains `stanford-corenlp-*.jar` files:

```powershell
python main.py --corenlp-home "C:\Users\sugarfree\AppData\Local\StanfordNLP\stanza\Cache\1.11.0\corenlp"
```
