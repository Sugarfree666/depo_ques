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

## Start Stanford CoreNLP

Download and unzip Stanford CoreNLP, then run this command from the CoreNLP directory:

```powershell
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

The default CLI URL is `http://localhost:9000`. Use `--corenlp-url` to change it.

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

