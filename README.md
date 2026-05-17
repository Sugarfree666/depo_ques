# DEPO Question Decomposition Pipeline

This project implements a DEPO-style question decomposition pipeline where the
dependency graph stays aligned to a selectively masked CoreNLP parse, while LLM
anchor decisions are made on restored original question text.

## Architecture

1. **Mask span extraction**
   Step 1 is not full anchor extraction. It only finds complex named entities
   and multi-word type/function noun phrases that should be masked to protect
   CoreNLP parsing. It does not select anchors, infer implicit variables, choose
   operators, split coordination, build an AST, or generate subquestions.

2. **Selective masking**
   Complex spans are replaced with POS-hint placeholders such as `MovieA`,
   `CompanyA`, `NetworkA`, or `TypeVarA`. Simple type variables such as
   `director`, `CEO`, `university`, `city`, and `nationality` normally remain
   in natural language.

3. **CoreNLP parse**
   CoreNLP parses the masked question. The masked placeholders remain the
   internal graph tokens so token indices and dependency node IDs stay stable.

4. **Weighted undirected dependency graph**
   The existing dependency relation weight scheme is preserved. Core relations
   such as `nsubj`, `obj`, `iobj`, `ccomp`, and `xcomp` stay low weight;
   modifiers such as `nmod`, `obl`, `amod`, and `compound` stay medium weight;
   `det`, `punct`, and coordination penalties keep their previous behavior.

5. **Restored graph node candidates**
   Before LLM anchor selection, graph node candidates are restored for display.
   The internal graph still contains placeholders like `MovieA`, but the LLM
   sees candidate text directly from the original question:

   ```json
   {"node_id": "8", "text": "Ten9Eight: Shoot For The Moon"}
   ```

   It is never rendered as `MovieA [Ten9Eight: Shoot For The Moon]`.

6. **Explicit anchor selection**
   Step 4 asks the LLM to select only explicit anchors from restored graph node
   candidates. Allowed anchor kinds are `entity` and `type_variable`. Operator
   cues and implicit variables are forbidden here, so words such as `same`,
   `older`, `largest`, `before`, `after`, `and`, and `or` are filtered by code
   validation if the LLM returns them.

7. **Anchor connected subgraph**
   Step 5 uses the selected anchor `node_id` values to return to the masked
   weighted graph and compute shortest-path evidence connecting anchors. This
   subgraph is syntactic evidence, not the final AST.

8. **Semantic AST optimization**
   Step 6 is the only stage that may add implicit type variables and choose a
   primary operator. The LLM receives the original question, selected anchors,
   restored anchor connected subgraph, mask restore information, and the fixed
   allowed operator set:

   `NONE`, `COMPARE_SAME`, `COMPARE_DIFF`, `COMPARE_GREATER`,
   `COMPARE_LESS`, `ARGMAX`, `ARGMIN`, `INTERSECTION`, `UNION`,
   `DIFFERENCE`, `LOGICAL_AND`, `LOGICAL_OR`.

   Examples: `same` maps to `COMPARE_SAME`; `older` can create an implicit
   `age` variable and choose `COMPARE_GREATER`; `largest population` chooses
   `ARGMAX`. Non-`NONE` operators are materialized as AST operator nodes, and
   branch endpoint variables point into that operator node.

9. **Execution DAG and atomic subquestion generation**
   Step 8 first compiles the final semantic AST into a deterministic execution
   DAG. This code layer decides edge order, variable bindings such as `X1` and
   `X2`, and the final operator step. Operator nodes are AST join nodes, not
   ordinary attribute hops.

   The LLM then receives only one compiled plan step at a time. For an edge
   step, it turns `known -> ask` into one atomic subquestion whose answer is the
   assigned variable. For an operator step, it applies the operator to the bound
   branch variables. The LLM is no longer allowed to see and re-plan the full AST
   during subquestion generation, which prevents multi-hop fusion and accidental
   expansion of already-bound variables.

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install Stanford CoreNLP for Stanza once:

```powershell
python -c "import stanza; stanza.install_corenlp()"
```

Run `questions.json`:

```powershell
python main.py
```

Run one question:

```powershell
python main.py --question "Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?"
```

Run with detailed intermediate output:

```powershell
python main.py --debug --question "Which actor is older?"
```

If Stanza cannot find CoreNLP, pass the CoreNLP directory:

```powershell
python main.py --corenlp-home "C:\path\to\corenlp"
```

If a managed port is occupied, choose another endpoint:

```powershell
python main.py --corenlp-url "http://localhost:9007"
```

## Tests

The unit tests use mocked `DependencyParse` objects and fake LLM clients; they
do not require a live CoreNLP server.

```powershell
python -m unittest
```
