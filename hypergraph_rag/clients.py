from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


EXTRACTION_SYSTEM_PROMPT = """
You extract typed query units for Hypergraph RAG decomposition.

Return JSON only. Do not use markdown.

Required JSON schema:
{
  "entities": [
    {
      "id": "stable_ascii_id",
      "label": "surface form",
      "type": "semantic type",
      "span": "exact or near-exact span from the question"
    }
  ],
  "type_variables": [
    {
      "id": "x_variable_id",
      "label": "surface form or concise variable label",
      "type": "semantic type",
      "span": "exact or near-exact span from the question",
      "is_answer": true
    }
  ],
  "relations": [
    {
      "id": "r1",
      "source_id": "entity_or_variable_id",
      "target_id": "entity_or_variable_id",
      "relation": "machine_readable_relation",
      "surface": "relation phrase from the question",
      "confidence": 0.0,
      "order_hint": 1
    }
  ],
  "operators": [
    {
      "id": "op1",
      "operator": "compare_eq|intersection|count|argmax|argmin|union|compare_gt|compare_lt",
      "input_ids": ["node_id_1", "node_id_2"],
      "output_id": "x_output",
      "output_label": "result label",
      "output_type": "Boolean",
      "description": "short description"
    }
  ],
  "branch_hints": [
    {
      "kind": "chain|parallel|merge|comparison",
      "node_ids": ["node_id"],
      "description": "short note"
    }
  ]
}

Rules:
- Extract only the units needed to answer the question.
- Each relation must be exactly one one-hop relation between adjacent nodes.
- Every relation will later become exactly one atomic sub-question.
- Do not fuse multiple relations into one relation unit.
- Comparison, aggregation, conjunction, counting, equality, intersection, or filtering must appear in operators, not in relations.
- Use concrete entities for named things and x_ variables for unknown typed targets.
- Prefer relation names such as developed_by, has_ceo, graduated_from, located_in, directed_by, has_nationality.
- Use ASCII ids and relation names.
""".strip()


ATOMIC_EDGE_SYSTEM_PROMPT = """
You generate one atomic sub-question for a single graph edge in a typed query graph.

Return JSON only. Do not use markdown.

Required JSON schema:
{
  "question": "one atomic sub-question"
}

Instruction:
"Generate exactly one atomic sub-question for this single graph edge. Do not include any other relation, condition, comparison, aggregation, or extra reasoning step. Do not answer the question."

Rules:
- Use the same language as the original question.
- The sub-question must correspond to exactly one relation edge.
- Ask for exactly one unknown target variable.
- Do not mention any operator step.
- Do not answer the question.
""".strip()


class LLMClient(Protocol):
    model: str

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        purpose: str,
    ) -> dict[str, Any]:
        ...


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 90,
        retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY or pass --api-key."
            )

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        parsed = urllib.parse.urlparse(self.base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        if path:
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        purpose: str,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._post_json(payload)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"{purpose} request failed with HTTP {exc.code}: {body}"
                )
                if exc.code == 400 and "response_format" in body:
                    payload.pop("response_format", None)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc

            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(str(last_error))

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw_payload = json.loads(response.read().decode("utf-8"))

        try:
            content = raw_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Unexpected response shape from OpenAI-compatible API: {raw_payload!r}"
            ) from exc

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        return extract_json_object(str(content))


class MockLLMClient:
    model = "mock-gpt-4o-mini"

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        purpose: str,
    ) -> dict[str, Any]:
        if purpose == "extract_query_units":
            return self._mock_extract(user_prompt)
        if purpose == "atomic_subquestion":
            return self._mock_atomic(user_prompt)
        raise RuntimeError(f"Unsupported mock purpose: {purpose}")

    def _mock_extract(self, question: str) -> dict[str, Any]:
        if "AlphaGo" in question:
            return {
                "entities": [
                    {"id": "alphago", "label": "AlphaGo", "type": "AI_System", "span": "AlphaGo"}
                ],
                "type_variables": [
                    {
                        "id": "x_company",
                        "label": "artificial intelligence company",
                        "type": "AI_Company",
                        "span": "artificial intelligence company",
                        "is_answer": False,
                    },
                    {
                        "id": "x_ceo",
                        "label": "CEO",
                        "type": "Person",
                        "span": "CEO",
                        "is_answer": False,
                    },
                    {
                        "id": "x_university",
                        "label": "university",
                        "type": "University",
                        "span": "university",
                        "is_answer": True,
                    },
                    {
                        "id": "x_city",
                        "label": "city",
                        "type": "City",
                        "span": "city",
                        "is_answer": True,
                    },
                ],
                "relations": [
                    {
                        "id": "r1",
                        "source_id": "alphago",
                        "target_id": "x_company",
                        "relation": "developed_by",
                        "surface": "company that developed AlphaGo",
                        "confidence": 0.99,
                        "order_hint": 1,
                    },
                    {
                        "id": "r2",
                        "source_id": "x_company",
                        "target_id": "x_ceo",
                        "relation": "has_ceo",
                        "surface": "CEO of the company",
                        "confidence": 0.99,
                        "order_hint": 2,
                    },
                    {
                        "id": "r3",
                        "source_id": "x_ceo",
                        "target_id": "x_university",
                        "relation": "graduated_from",
                        "surface": "graduate from",
                        "confidence": 0.99,
                        "order_hint": 3,
                    },
                    {
                        "id": "r4",
                        "source_id": "x_university",
                        "target_id": "x_city",
                        "relation": "located_in",
                        "surface": "located in",
                        "confidence": 0.99,
                        "order_hint": 4,
                    },
                ],
                "operators": [],
                "branch_hints": [{"kind": "chain", "node_ids": ["alphago"], "description": "single chain"}],
            }

        if "Ten9Eight" in question and "Sabotage" in question:
            return {
                "entities": [
                    {
                        "id": "ten9eight",
                        "label": "Ten9Eight: Shoot For The Moon",
                        "type": "Film",
                        "span": "Ten9Eight: Shoot For The Moon",
                    },
                    {
                        "id": "sabotage_1936_film",
                        "label": "Sabotage (1936 Film)",
                        "type": "Film",
                        "span": "Sabotage (1936 Film)",
                    },
                ],
                "type_variables": [
                    {
                        "id": "x_director_1",
                        "label": "director",
                        "type": "Person",
                        "span": "director of film Ten9Eight: Shoot For The Moon",
                        "is_answer": False,
                    },
                    {
                        "id": "x_nationality_1",
                        "label": "nationality",
                        "type": "Nationality",
                        "span": "same nationality",
                        "is_answer": False,
                    },
                    {
                        "id": "x_director_2",
                        "label": "director",
                        "type": "Person",
                        "span": "director of film Sabotage (1936 Film)",
                        "is_answer": False,
                    },
                    {
                        "id": "x_nationality_2",
                        "label": "nationality",
                        "type": "Nationality",
                        "span": "same nationality",
                        "is_answer": False,
                    },
                ],
                "relations": [
                    {
                        "id": "r1",
                        "source_id": "ten9eight",
                        "target_id": "x_director_1",
                        "relation": "directed_by",
                        "surface": "director of film",
                        "confidence": 0.99,
                        "order_hint": 1,
                    },
                    {
                        "id": "r2",
                        "source_id": "x_director_1",
                        "target_id": "x_nationality_1",
                        "relation": "has_nationality",
                        "surface": "share the same nationality",
                        "confidence": 0.99,
                        "order_hint": 3,
                    },
                    {
                        "id": "r3",
                        "source_id": "sabotage_1936_film",
                        "target_id": "x_director_2",
                        "relation": "directed_by",
                        "surface": "director of film",
                        "confidence": 0.99,
                        "order_hint": 2,
                    },
                    {
                        "id": "r4",
                        "source_id": "x_director_2",
                        "target_id": "x_nationality_2",
                        "relation": "has_nationality",
                        "surface": "share the same nationality",
                        "confidence": 0.99,
                        "order_hint": 4,
                    },
                ],
                "operators": [
                    {
                        "id": "compare_eq_1",
                        "operator": "compare_eq",
                        "input_ids": ["x_nationality_1", "x_nationality_2"],
                        "output_id": "x_same_nationality",
                        "output_label": "same nationality",
                        "output_type": "Boolean",
                        "description": "check whether the two nationalities are equal",
                    }
                ],
                "branch_hints": [
                    {
                        "kind": "parallel",
                        "node_ids": ["ten9eight", "sabotage_1936_film"],
                        "description": "two independent film branches",
                    },
                    {
                        "kind": "comparison",
                        "node_ids": ["x_nationality_1", "x_nationality_2"],
                        "description": "merge with equality comparison",
                    },
                ],
            }

        raise RuntimeError(
            "MockLLMClient only supports the two reference example questions."
        )

    def _mock_atomic(self, user_prompt: str) -> dict[str, Any]:
        payload = json.loads(user_prompt)
        relation = payload["relation"]
        source_label = payload["source_node"]["label"]
        source_id = payload["source_node"]["id"]

        if relation == "developed_by":
            return {"question": "Which AI company developed AlphaGo?"}
        if relation == "has_ceo":
            return {"question": "Who is the CEO of x_company?"}
        if relation == "graduated_from":
            return {"question": "Which university did x_ceo graduate from?"}
        if relation == "located_in":
            return {"question": "In which city is x_university located?"}
        if relation == "directed_by":
            if source_id == "ten9eight":
                return {"question": "Who directed Ten9Eight: Shoot For The Moon?"}
            return {"question": "Who directed Sabotage (1936 Film)?"}
        if relation == "has_nationality":
            if source_id == "x_director_1":
                return {"question": "What is the nationality of x_director_1?"}
            return {"question": "What is the nationality of x_director_2?"}

        return {"question": f"What is the {relation} of {source_label}?"}


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise ValueError("Model response did not contain a JSON object.")
        depth = 0
        in_string = False
        escape = False
        end = -1
        for index, char in enumerate(candidate[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end < 0:
            raise ValueError("Model response contained incomplete JSON.")
        parsed = json.loads(candidate[start:end])

    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed
