from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib import error, request


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
ALLOWED_LLM_STAGES = {"mention_extraction", "subquestion_verbalization"}


def parse_json_content(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON: {content}") from exc
        return json.loads(text[start : end + 1])


@dataclass
class OpenAICompatibleClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = 60
    retries: int = 2
    call_stages: list[str] = field(default_factory=list)

    def call_json(self, stage: str, messages: list[dict[str, str]], temperature: float = 0.0) -> Any:
        content = self.call_text(stage, messages, temperature=temperature, response_format={"type": "json_object"})
        return parse_json_content(content)

    def call_text(
        self,
        stage: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ) -> str:
        self._record_stage(stage)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return self._post_chat_completions(payload)

    def _record_stage(self, stage: str) -> None:
        if stage not in ALLOWED_LLM_STAGES:
            raise RuntimeError(f"LLM call is forbidden in stage: {stage}")
        self.call_stages.append(stage)

    def _post_chat_completions(self, payload: dict[str, Any]) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                return data["choices"][0]["message"]["content"]
            except error.HTTPError as exc:
                last_error = exc
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code < 500 or attempt == self.retries:
                    raise RuntimeError(f"HTTP {exc.code} from chat endpoint: {detail}") from exc
            except (error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.retries:
                    raise RuntimeError(f"Failed to call chat endpoint: {exc}") from exc

            time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Failed to call chat endpoint: {last_error}")
