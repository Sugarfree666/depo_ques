from __future__ import annotations

import json
from typing import Any

import requests

from models import CoreNLPToken, DependencyEdge, DependencyParse


class CoreNLPConnectionError(RuntimeError):
    pass


class CoreNLPParser:
    def __init__(self, url: str = "http://localhost:9000", timeout: int = 30) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def parse(self, text: str) -> DependencyParse:
        properties = {
            "annotators": "tokenize,ssplit,pos,lemma,depparse",
            "outputFormat": "json",
            "depparse.extradependencies": "MAXIMAL",
        }
        try:
            response = requests.post(
                self.url,
                params={"properties": json.dumps(properties)},
                data=text.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise CoreNLPConnectionError(
                f"CoreNLP server is not available. Please start Stanford CoreNLP server at {self.url}."
            ) from exc

        if response.status_code != 200:
            raise CoreNLPConnectionError(
                f"CoreNLP server returned HTTP {response.status_code}. "
                f"Please check Stanford CoreNLP server at {self.url}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise CoreNLPConnectionError("CoreNLP server did not return valid JSON.") from exc

        return self._parse_payload(payload)

    def _parse_payload(self, payload: dict[str, Any]) -> DependencyParse:
        tokens: list[CoreNLPToken] = []
        edges: list[DependencyEdge] = []
        token_offset = 0

        for sentence in payload.get("sentences", []):
            local_to_global: dict[int, int] = {}
            for token in sentence.get("tokens", []):
                local_index = int(token["index"])
                global_index = token_offset + local_index
                local_to_global[local_index] = global_index
                tokens.append(
                    CoreNLPToken(
                        index=global_index,
                        word=token.get("word", ""),
                        lemma=token.get("lemma"),
                        pos=token.get("pos"),
                    )
                )

            dependencies = sentence.get("enhancedPlusPlusDependencies")
            if dependencies is None:
                raise CoreNLPConnectionError(
                    "CoreNLP response did not contain enhancedPlusPlusDependencies. "
                    "Make sure the depparse annotator is enabled."
                )

            for dep in dependencies:
                governor = int(dep.get("governor", 0))
                dependent = int(dep.get("dependent", 0))
                if governor == 0 or dependent == 0:
                    continue
                if governor not in local_to_global or dependent not in local_to_global:
                    continue
                edges.append(
                    DependencyEdge(
                        source=dep.get("governorGloss", ""),
                        relation=dep.get("dep", ""),
                        target=dep.get("dependentGloss", ""),
                        source_index=local_to_global[governor],
                        target_index=local_to_global[dependent],
                    )
                )

            token_offset += len(sentence.get("tokens", []))

        return DependencyParse(tokens=tokens, edges=edges, raw=payload)

