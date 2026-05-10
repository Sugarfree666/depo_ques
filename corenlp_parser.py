from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from models import CoreNLPToken, DependencyEdge, DependencyParse


class CoreNLPConnectionError(RuntimeError):
    pass


class CoreNLPParser:
    def __init__(
        self,
        url: str = "http://localhost:9000",
        timeout_ms: int = 60000,
        memory: str = "4G",
        be_quiet: bool = True,
        corenlp_home: str | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.memory = memory
        self.be_quiet = be_quiet
        self.corenlp_home = corenlp_home
        self.client: Any | None = None
        self._client_manager: Any | None = None
        self.properties = {
            "depparse.extradependencies": "MAXIMAL",
        }

    def __enter__(self) -> "CoreNLPParser":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self.client is not None:
            return
        try:
            from stanza.server import CoreNLPClient

            corenlp_home = self._resolve_corenlp_home()
            client_kwargs: dict[str, Any] = {}
            if corenlp_home is not None:
                client_kwargs["classpath"] = self._build_classpath(corenlp_home)

            self._client_manager = CoreNLPClient(
                endpoint=self.url,
                annotators="tokenize,ssplit,pos,lemma,depparse",
                output_format="json",
                properties=self.properties,
                timeout=self.timeout_ms,
                memory=self.memory,
                be_quiet=self.be_quiet,
                **client_kwargs,
            )
            self.client = self._client_manager.__enter__()
        except ModuleNotFoundError:
            raise
        except Exception as exc:
            self.client = None
            self._client_manager = None
            raise CoreNLPConnectionError(
                "CoreNLPClient could not start Stanford CoreNLP. "
                "Make sure Java is installed and CoreNLP is installed with "
                "`python -c \"import stanza; stanza.install_corenlp()\"` "
                "or pass --corenlp-home / set CORENLP_HOME to a valid CoreNLP directory. "
                f"Endpoint: {self.url}. Original error: {exc}"
            ) from exc

    def stop(self) -> None:
        if self._client_manager is None:
            return
        try:
            self._client_manager.__exit__(None, None, None)
        finally:
            self.client = None
            self._client_manager = None

    def parse(self, text: str) -> DependencyParse:
        if self.client is None:
            self.start()

        try:
            payload = self.client.annotate(text)
        except Exception as exc:
            raise CoreNLPConnectionError(
                f"CoreNLPClient failed to annotate text through endpoint {self.url}: {exc}"
            ) from exc

        payload = self._coerce_json_payload(payload)

        return self._parse_payload(payload)

    def _resolve_corenlp_home(self) -> Path | None:
        explicit_home = self.corenlp_home or os.getenv("CORENLP_HOME")
        if explicit_home:
            home = Path(explicit_home).expanduser()
            if not self._is_valid_corenlp_home(home):
                raise CoreNLPConnectionError(
                    f"CoreNLP home does not contain Stanford CoreNLP jar files: {home}"
                )
            return home

        for candidate in self._candidate_corenlp_homes():
            if self._is_valid_corenlp_home(candidate):
                return candidate
        return None

    @staticmethod
    def _candidate_corenlp_homes() -> list[Path]:
        candidates: list[Path] = []
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            cache_root = Path(local_appdata) / "StanfordNLP" / "stanza" / "Cache"
            if cache_root.exists():
                candidates.extend(sorted(cache_root.glob("*/corenlp"), reverse=True))
        candidates.append(Path.home() / "stanza_corenlp")
        return candidates

    @staticmethod
    def _is_valid_corenlp_home(path: Path) -> bool:
        return path.exists() and any(path.glob("stanford-corenlp*.jar"))

    @staticmethod
    def _build_classpath(corenlp_home: Path) -> str:
        jars = sorted(str(path) for path in corenlp_home.glob("*.jar"))
        if not jars:
            raise CoreNLPConnectionError(
                f"CoreNLP home does not contain jar files: {corenlp_home}"
            )
        return os.pathsep.join(jars)

    @staticmethod
    def _coerce_json_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except ValueError as exc:
                raise CoreNLPConnectionError("CoreNLPClient did not return valid JSON.") from exc
            if isinstance(parsed, dict):
                return parsed
        raise CoreNLPConnectionError(
            "CoreNLPClient returned an unsupported annotation payload. "
            "Expected JSON dict output from Stanza CoreNLPClient."
        )

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
                        character_offset_begin=int(token.get("characterOffsetBegin", -1)),
                        character_offset_end=int(token.get("characterOffsetEnd", -1)),
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
