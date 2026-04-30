from __future__ import annotations

import re
from functools import lru_cache

from models import NounChunk, ParserOutput, ParserToken, SurfaceLink


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?|[^\w\s]")
VERB_LEMMAS = {
    "developed": "develop",
    "develop": "develop",
    "graduated": "graduate",
    "graduate": "graduate",
    "located": "locate",
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "did": "do",
    "does": "do",
}
PREPOSITIONS = {"of", "from", "in", "on", "at", "by", "with", "for", "to"}
DETERMINERS = {"the", "a", "an", "which", "what", "who", "that", "this", "these", "those", "its"}
PUNCT = {",", ".", "?", ";", ":"}


class RuleBasedEnglishDependencyParser:
    name = "rule_based_english_parser"

    def parse(self, question: str) -> ParserOutput:
        parts = TOKEN_RE.findall(question)
        root_index = self._find_root(parts)
        tokens: list[ParserToken] = []
        for index, text in enumerate(parts):
            lower = text.lower()
            lemma = self._lemma(lower)
            pos = self._pos(lower, text)
            dep = self._dep(lower, index, root_index)
            head_index = root_index if index != root_index else index
            head_text = parts[head_index] if parts else text
            tokens.append(ParserToken(index, text, lemma, pos, dep, head_index, head_text))

        return ParserOutput(
            question=question,
            parser_name=self.name,
            tokens=tokens,
            noun_chunks=self._noun_chunks(parts),
            surface_links=detect_surface_links(question),
        )

    def _find_root(self, parts: list[str]) -> int:
        for index, token in enumerate(parts):
            if token.lower() in VERB_LEMMAS:
                return index
        return 0

    def _lemma(self, lower: str) -> str:
        if lower in VERB_LEMMAS:
            return VERB_LEMMAS[lower]
        if lower.endswith("ies") and len(lower) > 4:
            return lower[:-3] + "y"
        if lower.endswith("s") and len(lower) > 3:
            return lower[:-1]
        return lower

    def _pos(self, lower: str, text: str) -> str:
        if lower in PUNCT:
            return "PUNCT"
        if lower in DETERMINERS:
            return "DET"
        if lower in PREPOSITIONS:
            return "ADP"
        if lower in VERB_LEMMAS:
            return "VERB"
        if text[:1].isupper():
            return "PROPN"
        return "NOUN"

    def _dep(self, lower: str, index: int, root_index: int) -> str:
        if index == root_index:
            return "ROOT"
        if lower in PREPOSITIONS:
            return "prep"
        if lower in DETERMINERS:
            return "det"
        if lower in PUNCT:
            return "punct"
        return "dep"

    def _noun_chunks(self, parts: list[str]) -> list[NounChunk]:
        chunks: list[NounChunk] = []
        start: int | None = None
        current: list[str] = []
        for index, token in enumerate(parts + ["."]):
            lower = token.lower()
            is_chunk_token = lower not in PREPOSITIONS and lower not in VERB_LEMMAS and lower not in PUNCT
            if is_chunk_token:
                if start is None:
                    start = index
                current.append(token)
                continue

            if start is not None and current:
                text = " ".join(current).strip()
                words = [word for word in current if word.lower() not in DETERMINERS]
                if words:
                    chunks.append(NounChunk(text=text, start=start, end=index, root=words[-1]))
            start = None
            current = []

        return chunks


class SpacyDependencyParser:
    name = "spacy"

    def __init__(self) -> None:
        import spacy

        self.nlp = spacy.load("en_core_web_sm")

    def parse(self, question: str) -> ParserOutput:
        doc = self.nlp(question)
        tokens = [
            ParserToken(
                index=token.i,
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                dep=token.dep_,
                head_index=token.head.i,
                head_text=token.head.text,
            )
            for token in doc
        ]
        noun_chunks = [
            NounChunk(text=chunk.text, start=chunk.start, end=chunk.end, root=chunk.root.text)
            for chunk in doc.noun_chunks
        ]
        return ParserOutput(
            question=question,
            parser_name=self.name,
            tokens=tokens,
            noun_chunks=noun_chunks,
            surface_links=detect_surface_links(question),
        )


def detect_surface_links(question: str) -> list[SurfaceLink]:
    links: list[SurfaceLink] = []
    for match in re.finditer(r"\b(this|that|these|those|its)\s+([A-Za-z][A-Za-z-]*)\b", question, flags=re.I):
        determiner = match.group(1)
        noun = match.group(2)
        if noun.lower() in VERB_LEMMAS:
            continue
        links.append(SurfaceLink(text=f"{determiner} {noun}", target_text=noun, link_type="surface_coreference"))
    return links


@lru_cache(maxsize=1)
def get_default_parser() -> object:
    try:
        return SpacyDependencyParser()
    except Exception:
        return RuleBasedEnglishDependencyParser()


def parse_dependencies(question: str) -> ParserOutput:
    parser = get_default_parser()
    return parser.parse(question)
