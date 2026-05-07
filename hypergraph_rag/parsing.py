from __future__ import annotations

import re

from hypergraph_rag.models import DependencyParse, DependencyToken, SpanRecord


DEFAULT_SPACY_MODEL = "en_core_web_sm"
_FALLBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "did",
    "do",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
    "who",
}


def build_dependency_tree(
    question: str,
    model_name: str = DEFAULT_SPACY_MODEL,
) -> DependencyParse:
    try:
        import spacy
    except ImportError:
        return _fallback_dependency_tree(
            question,
            warning="spaCy is not installed. Using a heuristic tokenization fallback.",
        )

    warnings: list[str] = []
    active_model = model_name
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:
        warnings.append(
            f"Could not load spaCy model '{model_name}': {exc}. Falling back to spacy.blank('en')."
        )
        nlp = spacy.blank("en")
        active_model = "spacy.blank('en')"
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    doc = nlp(question)
    tokens = [
        DependencyToken(
            index=token.i,
            text=token.text,
            lemma=token.lemma_ or token.text.lower(),
            pos=token.pos_ or token.tag_ or "",
            dep=token.dep_ or ("ROOT" if token.i == 0 else ""),
            head=token.head.text if token.head is not None else token.text,
            head_index=token.head.i if token.head is not None else token.i,
            is_stop=bool(token.is_stop),
            start_char=token.idx,
            end_char=token.idx + len(token.text),
        )
        for token in doc
    ]

    noun_chunks: list[SpanRecord] = []
    try:
        noun_chunks = [
            SpanRecord(
                text=chunk.text,
                label="noun_chunk",
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            for chunk in doc.noun_chunks
        ]
    except Exception:
        warnings.append(
            "spaCy pipeline does not provide noun chunks. Returning an empty noun chunk list."
        )

    named_entities = [
        SpanRecord(
            text=entity.text,
            label=entity.label_,
            start_char=entity.start_char,
            end_char=entity.end_char,
        )
        for entity in getattr(doc, "ents", [])
    ]

    if not named_entities:
        named_entities = _heuristic_named_entities(question)
        if named_entities:
            warnings.append(
                "spaCy pipeline did not produce named entities. Added heuristic entity spans."
            )

    return DependencyParse(
        model_name=active_model,
        tokens=tokens,
        noun_chunks=noun_chunks,
        named_entities=named_entities,
        warnings=warnings,
    )


def _fallback_dependency_tree(question: str, warning: str) -> DependencyParse:
    tokens: list[DependencyToken] = []
    for index, match in enumerate(re.finditer(r"\S+", question)):
        text = match.group(0)
        head_index = max(index - 1, 0)
        tokens.append(
            DependencyToken(
                index=index,
                text=text,
                lemma=text.lower(),
                pos="",
                dep="ROOT" if index == 0 else "",
                head=tokens[head_index].text if tokens else text,
                head_index=head_index,
                is_stop=text.lower().strip("?,.!") in _FALLBACK_STOP_WORDS,
                start_char=match.start(),
                end_char=match.end(),
            )
        )

    return DependencyParse(
        model_name="heuristic-fallback",
        tokens=tokens,
        noun_chunks=[],
        named_entities=_heuristic_named_entities(question),
        warnings=[warning],
    )


def _heuristic_named_entities(question: str) -> list[SpanRecord]:
    entities: list[SpanRecord] = []
    pattern = re.compile(
        r"([A-Z][A-Za-z0-9]+(?:[:()\-][A-Za-z0-9]+|(?:\s+[A-Z][A-Za-z0-9:()\-]+)*)+)"
    )
    for match in pattern.finditer(question):
        text = match.group(0).strip()
        if len(text) < 2:
            continue
        entities.append(
            SpanRecord(
                text=text,
                label="HEURISTIC_ENTITY",
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return entities
