from __future__ import annotations

import re

from models import Mention, ParserOutput, TypedClause


def normalize_dependencies(parser_output: ParserOutput, mentions: list[Mention]) -> list[TypedClause]:
    question = parser_output.question
    clauses: list[TypedClause] = []

    company = _find_mention(mentions, type_hint="Company", text_contains="company")
    ai_system = _first_constant(mentions)
    ceo = _find_mention(mentions, text_equals="CEO") or _find_mention(mentions, type_hint="Person", text_contains="CEO")
    university = _find_mention(mentions, type_hint="University", text_contains="university")
    city = _find_mention(mentions, type_hint="City", text_contains="city")

    if company and ai_system and _has_developed_relation(question, company, ai_system):
        clauses.append(
            _clause(
                clauses,
                text=_source_text(company.text, "that developed", ai_system.text),
                predicate="developed",
                clause_type="relative_clause",
                subject=company.id,
                object=ai_system.id,
            )
        )

    if ceo and company and _has_of_relation(question, ceo.text, company.text):
        clauses.append(
            _clause(
                clauses,
                text=_source_text(ceo.text, "of", company.text),
                predicate="CEO_of",
                clause_type="possessive",
                subject=ceo.id,
                object=company.id,
            )
        )

    if ceo and university and _has_graduate_from(question):
        clauses.append(
            _clause(
                clauses,
                text=_source_text(ceo.text, "graduated from", university.text),
                predicate="graduated_from",
                clause_type="prepositional_relation",
                subject=ceo.id,
                object=university.id,
            )
        )

    if university and city and _has_located_in(question):
        clauses.append(
            _clause(
                clauses,
                text=_source_text(university.text, "located in", city.text),
                predicate="located_in",
                clause_type="prepositional_relation",
                subject=university.id,
                object=city.id,
            )
        )

    clauses.extend(_coreference_clauses(parser_output, mentions, len(clauses)))
    return clauses


def _clause(
    clauses: list[TypedClause],
    text: str,
    predicate: str,
    clause_type: str,
    subject: str,
    object: str,
) -> TypedClause:
    return TypedClause(
        id=f"c{len(clauses) + 1}",
        text=text,
        predicate=predicate,
        clause_type=clause_type,
        subject=subject,
        object=object,
    )


def _coreference_clauses(
    parser_output: ParserOutput,
    mentions: list[Mention],
    start_index: int,
) -> list[TypedClause]:
    clauses: list[TypedClause] = []
    for link in parser_output.surface_links:
        target = _find_mention(mentions, text_contains=link.target_text)
        if target is None:
            continue
        clauses.append(
            TypedClause(
                id=f"c{start_index + len(clauses) + 1}",
                text=link.text,
                predicate="corefers_to",
                clause_type="coreference",
                subject=link.text,
                object=target.id,
            )
        )
    return clauses


def _find_mention(
    mentions: list[Mention],
    type_hint: str | None = None,
    text_contains: str | None = None,
    text_equals: str | None = None,
) -> Mention | None:
    for mention in mentions:
        if mention.kind == "coreference":
            continue
        if type_hint and _norm(mention.type_hint) != _norm(type_hint):
            continue
        if text_contains and _norm(text_contains) not in _norm(mention.text):
            continue
        if text_equals and _norm(mention.text) != _norm(text_equals):
            continue
        return mention
    return None


def _first_constant(mentions: list[Mention]) -> Mention | None:
    for mention in mentions:
        if mention.kind == "constant":
            return mention
    return None


def _has_developed_relation(question: str, subject: Mention, object: Mention) -> bool:
    q = _norm(question)
    object_text = re.escape(_norm(object.text))
    return bool(re.search(rf"\b(developed|develop)\s+{object_text}\b", q)) and _norm(subject.text) in q


def _has_of_relation(question: str, left_text: str, right_text: str) -> bool:
    q = _norm(question)
    left = re.escape(_norm(left_text))
    right = re.escape(_norm(right_text))
    if re.search(rf"\b{left}\s+of\s+(the\s+|that\s+|this\s+)?{right}\b", q):
        return True
    return f"{_norm(left_text)} of" in q and _norm(right_text) in q


def _has_graduate_from(question: str) -> bool:
    return bool(re.search(r"\b(graduated?|graduate)\s+from\b", _norm(question)))


def _has_located_in(question: str) -> bool:
    q = _norm(question)
    return "located in" in q or bool(re.search(r"\bin which .+ located\b", q))


def _source_text(subject: str, predicate: str, object: str) -> str:
    return f"{subject} {predicate} {object}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
