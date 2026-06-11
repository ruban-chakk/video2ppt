from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .mermaid import sanitize_mermaid


OFF_TOPIC_PATTERNS = [
    r"\blike and subscribe\b",
    r"\bsubscribe\b",
    r"\bhit the bell\b",
    r"\bfollow me\b",
    r"\bmy channel\b",
    r"\bsponsor(ed)?\b",
    r"\bhousekeeping\b",
    r"\bbreak time\b",
    r"\bwelcome back\b",
    r"\bthanks for watching\b",
]

META_SECTION_PATTERNS = [
    r"^\s*(introduction|overview)\s+to\s+the\s+lecture\s*$",
    r"^\s*lecture\s+(introduction|overview|agenda|outline)\s*$",
    r"^\s*course\s+housekeeping\s*$",
]

META_REWRITES = [
    (r"\b[Tt]he lecture defines\s+(.+?)\s+as\s+", r"\1 is "),
    (r"\b[Tt]he lecturer defines\s+(.+?)\s+as\s+", r"\1 is "),
    (r"\b[Tt]he instructor defines\s+(.+?)\s+as\s+", r"\1 is "),
    (r"\b[Tt]he lecture begins by\s+", ""),
    (r"\b[Tt]he lecture begins\s+", ""),
    (r"\b[Tt]he lecture concludes by\s+", ""),
    (r"\b[Tt]he lecture concludes\s+", ""),
    (r"\b[Tt]he lecture transitions to\s+", ""),
    (r"\b[Tt]he lecture shifts focus from\s+", ""),
    (r"\b[Tt]he lecture series splits\s+", ""),
    (r"\b[Tt]he lecture introduces\s+", ""),
    (r"\b[Tt]his lecture introduces\s+", ""),
    (r"\b[Tt]he lecture identifies\s+", ""),
    (r"\b[Tt]his lecture identifies\s+", ""),
    (r"\b[Tt]he lecture explains\s+", ""),
    (r"\b[Tt]his lecture explains\s+", ""),
    (r"\b[Tt]he lecture discusses\s+", ""),
    (r"\b[Tt]his lecture discusses\s+", ""),
    (r"\b[Tt]he lecture demonstrates\s+", ""),
    (r"\b[Tt]he lecture describes\s+", ""),
    (r"\b[Tt]he lecture notes that\s+", ""),
    (r"\b[Tt]he lecture notes\s+", ""),
    (r"\b[Tt]he lecture clarifies\s+", ""),
    (r"\b[Tt]he lecture emphasizes\s+", ""),
    (r"\b[Tt]he lecture outlines\s+", ""),
    (r"\b[Tt]he lecture uses\s+", ""),
    (r"\b[Tt]he lecture maps\s+", ""),
    (r"\b[Tt]he lecture distinguishes\s+", ""),
    (r"\b[Tt]he lecture recommends\s+", ""),
    (r"\b[Tt]he lecture references\s+", ""),
    (r"\b[Tt]he lecture illustrates\s+", ""),
    (r"\b[Tt]he lecture states that\s+", ""),
    (r"\b[Tt]he lecture states\s+", ""),
    (r"\b[Tt]he lecture implies\s+", ""),
    (r"\b[Tt]he lecture provides\s+", ""),
    (r"\b[Tt]he lecture contrasts\s+", ""),
    (r"\b[Tt]he lecture covers\s+", ""),
    (r"\b[Tt]he lecture details\s+", ""),
    (r"\b[Tt]he lecture highlights\s+", ""),
    (r"\b[Tt]he lecture briefly introduces\s+", ""),
    (r"\b[Tt]he lecturer demonstrates\s+", ""),
    (r"\b[Tt]he lecturer explains how to\s+", "To "),
    (r"\b[Tt]he lecturer explains\s+", ""),
    (r"\b[Tt]he lecturer notes that\s+", ""),
    (r"\b[Tt]he lecturer notes\s+", ""),
    (r"\b[Tt]he lecturer clarifies\s+", ""),
    (r"\b[Tt]he lecturer distinguishes\s+", ""),
    (r"\b[Tt]he lecturer suggests\s+", ""),
    (r"\b[Tt]he lecturer advises\s+", ""),
    (r"\b[Tt]he lecturer details\s+", ""),
    (r"\b[Tt]he lecturer confirms\s+", ""),
    (r"\b[Tt]he lecturer emphasizes\s+", ""),
    (r"\b[Tt]he lecturer provides\s+", ""),
    (r"\b[Tt]he lecturer shows how to\s+", "To "),
    (r"\b[Tt]he lecturer shows\s+", ""),
    (r"\b[Tt]he lecturer introduces\s+", ""),
    (r"\b[Tt]he lecturer claims\s+", ""),
    (r"\b[Tt]he lecturer refutes\s+", ""),
    (r"\b[Tt]he lecturer questions\s+", ""),
    (r"\b[Tt]he lecturer argues that\s+", ""),
    (r"\b[Tt]he lecturer argues\s+", ""),
    (r"\b[Tt]he lecturer states that\s+", ""),
    (r"\b[Tt]he lecturer states\s+", ""),
    (r"\b[Tt]he instructor introduces\s+", ""),
    (r"\b[Tt]he instructor identifies\s+", ""),
    (r"\b[Tt]he instructor explains\s+", ""),
    (r"\b[Tt]he instructor discusses\s+", ""),
    (r"\b[Tt]he instructor outlines a scenario where\s+", ""),
    (r"\b[Tt]he instructor outlines how\s+", ""),
    (r"\b[Tt]he instructor outlines\s+", ""),
    (r"\b[Tt]he speaker introduces\s+", ""),
    (r"\b[Tt]he speaker explains\s+", ""),
    (r"\b[Tt]his section covers\s+", ""),
    (r"\b[Tt]his part covers\s+", ""),
    (r"\b[Tt]his segment covers\s+", ""),
    (r"\b[Ii]n this lecture,?\s+", ""),
    (r"\b[Ii]n this section,?\s+", ""),
    (r"\b[Ii]n this segment,?\s+", ""),
]


def refine_notes_for_teaching(notes: dict[str, Any]) -> dict[str, Any]:
    refined = deepcopy(notes)
    sections = []
    for section in refined.get("sections", []):
        cleaned = refine_section(section)
        if cleaned:
            sections.append(cleaned)

    refined["sections"] = sections
    refined["refinement"] = {
        "removed_meta_language": True,
        "removed_off_topic_content": True,
        "style": "direct teaching notes",
    }
    return refined


def refine_section(section: dict[str, Any]) -> dict[str, Any] | None:
    title = clean_teaching_text(str(section.get("title", "")))
    summary = clean_teaching_text(str(section.get("summary", "")))
    body_text = " ".join(
        [
            title,
            summary,
            " ".join(str(item) for item in section.get("bullets", [])),
            " ".join(str(item) for item in section.get("examples", [])),
        ]
    )

    if is_off_topic(body_text) or is_meta_section_title(title):
        return None

    cleaned = dict(section)
    cleaned["title"] = title or "Concept"
    cleaned["summary"] = summary
    cleaned["bullets"] = clean_string_list(section.get("bullets", []))
    cleaned["examples"] = clean_string_list(section.get("examples", []))
    cleaned["claims"] = clean_string_list(section.get("claims", []))
    cleaned["key_terms"] = clean_key_terms(section.get("key_terms", []))
    cleaned["aids"] = clean_aids(section.get("aids", {}))
    return cleaned


def clean_string_list(items: list[Any]) -> list[str]:
    cleaned = []
    for item in items:
        text = clean_teaching_text(str(item))
        if text and not is_off_topic(text):
            cleaned.append(text)
    return cleaned


def clean_key_terms(items: list[Any]) -> list[dict[str, str]]:
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        term = clean_teaching_text(str(item.get("term", "")))
        definition = clean_teaching_text(str(item.get("definition", "")))
        if term and definition and not is_off_topic(f"{term} {definition}"):
            cleaned.append({"term": term, "definition": definition})
    return cleaned


def clean_aids(raw: Any) -> dict[str, list[dict[str, Any]]]:
    aids = {
        "architecture_diagrams": [],
        "flow_diagrams": [],
        "code_snippets": [],
        "syntax": [],
        "tables": [],
    }
    if not isinstance(raw, dict):
        return aids

    for key in ["architecture_diagrams", "flow_diagrams"]:
        for item in raw.get(key, []):
            if not isinstance(item, dict):
                continue
            title = clean_teaching_text(str(item.get("title", "")))
            mermaid = sanitize_mermaid(str(item.get("mermaid", "")))
            if title and mermaid and not is_off_topic(title):
                aids[key].append({"title": title, "mermaid": mermaid})

    for key in ["code_snippets", "syntax"]:
        for item in raw.get(key, []):
            if not isinstance(item, dict):
                continue
            title = clean_teaching_text(str(item.get("title", "")))
            code = str(item.get("code", "")).strip()
            explanation = clean_teaching_text(str(item.get("explanation", "")))
            if title and code and not is_off_topic(f"{title} {explanation}"):
                aids[key].append(
                    {
                        "title": title,
                        "language": clean_teaching_text(str(item.get("language", "text"))).lower() or "text",
                        "code": code,
                        "explanation": explanation,
                    }
                )

    for item in raw.get("tables", []):
        if not isinstance(item, dict):
            continue
        title = clean_teaching_text(str(item.get("title", "")))
        headers = [clean_teaching_text(str(header)) for header in item.get("headers", [])]
        rows = [
            [clean_teaching_text(str(cell)) for cell in row]
            for row in item.get("rows", [])
            if isinstance(row, list)
        ]
        if title and headers and rows and not is_off_topic(title):
            aids["tables"].append({"title": title, "headers": headers, "rows": rows})

    return aids


def clean_teaching_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    for pattern, replacement in META_REWRITES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\b[Tt]he (?:lecture|lecturer|instructor|speaker)\b\s*", "", text)
    text = re.sub(r"\bthat\s+([A-Z][A-Za-z0-9_-]+\s+(?:reads|writes|processes|stores))", r"\1", text)
    text = text.strip(" -:")
    return capitalize_sentence(text)


def capitalize_sentence(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def is_off_topic(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in OFF_TOPIC_PATTERNS)


def is_meta_section_title(title: str) -> bool:
    lowered = title.lower().strip()
    return any(re.search(pattern, lowered) for pattern in META_SECTION_PATTERNS)
