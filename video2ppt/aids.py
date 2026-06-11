from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .llm import OllamaClient
from .notes import compact_text, normalize_aids


def enrich_learning_aids(
    notes: dict[str, Any],
    client: OllamaClient,
    batch_size: int = 4,
    after_batch: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    sections = notes.get("sections", [])
    missing = [
        (index, section)
        for index, section in enumerate(sections)
        if not has_learning_aids(section.get("aids", {}))
    ]
    if not missing:
        return notes

    total_batches = (len(missing) + batch_size - 1) // batch_size
    for batch_index, offset in enumerate(range(0, len(missing), batch_size), start=1):
        batch = missing[offset : offset + batch_size]
        print(f"    learning aids batch {batch_index}/{total_batches}")
        response = generate_aids_batch(client, batch)
        aids_by_index = {
            int(item["section_index"]): normalize_aids(item.get("aids", {}))
            for item in response.get("sections", [])
            if isinstance(item, dict) and is_int(item.get("section_index"))
        }
        for section_index, section in batch:
            section["aids"] = aids_by_index.get(section_index, normalize_aids({}))

        notes["learning_aids"] = {
            "generated": True,
            "description": "Architecture diagrams, flow diagrams, code snippets, syntax, and tables are included where useful.",
        }
        if after_batch:
            after_batch(notes)

    notes["learning_aids"] = {
        "generated": True,
        "description": "Architecture diagrams, flow diagrams, code snippets, syntax, and tables are included where useful.",
    }
    return notes


def generate_aids_batch(client: OllamaClient, batch: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    system = """
You add useful learning aids to direct teaching notes.
Only generate aids when they improve understanding. Prefer no aid over a weak or invented aid.
Use Mermaid for architecture and flow diagrams. Keep diagrams compact and valid.
Use code snippets only for concrete runnable or near-runnable examples from the section.
Use syntax snippets for commands, APIs, SQL, config, or function signatures.
Use tables for comparisons, formulas, options, or tradeoffs.
Do not invent technologies, commands, paths, credentials, or APIs that are absent from the note.
Return strict JSON only.
"""
    user = f"""
Sections:
{build_batch_payload(batch)}

Return JSON:
{{
  "sections": [
    {{
      "section_index": 0,
      "aids": {{
        "architecture_diagrams": [{{"title": "diagram title", "mermaid": "flowchart TD\\n  A[Component] --> B[Component]"}}],
        "flow_diagrams": [{{"title": "diagram title", "mermaid": "flowchart LR\\n  A[Step] --> B[Step]"}}],
        "code_snippets": [{{"title": "snippet title", "language": "python", "code": "print('example')", "explanation": "why this matters"}}],
        "syntax": [{{"title": "syntax title", "language": "sql", "code": "SELECT * FROM table;", "explanation": "when to use it"}}],
        "tables": [{{"title": "table title", "headers": ["Column A", "Column B"], "rows": [["value", "value"]]}}]
      }}
    }}
  ]
}}
"""
    return client.generate_json(system=system, user=user)


def build_batch_payload(batch: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    payload = []
    for section_index, section in batch:
        payload.append(
            {
                "section_index": section_index,
                "title": compact_text(str(section.get("title", ""))),
                "summary": compact_text(str(section.get("summary", ""))),
                "bullets": [compact_text(str(item)) for item in section.get("bullets", [])[:5]],
                "key_terms": section.get("key_terms", [])[:4],
                "examples": [compact_text(str(item)) for item in section.get("examples", [])[:3]],
            }
        )
    return payload


def has_learning_aids(raw: Any) -> bool:
    aids = normalize_aids(raw)
    return any(aids[key] for key in aids)


def is_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
