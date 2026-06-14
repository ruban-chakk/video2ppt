from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .io_utils import newer_than, read_json, write_json
from .llm import OllamaClient
from .mermaid import sanitize_mermaid
from .models import NoteSection, TranscriptSegment


def generate_notes(
    transcript_segments: list[TranscriptSegment],
    notes_path: Path,
    video_path: Path,
    client: OllamaClient,
    max_chunk_chars: int,
    force: bool = False,
) -> dict[str, Any]:
    if not force and newer_than(notes_path, video_path):
        return read_json(notes_path)

    chunks = chunk_transcript(transcript_segments, max_chunk_chars=max_chunk_chars)
    chunk_cache_dir = notes_path.parent / f"{notes_path.stem}.chunks"
    chunk_cache_dir.mkdir(exist_ok=True)

    chunk_notes = []
    for index, chunk_text in enumerate(chunks, start=1):
        chunk_path = chunk_cache_dir / f"chunk-{index:04d}.json"
        if not force and chunk_path.exists():
            chunk_notes.append(read_json(chunk_path))
            continue

        percent = index / len(chunks) * 100
        print(f"    note chunk {index}/{len(chunks)} ({percent:5.1f}%)")
        chunk_note = generate_chunk_notes_with_fallback(
            client,
            chunk_text,
            index,
            len(chunks),
            min_chunk_chars=max(1800, max_chunk_chars // 2),
        )
        write_json(chunk_path, chunk_note)
        chunk_notes.append(chunk_note)

    merged = merge_notes(video_path, chunk_notes)
    write_json(notes_path, merged)
    return merged


def chunk_transcript(segments: list[TranscriptSegment], max_chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for segment in segments:
        line = f"[{format_timestamp(segment.start)}-{format_timestamp(segment.end)}] {compact_text(segment.text)}"
        if current and current_size + len(line) > max_chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line) + 1

    if current:
        chunks.append("\n".join(current))
    return chunks


def generate_chunk_notes_with_fallback(
    client: OllamaClient,
    chunk_text: str,
    chunk_index: int,
    chunk_count: int,
    min_chunk_chars: int,
) -> dict[str, Any]:
    try:
        return generate_chunk_notes(client, chunk_text, str(chunk_index), chunk_count)
    except Exception as exc:
        if len(chunk_text) <= min_chunk_chars:
            raise

        print(f"      chunk {chunk_index} failed JSON parse; retrying as smaller subchunks")
        subchunks = split_chunk_text(chunk_text)
        sub_notes = [
            generate_chunk_notes_with_fallback(
                client,
                subchunk,
                f"{chunk_index}.{sub_index}",
                chunk_count,
                min_chunk_chars=min_chunk_chars,
            )
            for sub_index, subchunk in enumerate(subchunks, start=1)
        ]
        merged = merge_chunk_notes(f"Chunk {chunk_index}", sub_notes)
        merged["fallback_reason"] = str(exc)
        return merged


def generate_chunk_notes(
    client: OllamaClient,
    chunk_text: str,
    chunk_index: str,
    chunk_count: int,
) -> dict[str, Any]:
    system = """
You convert lecture transcripts into concise, high-quality PPT-like study notes.
Preserve technical accuracy. Do not invent facts beyond the transcript.
Write for a learner reading the notes, not as a description of the lecture.
Teach the concepts directly. Avoid phrases like "the lecture introduces", "the lecture identifies",
"the instructor explains", "the instructor outlines", and "this section covers".
Remove personal chatter, sponsorships, greetings, jokes, housekeeping, and unrelated topics.
Prefer dense but readable bullets. Extract factual claims that should be validated.
Generate learning aids only when the transcript context makes them useful:
- architecture diagrams for systems, clusters, components, or infrastructure
- flow diagrams for pipelines, job execution, lifecycle, or data movement
- code snippets for runnable examples
- syntax snippets for commands, APIs, SQL, configuration, or function signatures
- tables for comparisons, formulas, configuration matrices, or tradeoffs
Return strict JSON. All strings must be quoted. Arrays must not have trailing commas.
Use at most 3 sections, 5 bullets per section, 5 claims per section, and 4 key terms per section.
Use Mermaid syntax for diagrams. Keep diagrams compact and valid. Do not use markdown fences.
"""
    user = f"""
Transcript chunk {chunk_index} of {chunk_count}:

{chunk_text}

Create JSON with this schema:
{{
  "title": "short lecture/chunk title",
  "sections": [
    {{
      "title": "slide-like section title",
      "summary": "2-3 sentence direct teaching explanation without lecture meta-language",
      "bullets": ["direct teaching bullet"],
      "key_terms": [{{"term": "term", "definition": "definition from transcript"}}],
      "examples": ["example or application"],
      "aids": {{
        "architecture_diagrams": [{{"title": "diagram title", "mermaid": "flowchart TD\\n  A[Component] --> B[Component]"}}],
        "flow_diagrams": [{{"title": "diagram title", "mermaid": "flowchart LR\\n  A[Step] --> B[Step]"}}],
        "code_snippets": [{{"title": "snippet title", "language": "python", "code": "print('example')", "explanation": "why this matters"}}],
        "syntax": [{{"title": "syntax title", "language": "sql", "code": "SELECT * FROM table;", "explanation": "when to use it"}}],
        "tables": [{{"title": "table title", "headers": ["Column A", "Column B"], "rows": [["value", "value"]]}}]
      }},
      "timestamps": [0.0],
      "claims": ["specific factual claim worth checking"]
    }}
  ]
}}
"""
    return client.generate_json(system=system, user=user)


def split_chunk_text(chunk_text: str) -> list[str]:
    lines = [line for line in chunk_text.splitlines() if line.strip()]
    if len(lines) < 2:
        midpoint = max(1, len(chunk_text) // 2)
        return [chunk_text[:midpoint], chunk_text[midpoint:]]

    midpoint = len(lines) // 2
    return ["\n".join(lines[:midpoint]), "\n".join(lines[midpoint:])]


def merge_chunk_notes(title: str, chunk_notes: list[dict[str, Any]]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    titles: list[str] = []
    for chunk in chunk_notes:
        if not isinstance(chunk, dict):
            continue
        chunk_title = str(chunk.get("title", "")).strip()
        if chunk_title:
            titles.append(chunk_title)
        raw_sections = chunk.get("sections", [])
        if isinstance(raw_sections, list):
            sections.extend(raw_sections)
    return {
        "title": titles[0] if titles else title,
        "sections": sections,
    }


def merge_notes(video_path: Path, chunk_notes: list[dict[str, Any]]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    titles: list[str] = []

    for chunk in chunk_notes:
        if not isinstance(chunk, dict):
            continue
        title = str(chunk.get("title", "")).strip()
        if title:
            titles.append(title)
        raw_sections = chunk.get("sections", [])
        if isinstance(raw_sections, dict):
            raw_sections = [raw_sections]
        if not isinstance(raw_sections, list):
            continue
        for raw_section in raw_sections:
            section = normalize_section(raw_section)
            if section:
                sections.append(section)

    title = titles[0] if titles else video_path.stem.replace("_", " ").replace("-", " ").title()
    return {
        "source_video": str(video_path),
        "title": title,
        "content_hash": hashlib.sha256(jsonish(chunk_notes).encode("utf-8")).hexdigest(),
        "sections": sections,
    }


def normalize_section(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        text = compact_text(raw)
        if not text:
            return None
        raw = {"title": "Concept", "summary": text}
    if not isinstance(raw, dict):
        return None

    title = compact_text(str(raw.get("title", "")).strip())
    summary = compact_text(str(raw.get("summary", "")).strip())
    if not title and not summary:
        return None

    note = NoteSection(
        title=title or "Lecture Notes",
        summary=summary,
        bullets=[compact_text(str(item)) for item in raw.get("bullets", []) if compact_text(str(item))],
        key_terms=[
            {
                "term": compact_text(str(item.get("term", ""))),
                "definition": compact_text(str(item.get("definition", ""))),
            }
            for item in raw.get("key_terms", [])
            if isinstance(item, dict) and compact_text(str(item.get("term", "")))
        ],
        examples=[compact_text(str(item)) for item in raw.get("examples", []) if compact_text(str(item))],
        aids=normalize_aids(raw.get("aids", {})),
        timestamps=[float(item) for item in raw.get("timestamps", []) if _is_number(item)],
        claims=[compact_text(str(item)) for item in raw.get("claims", []) if compact_text(str(item))],
    )
    return {
        "title": note.title,
        "summary": note.summary,
        "bullets": note.bullets,
        "key_terms": note.key_terms,
        "examples": note.examples,
        "aids": note.aids,
        "timestamps": note.timestamps,
        "claims": note.claims,
    }


def normalize_aids(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        return empty_aids()

    aids = empty_aids()
    for key in ["architecture_diagrams", "flow_diagrams"]:
        for item in raw.get(key, []):
            if not isinstance(item, dict):
                continue
            title = compact_text(str(item.get("title", "")))
            mermaid = sanitize_mermaid(str(item.get("mermaid", "")))
            if title and mermaid:
                aids[key].append({"title": title, "mermaid": mermaid})

    for key in ["code_snippets", "syntax"]:
        for item in raw.get(key, []):
            if not isinstance(item, dict):
                continue
            title = compact_text(str(item.get("title", "")))
            code = str(item.get("code", "")).strip()
            if title and code:
                aids[key].append(
                    {
                        "title": title,
                        "language": compact_text(str(item.get("language", "text"))) or "text",
                        "code": code,
                        "explanation": compact_text(str(item.get("explanation", ""))),
                    }
                )

    for item in raw.get("tables", []):
        if not isinstance(item, dict):
            continue
        title = compact_text(str(item.get("title", "")))
        headers = [compact_text(str(header)) for header in item.get("headers", [])]
        rows = [
            [compact_text(str(cell)) for cell in row]
            for row in item.get("rows", [])
            if isinstance(row, list)
        ]
        if title and headers and rows:
            aids["tables"].append({"title": title, "headers": headers, "rows": rows})

    return aids


def empty_aids() -> dict[str, list[dict[str, Any]]]:
    return {
        "architecture_diagrams": [],
        "flow_diagrams": [],
        "code_snippets": [],
        "syntax": [],
        "tables": [],
    }


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def jsonish(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False)
