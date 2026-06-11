from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import newer_than, read_json, write_json
from .llm import OllamaClient
from .notes import compact_text
from .models import TranscriptSegment, ValidationResult


def validate_notes(
    notes: dict[str, Any],
    transcript_segments: list[TranscriptSegment],
    validation_path: Path,
    notes_path: Path,
    client: Any,
    force: bool = False,
    external_truth: bool = False,
) -> dict[str, Any]:
    if not force and newer_than(validation_path, notes_path):
        return read_json(validation_path)

    claims = collect_claims(notes)
    if not claims:
        payload = {"results": [], "summary": "No explicit factual claims were extracted."}
        write_json(validation_path, payload)
        return payload

    transcript_evidence = compact_transcript_for_validation(transcript_segments)
    results: list[dict[str, str]] = []
    for batch in batched(claims, 12):
        response = validate_claim_batch(client, batch, transcript_evidence, external_truth=external_truth)
        for item in response.get("results", []):
            normalized = normalize_validation(item)
            if normalized:
                results.append(normalized)

    payload = {
        "results": results,
        "summary": summarize_validation(results),
    }
    write_json(validation_path, payload)
    return payload


def collect_claims(notes: dict[str, Any]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for section in notes.get("sections", []):
        section_title = str(section.get("title", "Untitled")).strip()
        for claim in section.get("claims", []):
            text = compact_text(str(claim))
            if text:
                claims.append({"section_title": section_title, "claim": text})
    return claims


def compact_transcript_for_validation(segments: list[TranscriptSegment], max_chars: int = 14000) -> str:
    lines = [compact_text(segment.text) for segment in segments if compact_text(segment.text)]
    text = " ".join(lines)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...\n{tail}"


def validate_claim_batch(
    client: Any,
    claims: list[dict[str, str]],
    transcript_evidence: str,
    external_truth: bool = False,
) -> dict[str, Any]:
    if external_truth:
        system = """
You are a conservative factual verifier for technical lecture notes.
Check whether each claim is true using reliable sources and your available tools.
Prefer official documentation, project documentation, standards, textbooks, and reputable technical sources.
Use "verified" only when reliable sources support the claim.
Use "likely" when the claim is broadly correct but imprecise or source support is indirect.
Use "needs_review" when the claim depends on course-specific context, local environment, private demo data, or lacks enough evidence.
Use "unsupported" when reliable sources contradict the claim or the claim appears false.
Include source URLs when available.
"""
        evidence = "Use external verification. The transcript evidence below is optional context, not proof."
    else:
        system = """
You are a conservative lecture-notes credibility checker.
Judge only whether each claim is supported by the provided transcript evidence.
Use "verified" only when the transcript directly supports it.
Use "likely" when it is strongly implied but not directly stated.
Use "needs_review" when external sources or instructor material are required.
Use "unsupported" when the claim appears absent or contradicts the transcript.
"""
        evidence = "Transcript evidence"

    user = f"""
{evidence}:
{transcript_evidence}

Claims:
{claims}

Return JSON:
{{
  "results": [
    {{
      "section_title": "section",
      "claim": "claim text",
      "status": "verified|likely|needs_review|unsupported",
      "rationale": "short reason",
      "source_urls": ["https://source.example/path"]
    }}
  ]
}}
"""
    return client.generate_json(system=system, user=user)


def normalize_validation(item: dict[str, Any]) -> dict[str, str] | None:
    claim = compact_text(str(item.get("claim", "")))
    if not claim:
        return None
    status = compact_text(str(item.get("status", "needs_review"))).lower()
    if status not in {"verified", "likely", "needs_review", "unsupported"}:
        status = "needs_review"
    result = ValidationResult(
        claim=claim,
        status=status,
        rationale=compact_text(str(item.get("rationale", ""))) or "No rationale supplied.",
        section_title=compact_text(str(item.get("section_title", "Untitled"))) or "Untitled",
    )
    return {
        "section_title": result.section_title,
        "claim": result.claim,
        "status": result.status,
        "rationale": result.rationale,
        "source_urls": [
            compact_text(str(url))
            for url in item.get("source_urls", [])
            if compact_text(str(url))
        ][:5],
    }


def summarize_validation(results: list[dict[str, str]]) -> str:
    counts = {status: 0 for status in ["verified", "likely", "needs_review", "unsupported"]}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return (
        f"{counts['verified']} verified, {counts['likely']} likely, "
        f"{counts['needs_review']} need review, {counts['unsupported']} unsupported."
    )


def batched(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
