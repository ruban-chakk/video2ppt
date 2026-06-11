from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io_utils import read_json, write_json
from .llm import OpenAIClient
from .render import render_html
from .validation import (
    batched,
    collect_claims,
    normalize_validation,
    summarize_validation,
    validate_claim_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2notes-validate",
        description="Validate claims in an existing notes JSON using OpenAI and update HTML.",
    )
    parser.add_argument("notes_json", type=Path, help="Existing .notes.json file.")
    parser.add_argument("--output-html", type=Path, help="Output HTML path. Defaults to .html beside notes JSON.")
    parser.add_argument("--validation-json", type=Path, help="Validation JSON path. Defaults to .validation.json beside notes JSON.")
    parser.add_argument("--openai-model", default="gpt-5.5", help="OpenAI model for validation.")
    parser.add_argument("--openai-timeout", type=int, default=1800, help="Seconds to wait for each OpenAI request.")
    parser.add_argument("--batch-size", type=int, default=8, help="Claims per OpenAI validation request.")
    parser.add_argument("--no-web-search", action="store_true", help="Disable OpenAI web search.")
    parser.add_argument("--force", action="store_true", help="Revalidate claims even if validation JSON exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    notes_path = args.notes_json.expanduser().resolve()
    if not notes_path.exists():
        print(f"Notes JSON does not exist: {notes_path}", file=sys.stderr)
        return 2

    validation_path = (
        args.validation_json.expanduser().resolve()
        if args.validation_json
        else notes_path.with_suffix(".validation.json")
    )
    output_html = (
        args.output_html.expanduser().resolve()
        if args.output_html
        else notes_path.with_suffix(".html")
    )

    notes = read_json(notes_path)
    claims = collect_claims(notes)
    if not claims:
        payload = {"results": [], "summary": "No explicit factual claims were extracted."}
        write_json(validation_path, payload)
        render_html(notes_path, validation_path, output_html)
        print("No claims found.")
        return 0

    existing = read_existing_validation(validation_path) if not args.force else []
    done_claims = {item.get("claim") for item in existing}
    pending = [claim for claim in claims if claim["claim"] not in done_claims]
    results = list(existing)

    if not pending:
        print("All claims already validated.")
        render_html(notes_path, validation_path, output_html)
        return 0

    client = OpenAIClient(
        model=args.openai_model,
        timeout=args.openai_timeout,
        use_web_search=not args.no_web_search,
    )
    total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    print(f"Validating {len(pending)} pending claim(s) with {args.openai_model}.")

    for batch_index, batch in enumerate(batched(pending, args.batch_size), start=1):
        print(f"  validation batch {batch_index}/{total_batches}")
        response = validate_claim_batch(
            client,
            batch,
            transcript_evidence="No transcript supplied for this combined notes validation.",
            external_truth=True,
        )
        for item in response.get("results", []):
            normalized = normalize_validation(item)
            if normalized:
                results.append(normalized)

        payload = {"results": results, "summary": summarize_validation(results)}
        write_json(validation_path, payload)
        render_html(notes_path, validation_path, output_html)
        print(f"  saved progress to {validation_path}")

    print(f"Wrote {validation_path}")
    print(f"Wrote {output_html}")
    return 0


def read_existing_validation(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    payload = read_json(path)
    return [
        item
        for item in payload.get("results", [])
        if isinstance(item, dict) and item.get("claim")
    ]


if __name__ == "__main__":
    raise SystemExit(main())
