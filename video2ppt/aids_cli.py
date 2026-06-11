from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aids import enrich_learning_aids
from .io_utils import read_json, write_json
from .llm import OllamaClient
from .refine import refine_notes_for_teaching
from .render import render_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2notes-aids",
        description="Add learning aids to an existing combined notes JSON and regenerate HTML.",
    )
    parser.add_argument("notes_json", type=Path, help="Combined or generated .notes.json file.")
    parser.add_argument("--output-html", type=Path, help="Output HTML path. Defaults to .html beside notes JSON.")
    parser.add_argument("--validation-json", type=Path, help="Optional validation JSON file.")
    parser.add_argument("--llm-model", default="qwen3.6:latest", help="Local Ollama model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--llm-timeout", type=int, default=3600, help="Seconds to wait for each local LLM request.")
    parser.add_argument("--batch-size", type=int, default=4, help="Sections per Ollama request.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    notes_path = args.notes_json.expanduser().resolve()
    if not notes_path.exists():
        print(f"Notes JSON does not exist: {notes_path}", file=sys.stderr)
        return 2

    output_html = (
        args.output_html.expanduser().resolve()
        if args.output_html
        else notes_path.with_suffix(".html")
    )
    validation_path = args.validation_json.expanduser().resolve() if args.validation_json else None

    print(f"reading {notes_path}")
    notes = refine_notes_for_teaching(read_json(notes_path))
    client = OllamaClient(model=args.llm_model, base_url=args.ollama_url, timeout=args.llm_timeout)

    def save_progress(updated_notes: dict) -> None:
        write_json(notes_path, updated_notes)
        render_html(notes_path, validation_path, output_html)
        print(f"    saved progress to {output_html}")

    print(f"adding learning aids with {args.llm_model}")
    notes = enrich_learning_aids(
        notes,
        client=client,
        batch_size=args.batch_size,
        after_batch=save_progress,
    )
    save_progress(notes)
    print(f"wrote {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
