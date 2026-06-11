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
        prog="video2notes-render",
        description="Refine, enrich, and render an existing notes JSON file.",
    )
    parser.add_argument("notes_json", type=Path, help="Existing .notes.json file to render.")
    parser.add_argument("--output-html", type=Path, help="Output HTML path. Defaults to .html beside notes JSON.")
    parser.add_argument("--validation-json", type=Path, help="Optional validation JSON file.")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="Local Ollama model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--llm-timeout", type=int, default=1800, help="Seconds to wait for each local LLM request.")
    parser.add_argument("--aids-batch-size", type=int, default=4, help="Sections per learning-aids request.")
    parser.add_argument("--skip-aids", action="store_true", help="Skip learning-aids enrichment.")
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

    if not args.skip_aids:
        print(f"enriching learning aids with {args.llm_model}")
        client = OllamaClient(model=args.llm_model, base_url=args.ollama_url, timeout=args.llm_timeout)
        notes = enrich_learning_aids(notes, client=client, batch_size=args.aids_batch_size)

    print("writing refined notes json")
    write_json(notes_path, notes)

    print(f"rendering {output_html}")
    render_html(notes_path, validation_path, output_html)
    print(f"wrote {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
