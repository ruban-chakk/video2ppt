from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .aids import enrich_learning_aids
from .llm import OllamaClient, OpenAIClient
from .media import MissingToolError, extract_audio
from .models import TranscriptSegment, VideoJob
from .notes import generate_notes
from .io_utils import write_json
from .refine import refine_notes_for_teaching
from .render import render_html
from .scanner import discover_videos
from .transcriber import transcribe_audio
from .validation import validate_notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2notes",
        description="Convert lecture videos into validated PPT-like HTML notes.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Root folder(s) or video file(s).")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size/name.")
    parser.add_argument("--transcribe-workers", type=int, default=1, help="Audio/transcription workers. Keep notes generation sequential.")
    parser.add_argument("--audio-stall-timeout", type=int, default=600, help="Seconds of no FFmpeg progress before audio extraction fails.")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="Local Ollama model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--llm-timeout", type=int, default=1800, help="Seconds to wait for each local LLM request.")
    parser.add_argument("--validation-provider", choices=["ollama", "openai"], default="ollama", help="Provider used for claim validation.")
    parser.add_argument("--openai-model", default="gpt-5.5", help="OpenAI model used when --validation-provider openai.")
    parser.add_argument("--openai-timeout", type=int, default=1800, help="Seconds to wait for each OpenAI validation request.")
    parser.add_argument("--no-openai-web-search", action="store_true", help="Disable OpenAI web search during claim validation.")
    parser.add_argument("--max-chunk-chars", type=int, default=9000, help="Transcript chars per LLM note chunk.")
    parser.add_argument("--force-audio", action="store_true", help="Re-extract audio even when cached.")
    parser.add_argument("--force-transcribe", action="store_true", help="Re-transcribe even when cached.")
    parser.add_argument("--force-notes", action="store_true", help="Regenerate notes even when cached.")
    parser.add_argument("--force-validation", action="store_true", help="Revalidate claims even when cached.")
    parser.add_argument("--skip-aids", action="store_true", help="Skip architecture diagrams, flows, code, syntax, and tables.")
    parser.add_argument("--skip-validation", action="store_true", help="Generate notes without validation.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovered videos without processing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    overall_start = time.perf_counter()
    args = build_parser().parse_args(argv)

    roots = [path.expanduser().resolve() for path in args.paths]
    missing = [root for root in roots if not root.exists()]
    if missing:
        for root in missing:
            print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    jobs = discover_jobs(roots)
    if not jobs:
        print("No video files found under supplied path(s).")
        return 0

    print(f"Found {len(jobs)} video(s) across {len(roots)} path(s).")
    if args.dry_run:
        for job in jobs:
            print(job.video_path)
        return 0

    client = OllamaClient(model=args.llm_model, base_url=args.ollama_url, timeout=args.llm_timeout)
    transcripts: dict[Path, list[TranscriptSegment]] = {}
    prep_times: dict[Path, float] = {}
    metrics: list[dict[str, Any]] = []
    failures = 0

    if args.transcribe_workers > 1:
        prep_start = time.perf_counter()
        print(f"\nPreparing audio/transcripts with {args.transcribe_workers} worker(s).")
        transcripts, failures, prep_times = prepare_transcripts_parallel(args, jobs)
        print(f"Transcript preparation wall time: {format_duration(time.perf_counter() - prep_start)}")

    for index, job in enumerate(jobs, start=1):
        video_start = time.perf_counter()
        stage_times: dict[str, float] = {}
        print(f"\n[{index}/{len(jobs)}] {job.video_path}")
        try:
            transcript = transcripts.get(job.video_path)
            if transcript is None:
                stage_start = time.perf_counter()
                transcript = prepare_transcript(args, job)
                stage_times["audio_transcription"] = time.perf_counter() - stage_start
            else:
                stage_times["audio_transcription"] = prep_times.get(job.video_path, 0.0)

            print(f"  generating notes with {args.llm_model}")
            stage_start = time.perf_counter()
            notes = generate_notes(
                transcript,
                job.notes_path,
                job.video_path,
                client=client,
                max_chunk_chars=args.max_chunk_chars,
                force=args.force_notes,
            )
            stage_times["notes"] = time.perf_counter() - stage_start
            print("  refining notes for teaching style")
            stage_start = time.perf_counter()
            notes = refine_notes_for_teaching(notes)
            write_json(job.notes_path, notes)
            stage_times["refine"] = time.perf_counter() - stage_start

            if not args.skip_aids:
                print("  enriching learning aids")
                stage_start = time.perf_counter()
                notes = enrich_learning_aids(notes, client=client)
                write_json(job.notes_path, notes)
                stage_times["aids"] = time.perf_counter() - stage_start

            validation_path = None
            if not args.skip_validation:
                print("  validating claims")
                stage_start = time.perf_counter()
                validation_client = (
                    OpenAIClient(
                        model=args.openai_model,
                        timeout=args.openai_timeout,
                        use_web_search=not args.no_openai_web_search,
                    )
                    if args.validation_provider == "openai"
                    else client
                )
                validate_notes(
                    notes,
                    transcript,
                    job.validation_path,
                    job.notes_path,
                    client=validation_client,
                    force=args.force_validation,
                    external_truth=args.validation_provider == "openai",
                )
                validation_path = job.validation_path
                stage_times["validation"] = time.perf_counter() - stage_start

            print("  rendering html")
            stage_start = time.perf_counter()
            render_html(job.notes_path, validation_path, job.html_path)
            stage_times["render"] = time.perf_counter() - stage_start
            print(f"  wrote {job.html_path}")

            total = time.perf_counter() - video_start
            metrics.append(
                {
                    "video": job.video_path,
                    "status": "ok",
                    "total": total,
                    "stages": stage_times,
                }
            )
            print(f"  video time: {format_duration(total)}")
        except MissingToolError as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # Keep batch processing alive for other lectures.
            failures += 1
            total = time.perf_counter() - video_start
            metrics.append(
                {
                    "video": job.video_path,
                    "status": "failed",
                    "total": total,
                    "stages": stage_times,
                    "error": str(exc),
                }
            )
            print(f"  failed: {exc}", file=sys.stderr)
            print(f"  video time before failure: {format_duration(total)}", file=sys.stderr)

    print_metrics_summary(metrics, overall_elapsed=time.perf_counter() - overall_start)
    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1

    print("\nCompleted successfully.")
    return 0

def discover_jobs(roots: list[Path]) -> list[VideoJob]:
    seen: set[Path] = set()
    jobs: list[VideoJob] = []
    for root in roots:
        for job in discover_videos(root):
            resolved = job.video_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            jobs.append(VideoJob(resolved))
    return jobs


def prepare_transcript(args: Any, job: VideoJob) -> list[TranscriptSegment]:
    print("  extracting audio")
    extract_audio(
        job.video_path,
        job.audio_path,
        force=args.force_audio,
        stall_timeout=args.audio_stall_timeout,
    )

    print(f"  transcribing with faster-whisper model: {args.whisper_model}")
    return transcribe_audio(
        job.audio_path,
        job.transcript_path,
        whisper_model=args.whisper_model,
        force=args.force_transcribe,
    )


def prepare_transcripts_parallel(
    args: Any,
    jobs: list[VideoJob],
) -> tuple[dict[Path, list[TranscriptSegment]], int, dict[Path, float]]:
    transcripts: dict[Path, list[TranscriptSegment]] = {}
    prep_times: dict[Path, float] = {}
    failures = 0
    workers = max(1, args.transcribe_workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(prepare_transcript_worker, args, job): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                transcript, elapsed = future.result()
                transcripts[job.video_path] = transcript
                prep_times[job.video_path] = elapsed
                print(
                    f"  [{completed}/{len(jobs)}] transcript ready: {job.video_path.name} "
                    f"({format_duration(elapsed)})"
                )
            except MissingToolError:
                raise
            except Exception as exc:
                failures += 1
                print(f"  [{completed}/{len(jobs)}] transcript failed for {job.video_path}: {exc}", file=sys.stderr)
    return transcripts, failures, prep_times


def prepare_transcript_worker(args: Any, job: VideoJob) -> tuple[list[TranscriptSegment], float]:
    start = time.perf_counter()
    extract_audio(
        job.video_path,
        job.audio_path,
        force=args.force_audio,
        stall_timeout=args.audio_stall_timeout,
    )
    transcript = transcribe_audio(
        job.audio_path,
        job.transcript_path,
        whisper_model=args.whisper_model,
        force=args.force_transcribe,
    )
    return transcript, time.perf_counter() - start


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def print_metrics_summary(metrics: list[dict[str, Any]], overall_elapsed: float) -> None:
    if not metrics:
        print(f"\nOverall time: {format_duration(overall_elapsed)}")
        return

    print("\nRun Metrics")
    print(f"Overall time: {format_duration(overall_elapsed)}")
    rows = [
        [
            str(index),
            item["video"].name,
            item["status"],
            format_duration(item["total"]),
            format_stage_summary(item.get("stages", {})),
            str(item.get("error", "")),
        ]
        for index, item in enumerate(metrics, start=1)
    ]
    print_table(["#", "Video", "Status", "Total", "Stages", "Error"], rows)


def format_stage_summary(stages: dict[str, float]) -> str:
    if not stages:
        return ""
    return ", ".join(
        f"{name}={format_duration(elapsed)}"
        for name, elapsed in stages.items()
    )


def print_table(headers: list[str], rows: list[list[str]], max_width: int = 42) -> None:
    clipped_rows = [[clip_cell(cell, max_width) for cell in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in clipped_rows))
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)
    print(" | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print(separator)
    for row in clipped_rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def clip_cell(value: str, max_width: int) -> str:
    value = value.replace("\n", " ").strip()
    if len(value) <= max_width:
        return value
    return value[: max_width - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
