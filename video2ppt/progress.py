from __future__ import annotations

import sys


def print_progress(label: str, percent: float | None = None, detail: str = "") -> None:
    if percent is None:
        message = f"    {label}"
    else:
        bounded = max(0.0, min(100.0, percent))
        message = f"    {label}: {bounded:5.1f}%"
    if detail:
        message = f"{message} {detail}"
    print(f"\r{message}", end="", flush=True)


def finish_progress(label: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"\r    {label}: 100.0%{suffix}")


def clear_progress_line() -> None:
    print("\r" + " " * 80 + "\r", end="", file=sys.stdout, flush=True)
