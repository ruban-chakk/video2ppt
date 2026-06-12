from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .progress import clear_progress_line, finish_progress, print_progress


class MissingToolError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise MissingToolError("ffmpeg was not found on PATH. Install ffmpeg and try again.")


def extract_audio(video_path: Path, audio_path: Path, force: bool = False) -> None:
    require_ffmpeg()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path.exists() and not force and audio_path.stat().st_mtime >= video_path.stat().st_mtime:
        print("    audio extraction: cached")
        return

    duration = get_media_duration(video_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        str(audio_path),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last_percent = -1.0
    assert process.stdout is not None
    for line in process.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_ms" and duration:
            try:
                percent = (int(value) / 1_000_000) / duration * 100
            except ValueError:
                continue
            if percent - last_percent >= 1 or percent >= 100:
                print_progress("audio extraction", percent)
                last_percent = percent

    stderr = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        clear_progress_line()
        detail = stderr.strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {detail}")
    finish_progress("audio extraction")


def get_media_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None
