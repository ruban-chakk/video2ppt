from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class MissingToolError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise MissingToolError("ffmpeg was not found on PATH. Install ffmpeg and try again.")


def extract_audio(video_path: Path, audio_path: Path, force: bool = False) -> None:
    require_ffmpeg()
    if audio_path.exists() and not force and audio_path.stat().st_mtime >= video_path.stat().st_mtime:
        return

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
        str(audio_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {detail}")
