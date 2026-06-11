from __future__ import annotations

from pathlib import Path

from .models import VIDEO_EXTENSIONS, VideoJob


def discover_videos(root: Path) -> list[VideoJob]:
    if root.is_file():
        if root.suffix.lower() in VIDEO_EXTENSIONS:
            return [VideoJob(root)]
        return []

    jobs: list[VideoJob] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            jobs.append(VideoJob(path))
    return jobs
