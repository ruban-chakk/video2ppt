from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}


@dataclass(frozen=True)
class VideoJob:
    video_path: Path

    @property
    def stem_path(self) -> Path:
        return self.video_path.with_suffix("")

    @property
    def work_dir(self) -> Path:
        return self.video_path.parent / self.video_path.stem

    @property
    def audio_path(self) -> Path:
        return self.work_dir / f"{self.video_path.stem}.audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.work_dir / f"{self.video_path.stem}.transcript.json"

    @property
    def notes_path(self) -> Path:
        return self.work_dir / f"{self.video_path.stem}.notes.json"

    @property
    def aids_path(self) -> Path:
        return self.work_dir / f"{self.video_path.stem}.aids.json"

    @property
    def validation_path(self) -> Path:
        return self.work_dir / f"{self.video_path.stem}.validation.json"

    @property
    def html_path(self) -> Path:
        return self.video_path.with_suffix(".notes.html")


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class NoteSection:
    title: str
    summary: str
    bullets: list[str] = field(default_factory=list)
    key_terms: list[dict[str, str]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    aids: dict = field(default_factory=dict)
    timestamps: list[float] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    claim: str
    status: str
    rationale: str
    section_title: str
