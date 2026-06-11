from __future__ import annotations

from pathlib import Path

from .io_utils import newer_than, read_json, write_json
from .models import TranscriptSegment


def transcribe_audio(
    audio_path: Path,
    transcript_path: Path,
    whisper_model: str,
    force: bool = False,
) -> list[TranscriptSegment]:
    if not force and newer_than(transcript_path, audio_path):
        return _segments_from_json(read_json(transcript_path))

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run `pip install -e .` inside your environment."
        ) from exc

    model = WhisperModel(whisper_model, device="auto", compute_type="auto")
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        word_timestamps=False,
        beam_size=5,
    )

    payload = {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": [
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }
            for segment in segments
            if segment.text.strip()
        ],
    }
    write_json(transcript_path, payload)
    return _segments_from_json(payload)


def _segments_from_json(payload: dict) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item["text"]).strip(),
        )
        for item in payload.get("segments", [])
        if str(item.get("text", "")).strip()
    ]
