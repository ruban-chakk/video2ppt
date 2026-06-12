from __future__ import annotations

import wave
from pathlib import Path

from .io_utils import newer_than, read_json, write_json
from .models import TranscriptSegment
from .progress import finish_progress, print_progress


def transcribe_audio(
    audio_path: Path,
    transcript_path: Path,
    whisper_model: str,
    force: bool = False,
) -> list[TranscriptSegment]:
    if not force and newer_than(transcript_path, audio_path):
        print("    transcription: cached")
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

    duration = get_wav_duration(audio_path)
    segment_payload = []
    last_percent = -1.0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if duration:
            percent = segment.end / duration * 100
            if percent - last_percent >= 1 or percent >= 100:
                print_progress("transcription", percent)
                last_percent = percent
        segment_payload.append(
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text,
            }
        )
    finish_progress("transcription")

    payload = {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segment_payload,
    }
    write_json(transcript_path, payload)
    return _segments_from_json(payload)


def get_wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate:
                return frames / float(rate)
    except (wave.Error, OSError):
        return None
    return None


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
