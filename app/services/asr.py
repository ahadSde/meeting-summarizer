from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=4)
def get_whisper_model(model_name: str, device: str = "auto", compute_type: str = "auto") -> Any:
    """Load and cache the Whisper model instance to avoid reload overhead on every job."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    model: Any = None,
) -> tuple[str, list[dict]]:
    """Transcribe audio natively using faster-whisper with Silero VAD filtering."""
    if model is None:
        model = get_whisper_model(model_name)
    segments, _info = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
    transcript_segments = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        transcript_segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text,
        })
    transcript = "\n".join(f"[{_format_time(item['start'])}] {item['text']}" for item in transcript_segments)
    return transcript, transcript_segments


def _format_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

