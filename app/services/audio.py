import json
import subprocess
from pathlib import Path


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}


class AudioError(ValueError):
    pass


def duration_seconds(path: Path) -> float:
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioError("The uploaded file could not be read as audio.")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError("The uploaded file has no readable duration.") from exc


def validate_audio(path: Path, size_bytes: int, max_bytes: int, max_duration_seconds: int) -> float:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError("Unsupported audio format.")
    if size_bytes > max_bytes:
        raise AudioError("This file is larger than the configured upload limit.")
    duration = duration_seconds(path)
    if duration <= 0:
        raise AudioError("The audio duration must be greater than zero.")
    if duration > max_duration_seconds:
        raise AudioError("This meeting exceeds the configured maximum duration.")
    return duration


def normalize_audio(source: Path, output_path: Path) -> Path:
    """Normalize uploaded audio to mono 16 kHz 16-bit PCM WAV for native Whisper transcription."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioError("Audio preparation failed. Please try another file.")
    return output_path

