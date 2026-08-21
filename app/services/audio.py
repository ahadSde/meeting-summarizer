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


def prepare_chunks(source: Path, duration: float, output_dir: Path, threshold_seconds: int, chunk_seconds: int) -> list[tuple[Path, float]]:
    """Normalize audio and split long recordings with a three-second overlap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if duration <= threshold_seconds:
        chunks = [(output_dir / "chunk-000.wav", 0.0, duration)]
    else:
        overlap, start, index, chunks = 3, 0.0, 0, []
        while start < duration:
            chunks.append((output_dir / f"chunk-{index:03}.wav", start, min(chunk_seconds, duration - start)))
            start += chunk_seconds - overlap
            index += 1
    prepared = []
    for output, start, length in chunks:
        command = ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(source), "-t", str(length), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise AudioError("Audio preparation failed. Please try another file.")
        prepared.append((output, start))
    return prepared

