from pathlib import Path


def transcribe_chunks(chunks: list[tuple[Path, float]], model_name: str) -> tuple[str, list[dict]]:
    """Transcribe locally and return readable text plus timestamped segments."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="auto")
    transcript_segments = []
    for chunk_path, offset in chunks:
        segments, _info = model.transcribe(str(chunk_path), vad_filter=True, beam_size=5)
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if transcript_segments and text.lower() == transcript_segments[-1]["text"].lower():
                continue
            transcript_segments.append({"start": round(offset + segment.start, 2), "end": round(offset + segment.end, 2), "text": text})
    transcript = "\n".join(f"[{_format_time(item['start'])}] {item['text']}" for item in transcript_segments)
    return transcript, transcript_segments


def _format_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

