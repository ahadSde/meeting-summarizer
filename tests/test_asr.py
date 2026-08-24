from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.asr import _format_time, get_whisper_model, transcribe_audio


def test_format_time() -> None:
    assert _format_time(0.0) == "00:00:00"
    assert _format_time(59.9) == "00:00:59"
    assert _format_time(65.0) == "00:01:05"
    assert _format_time(3661.0) == "01:01:01"
    assert _format_time(7325.0) == "02:02:05"


def test_get_whisper_model_caching() -> None:
    with patch("faster_whisper.WhisperModel") as mock_cls:
        instance_a = get_whisper_model("test-model", "cpu", "int8")
        instance_b = get_whisper_model("test-model", "cpu", "int8")
        assert instance_a is instance_b
        # Only called once due to lru_cache
        assert mock_cls.call_count == 1


def test_transcribe_audio_formats_transcript_and_segments(tmp_path: Path) -> None:
    audio_file = tmp_path / "meeting.wav"
    audio_file.write_bytes(b"data")

    mock_model = MagicMock()
    seg1 = MagicMock(start=0.5, end=3.2, text=" Welcome everyone. ")
    seg2 = MagicMock(start=4.0, end=7.5, text=" Let's begin the review. ")
    seg3 = MagicMock(start=601.0, end=604.0, text=" We will ship on Friday. ")

    mock_model.transcribe.return_value = ([seg1, seg2, seg3], None)

    transcript, segments = transcribe_audio(audio_file, model_name="base", model=mock_model)

    assert len(segments) == 3
    assert segments[0] == {"start": 0.5, "end": 3.2, "text": "Welcome everyone."}
    assert segments[1] == {"start": 4.0, "end": 7.5, "text": "Let's begin the review."}
    assert segments[2] == {"start": 601.0, "end": 604.0, "text": "We will ship on Friday."}

    expected_transcript = (
        "[00:00:00] Welcome everyone.\n"
        "[00:00:04] Let's begin the review.\n"
        "[00:10:01] We will ship on Friday."
    )
    assert transcript == expected_transcript
    mock_model.transcribe.assert_called_once_with(str(audio_file), vad_filter=True, beam_size=5)


def test_transcribe_audio_skips_empty_segments(tmp_path: Path) -> None:
    audio_file = tmp_path / "meeting.wav"
    audio_file.write_bytes(b"data")

    mock_model = MagicMock()
    seg_empty = MagicMock(start=0.0, end=1.0, text="   ")
    seg_valid = MagicMock(start=1.0, end=2.0, text="Hello world")

    mock_model.transcribe.return_value = ([seg_empty, seg_valid], None)

    transcript, segments = transcribe_audio(audio_file, model_name="base", model=mock_model)

    assert len(segments) == 1
    assert segments[0]["text"] == "Hello world"
    assert transcript == "[00:00:01] Hello world"
