import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.audio import AudioError, duration_seconds, normalize_audio, validate_audio


def test_invalid_extension_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "not-audio.txt"
    file.write_text("not audio")
    with pytest.raises(AudioError, match="Unsupported"):
        validate_audio(file, 11, 1000, 60)


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "meeting.mp3"
    file.write_bytes(b"dummy audio content")
    with pytest.raises(AudioError, match="larger than the configured upload limit"):
        validate_audio(file, size_bytes=2000, max_bytes=1000, max_duration_seconds=3600)


def test_zero_or_negative_duration_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "empty.wav"
    file.write_bytes(b"dummy")
    with patch("app.services.audio.duration_seconds", return_value=0.0):
        with pytest.raises(AudioError, match="greater than zero"):
            validate_audio(file, size_bytes=100, max_bytes=1000, max_duration_seconds=3600)


def test_excessive_duration_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "long.wav"
    file.write_bytes(b"dummy")
    with patch("app.services.audio.duration_seconds", return_value=7300.0):
        with pytest.raises(AudioError, match="exceeds the configured maximum duration"):
            validate_audio(file, size_bytes=100, max_bytes=1000, max_duration_seconds=7200)


def test_valid_audio_validation_succeeds(tmp_path: Path) -> None:
    file = tmp_path / "valid.m4a"
    file.write_bytes(b"dummy")
    with patch("app.services.audio.duration_seconds", return_value=125.5):
        duration = validate_audio(file, size_bytes=500, max_bytes=1000, max_duration_seconds=3600)
        assert duration == 125.5


def test_duration_seconds_ffprobe_failure(tmp_path: Path) -> None:
    file = tmp_path / "corrupt.mp3"
    file.write_bytes(b"not valid audio data")
    mock_result = MagicMock(returncode=1, stdout="", stderr="Invalid data")
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(AudioError, match="could not be read as audio"):
            duration_seconds(file)


def test_duration_seconds_invalid_json(tmp_path: Path) -> None:
    file = tmp_path / "valid_name.mp3"
    file.write_bytes(b"data")
    mock_result = MagicMock(returncode=0, stdout="not json", stderr="")
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(AudioError, match="no readable duration"):
            duration_seconds(file)


def test_duration_seconds_valid(tmp_path: Path) -> None:
    file = tmp_path / "meeting.wav"
    file.write_bytes(b"data")
    mock_output = json.dumps({"format": {"duration": "42.50"}})
    mock_result = MagicMock(returncode=0, stdout=mock_output, stderr="")
    with patch("subprocess.run", return_value=mock_result):
        assert duration_seconds(file) == 42.50


def test_normalize_audio_success(tmp_path: Path) -> None:
    source = tmp_path / "incoming.mp3"
    source.write_bytes(b"dummy mp3 data")
    output = tmp_path / "normalized.wav"

    mock_result = MagicMock(returncode=0)
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result_path = normalize_audio(source, output)
        assert result_path == output
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-ar" in cmd and "16000" in cmd
        assert "-ac" in cmd and "1" in cmd


def test_normalize_audio_failure(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"corrupt")
    output = tmp_path / "output.wav"

    mock_result = MagicMock(returncode=1, stderr="Transcoding failed")
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(AudioError, match="Audio preparation failed"):
            normalize_audio(source, output)


