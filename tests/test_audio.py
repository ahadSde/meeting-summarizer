from pathlib import Path

import pytest

from app.services.audio import AudioError, validate_audio


def test_invalid_extension_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "not-audio.txt"
    file.write_text("not audio")
    with pytest.raises(AudioError, match="Unsupported"):
        validate_audio(file, 11, 1000, 60)

