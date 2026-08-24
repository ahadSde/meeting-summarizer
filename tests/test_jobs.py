from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.database import SessionLocal
from app.jobs import process_meeting
from app.models import Meeting


def test_process_meeting_success(tmp_path: Path) -> None:
    db = SessionLocal()
    meeting_id = str(uuid4())
    meeting = Meeting(id=meeting_id, filename="test.wav", duration_seconds=60.0)
    db.add(meeting)
    db.commit()
    db.close()

    source_path = tmp_path / "source.wav"
    source_path.write_bytes(b"dummy audio")

    fake_transcript = "[00:00:01] Hello world"
    fake_segments = [{"start": 1.0, "end": 2.0, "text": "Hello world"}]
    fake_summary = {
        "overview": "Overview text",
        "key_points": ["Point 1"],
        "decisions": [],
        "action_items": [],
        "open_questions": [],
    }

    with patch("app.jobs.normalize_audio") as mock_norm, \
         patch("app.jobs.transcribe_audio", return_value=(fake_transcript, fake_segments)) as mock_trans, \
         patch("app.jobs.summarize_transcript", return_value=fake_summary) as mock_summ:

        process_meeting(meeting_id, source_path)

        mock_norm.assert_called_once()
        mock_trans.assert_called_once()
        mock_summ.assert_called_once()

    db = SessionLocal()
    updated = db.get(Meeting, meeting_id)
    assert updated is not None
    assert updated.status == "completed"
    assert updated.stage == "Completed"
    assert updated.transcript == fake_transcript
    assert updated.summary == fake_summary
    assert updated.completed_at is not None
    assert updated.processing_seconds is not None
    db.close()

    # Verify source file was cleaned up
    assert not source_path.exists()


def test_process_meeting_nonexistent_meeting(tmp_path: Path) -> None:
    source_path = tmp_path / "nonexistent.wav"
    source_path.write_bytes(b"data")
    # Should exit gracefully without raising
    process_meeting("does-not-exist", source_path)
    assert not source_path.exists()


def test_process_meeting_failure_updates_status(tmp_path: Path) -> None:
    db = SessionLocal()
    meeting_id = str(uuid4())
    meeting = Meeting(id=meeting_id, filename="fail.wav", duration_seconds=60.0)
    db.add(meeting)
    db.commit()
    db.close()

    source_path = tmp_path / "fail.wav"
    source_path.write_bytes(b"dummy")

    with patch("app.jobs.normalize_audio", side_effect=RuntimeError("FFmpeg crashed")):
        process_meeting(meeting_id, source_path)

    db = SessionLocal()
    updated = db.get(Meeting, meeting_id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.stage == "Failed"
    assert updated.error_message is not None
    assert "Processing failed" in updated.error_message
    db.close()

    # Source file cleaned up in finally
    assert not source_path.exists()
