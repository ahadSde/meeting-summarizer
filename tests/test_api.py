import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import Meeting
from app.services.audio import AudioError

from uuid import uuid4

client = TestClient(app)


def test_index_page_returns_html() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Meeting Summarizer" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "groq_configured" in data


def test_get_meeting_not_found() -> None:
    response = client.get("/api/meetings/non-existent-uuid")
    assert response.status_code == 404
    assert response.json()["detail"] == "Meeting not found."


def test_get_meeting_found() -> None:
    db = SessionLocal()
    meeting_id = str(uuid4())
    meeting = Meeting(
        id=meeting_id,
        filename="sample.mp3",
        status="completed",
        stage="Completed",
        duration_seconds=120.0,
        transcript="[00:00:01] Hello",
        summary={"overview": "Test sync", "key_points": [], "decisions": [], "action_items": [], "open_questions": []},
        processing_seconds=4.2,
    )
    db.add(meeting)
    db.commit()
    db.close()

    response = client.get(f"/api/meetings/{meeting_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == meeting_id
    assert data["filename"] == "sample.mp3"
    assert data["status"] == "completed"
    assert data["duration_seconds"] == 120.0
    assert data["transcript"] == "[00:00:01] Hello"
    assert data["summary"]["overview"] == "Test sync"


def test_create_meeting_success() -> None:
    fake_audio_bytes = b"fake wav audio content"
    files = {"audio": ("sample.wav", io.BytesIO(fake_audio_bytes), "audio/wav")}

    with patch("app.main.validate_audio", return_value=95.0), \
         patch("app.main.process_meeting"):
        response = client.post("/api/meetings", files=files)

        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["filename"] == "sample.wav"
        assert data["status"] == "queued"
        assert data["duration_seconds"] == 95.0

        # Verify DB entry exists
        db = SessionLocal()
        db_meeting = db.get(Meeting, data["id"])
        assert db_meeting is not None
        assert db_meeting.filename == "sample.wav"
        db.close()


def test_create_meeting_unsupported_format_rejected() -> None:
    files = {"audio": ("notes.txt", io.BytesIO(b"just text"), "text/plain")}

    response = client.post("/api/meetings", files=files)
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_create_meeting_oversized_stream_rejected() -> None:
    huge_data = b"x" * 200
    files = {"audio": ("huge.wav", io.BytesIO(huge_data), "audio/wav")}

    # Configure max upload to only 100 bytes for this test
    with patch("app.config.Settings.max_upload_bytes", new_callable=lambda: 100):
        response = client.post("/api/meetings", files=files)
        assert response.status_code == 400
        assert "larger than the configured upload limit" in response.json()["detail"]


def test_create_meeting_audio_validation_error() -> None:
    files = {"audio": ("corrupt.mp3", io.BytesIO(b"garbage"), "audio/mp3")}

    with patch("app.main.validate_audio", side_effect=AudioError("The audio duration must be greater than zero.")):
        response = client.post("/api/meetings", files=files)
        assert response.status_code == 400
        assert "greater than zero" in response.json()["detail"]
