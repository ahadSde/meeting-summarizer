import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import BASE_DIR, settings
from app.database import SessionLocal
from app.models import Meeting
from app.services.asr import transcribe_chunks
from app.services.audio import prepare_chunks
from app.services.summarizer import summarize_transcript

logger = logging.getLogger(__name__)


def process_meeting(meeting_id: str, source_path: Path) -> None:
    started, db, chunks_dir = time.monotonic(), SessionLocal(), BASE_DIR / "tmp" / meeting_id
    try:
        meeting = db.get(Meeting, meeting_id)
        if not meeting:
            return
        meeting.status, meeting.stage = "processing", "Preparing audio"
        db.commit()
        logger.info("meeting_processing_started", extra={"meeting_id": meeting_id})
        chunks = prepare_chunks(source_path, meeting.duration_seconds or 0, chunks_dir, settings.chunk_threshold_minutes * 60, settings.chunk_duration_minutes * 60)
        meeting.stage = "Transcribing audio"
        db.commit()
        transcript, _segments = transcribe_chunks(chunks, settings.whisper_model)
        meeting.transcript, meeting.stage = transcript, "Generating summary"
        db.commit()
        meeting.summary = summarize_transcript(transcript, settings.groq_api_key, settings.groq_model)
        meeting.status, meeting.stage, meeting.completed_at = "completed", "Completed", datetime.now(timezone.utc)
        meeting.processing_seconds = round(time.monotonic() - started, 2)
        db.commit()
        logger.info("meeting_processing_completed", extra={"meeting_id": meeting_id, "seconds": meeting.processing_seconds})
    except Exception:
        logger.exception("meeting_processing_failed", extra={"meeting_id": meeting_id})
        meeting = db.get(Meeting, meeting_id)
        if meeting:
            meeting.status, meeting.stage = "failed", "Failed"
            meeting.error_message = "Processing failed. Check server logs and confirm the API configuration."
            meeting.processing_seconds = round(time.monotonic() - started, 2)
            db.commit()
    finally:
        db.close()
        shutil.rmtree(chunks_dir, ignore_errors=True)
        source_path.unlink(missing_ok=True)
