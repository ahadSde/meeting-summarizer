from contextlib import asynccontextmanager
import logging
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import BASE_DIR, settings
from app.database import Base, engine, get_db
from app.jobs import process_meeting
from app.models import Meeting
from app.schemas import MeetingResponse
from app.services.audio import AudioError, validate_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

UPLOAD_DIR = BASE_DIR / "uploads"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    Base.metadata.create_all(bind=engine)
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    yield
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(title="Meeting Summarizer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "groq_configured": bool(settings.groq_api_key)}


@app.post("/api/meetings", response_model=MeetingResponse, status_code=202)
async def create_meeting(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Meeting:
    suffix = Path(audio.filename or "audio").suffix.lower()
    meeting_id = str(uuid4())
    source_path = UPLOAD_DIR / f"{meeting_id}{suffix}"
    size = 0
    try:
        with source_path.open("wb") as output:
            while content := await audio.read(1024 * 1024):
                size += len(content)
                if size > settings.max_upload_bytes:
                    raise AudioError("This file is larger than the configured upload limit.")
                output.write(content)
        duration = validate_audio(source_path, size, settings.max_upload_bytes, settings.max_audio_duration_minutes * 60)
    except AudioError as exc:
        source_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await audio.close()

    meeting = Meeting(id=meeting_id, filename=audio.filename or "audio", duration_seconds=duration)
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    background_tasks.add_task(process_meeting, meeting_id, source_path)
    return meeting


@app.get("/api/meetings/{meeting_id}", response_model=MeetingResponse)
def get_meeting(meeting_id: str, db: Session = Depends(get_db)) -> Meeting:
    meeting = db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    return meeting


