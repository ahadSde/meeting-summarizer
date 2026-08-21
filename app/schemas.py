from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class MeetingResponse(BaseModel):
    id: str
    filename: str
    status: str
    stage: str
    duration_seconds: Optional[float]
    transcript: Optional[str]
    summary: Optional[dict[str, Any]]
    error_message: Optional[str]
    processing_seconds: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True
