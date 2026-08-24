from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class MeetingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
