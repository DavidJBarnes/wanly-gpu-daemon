from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class SegmentClaim(BaseModel):
    """Mirrors wanly-api SegmentClaimResponse."""

    id: UUID
    job_id: UUID
    index: int
    prompt: str
    duration_seconds: float
    start_image: Optional[str] = None
    loras: Optional[list[Any]] = None
    faceswap_enabled: bool
    faceswap_method: Optional[str] = None
    faceswap_source_type: Optional[str] = None
    faceswap_image: Optional[str] = None
    faceswap_faces_order: Optional[str] = None
    faceswap_faces_index: Optional[str] = None
    width: int
    height: int
    fps: int
    seed: int


class SegmentResult(BaseModel):
    """Payload for PATCH /segments/{id}."""

    status: str  # "completed" or "failed"
    output_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    error_message: Optional[str] = None
