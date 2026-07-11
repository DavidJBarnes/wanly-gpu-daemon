from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class LoraItem(BaseModel):
    """A resolved LoRA with S3 URIs for file sync."""

    lora_id: Optional[str] = None
    high_file: Optional[str] = None
    high_s3_uri: Optional[str] = None
    high_weight: float = 1.0
    low_file: Optional[str] = None
    low_s3_uri: Optional[str] = None
    low_weight: float = 1.0


class SegmentClaim(BaseModel):
    """Mirrors wanly-api SegmentClaimResponse."""

    id: UUID
    job_id: UUID
    index: int
    prompt: str
    duration_seconds: float
    speed: float = 1.0
    start_image: Optional[str] = None
    loras: Optional[list[LoraItem]] = None
    faceswap_enabled: bool
    faceswap_method: Optional[str] = None
    faceswap_source_type: Optional[str] = None
    faceswap_image: Optional[str] = None
    faceswap_faces_order: Optional[str] = None
    faceswap_faces_index: Optional[str] = None
    initial_reference_image: Optional[str] = None
    motion_keywords: Optional[list[str]] = None
    previous_motion_keywords: Optional[list[str]] = None
    previous_motion_magnitude: Optional[float] = None
    reference_frames: Optional[list[str]] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    steps_total: Optional[int] = None       # per-job sampler schedule length (None -> daemon default)
    high_noise_steps: Optional[int] = None   # high/low split boundary (None -> daemon default)
    flow_shift: Optional[float] = None       # ModelSamplingSD3 shift (None -> daemon default)
    negative_prompt: Optional[str] = None
    reprocess_type: Optional[str] = None
    output_path: Optional[str] = None
    # VACE continuation (resolved API-side). "vace" => generate this segment as a
    # video-conditioned continuation of previous_output_path's tail (vace_overlap_frames
    # kept frames). Daemon falls back to the traditional i2v path if not VACE-capable.
    continuation_mode: str = "traditional"
    previous_output_path: Optional[str] = None
    vace_overlap_frames: int = 12
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
    progress_log: Optional[str] = None
    motion_keywords: Optional[list[str]] = None
    motion_magnitude: Optional[float] = None
