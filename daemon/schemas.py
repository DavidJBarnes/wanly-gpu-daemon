from typing import Optional
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
    sampler_name: Optional[str] = None       # KSampler sampler (None/"" -> daemon default euler)
    scheduler: Optional[str] = None          # KSampler scheduler (None/"" -> simple)
    negative_prompt: Optional[str] = None
    reprocess_type: Optional[str] = None
    output_path: Optional[str] = None
    # Retired engine selectors, still sent by wanly-api on older jobs. Kept on the schema so
    # a claim carrying them still parses; execute_segment rejects "lynx" and ignores "vace".
    continuation_mode: str = "traditional"
    generation_engine: Optional[str] = None
    previous_output_path: Optional[str] = None
    # Seed re-anchor: when True (resolved API-side — setting on AND this segment has a
    # successor), faceswap the extracted last frame to the canonical identity face before
    # it seeds the next i2v segment. Falls back to the raw seed if no face is detected.
    seed_faceswap: bool = False
    # AR hologram (reprocess_type="ar_hologram"): source = the job's finalized stitched video.
    hologram_source_path: Optional[str] = None
    hologram_key_color: Optional[str] = None
    hologram_subject_height_m: Optional[float] = None
    hologram_flavor: Optional[str] = None  # "2d_matte" (default) or "2.5d_depth"
    hologram_depth_scale_m: Optional[float] = None  # relief depth override (2.5d only)
    # Foundry smashcut (reprocess_type="smashcut_concat"): ordered clip paths + transition.
    smashcut_clip_paths: Optional[list[str]] = None
    smashcut_transition: Optional[str] = None
    # Per-clip playback speed, aligned 1:1 with smashcut_clip_paths. None = no retiming.
    smashcut_clip_speeds: Optional[list[float]] = None
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
