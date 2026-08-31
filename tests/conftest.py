"""Shared fixtures for the daemon test suite."""

from typing import Any
from uuid import UUID

import pytest

from daemon.schemas import LoraItem, SegmentClaim

# Fixed IDs so golden-file comparisons and log assertions are reproducible.
SEGMENT_ID = UUID("11111111-2222-3333-4444-555555555555")
JOB_ID = UUID("66666666-7777-8888-9999-000000000000")


def make_segment(**overrides: Any) -> SegmentClaim:
    """Build a SegmentClaim with sane defaults.

    Defaults describe an 832x480 clip whose duration lands exactly on 81 WAN frames
    (81 frames / 15 generation fps = 5.4s), which is the smoke-test shape.
    """
    base: dict[str, Any] = {
        "id": SEGMENT_ID,
        "job_id": JOB_ID,
        "index": 0,
        "prompt": "a woman walking through a forest",
        "duration_seconds": 5.4,
        "width": 832,
        "height": 480,
        "fps": 15,
        "seed": 42,
    }
    base.update(overrides)
    return SegmentClaim(**base)


@pytest.fixture
def segment() -> SegmentClaim:
    return make_segment()


@pytest.fixture
def lora() -> LoraItem:
    return LoraItem(
        lora_id="abc", high_file="k3lly_high.safetensors", high_weight=0.8,
        low_file="k3lly_low.safetensors", low_weight=0.6,
    )
