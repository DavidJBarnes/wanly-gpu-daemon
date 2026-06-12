"""Shared fixtures for workflow_builder tests."""

from uuid import uuid4

import pytest

from daemon.schemas import SegmentClaim


@pytest.fixture
def make_segment():
    """Factory for SegmentClaim with sensible defaults; override via kwargs."""

    def _make(**overrides) -> SegmentClaim:
        defaults = {
            "id": uuid4(),
            "job_id": uuid4(),
            "index": 0,
            "prompt": "a calm lake at sunrise",
            "duration_seconds": 5.0,
            "faceswap_enabled": False,
            "width": 640,
            "height": 640,
            "fps": 30,
            "seed": 42,
        }
        defaults.update(overrides)
        return SegmentClaim(**defaults)

    return _make
