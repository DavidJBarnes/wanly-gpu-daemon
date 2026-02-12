import logging
import uuid
from typing import Optional

import httpx

from daemon.config import settings
from daemon.schemas import SegmentClaim, SegmentResult

logger = logging.getLogger(__name__)


class QueueClient:
    def __init__(self):
        self.base_url = settings.queue_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=10)

    async def claim_next(self, worker_id: uuid.UUID) -> Optional[SegmentClaim]:
        """Claim the next available segment. Returns None if no work available."""
        resp = await self.client.get(
            "/segments/next",
            params={"worker_id": str(worker_id), "worker_name": settings.friendly_name},
        )
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            return None
        return SegmentClaim.model_validate(data)

    async def update_segment(
        self, segment_id: uuid.UUID, result: SegmentResult
    ) -> None:
        """Report segment status update (processing/failed)."""
        resp = await self.client.patch(
            f"/segments/{segment_id}",
            json=result.model_dump(exclude_none=True),
        )
        resp.raise_for_status()

    async def upload_segment_output(
        self, segment_id: uuid.UUID, video_data: bytes, last_frame_data: bytes
    ) -> None:
        """Upload video + last frame to the API, which stores them in S3."""
        resp = await self.client.post(
            f"/segments/{segment_id}/upload",
            files={
                "video": ("output.mp4", video_data, "video/mp4"),
                "last_frame": ("last_frame.png", last_frame_data, "image/png"),
            },
            timeout=300,  # Large uploads may take a while
        )
        resp.raise_for_status()
        logger.info("Uploaded segment output via API for %s", segment_id)

    async def download_file(self, s3_path: str) -> bytes:
        """Download a file from S3 via the API proxy."""
        timeout = 600 if ".safetensors" in s3_path else 60
        resp = await self.client.get(
            "/files", params={"path": s3_path}, timeout=timeout
        )
        resp.raise_for_status()
        return resp.content

    async def close(self):
        await self.client.aclose()
