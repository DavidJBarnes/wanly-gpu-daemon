import logging
import uuid
from typing import Optional

import httpx

from daemon.config import settings
from daemon.schemas import SegmentClaim, SegmentResult

logger = logging.getLogger(__name__)


def _raise_with_details(resp: httpx.Response, context: str) -> None:
    """Log HTTP error details before raising."""
    try:
        body = resp.text[:500]
    except Exception:
        body = "<unreadable>"
    logger.error("%s — HTTP %d: %s", context, resp.status_code, body)
    resp.raise_for_status()


class QueueClient:
    """HTTP client for the wanly-api (queue + worker registry)."""

    def __init__(self):
        self.base_url = settings.queue_url
        headers = {}
        if settings.queue_api_key:
            headers["X-API-Key"] = settings.queue_api_key
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=10, headers=headers)

    async def claim_next(self, worker_id: uuid.UUID, worker_name: str | None = None) -> Optional[SegmentClaim]:
        """Claim the next available segment. Returns None if no work available."""
        resp = await self.client.get(
            "/segments/next",
            params={"worker_id": str(worker_id), "worker_name": worker_name or settings.friendly_name},
        )
        if not resp.is_success:
            _raise_with_details(resp, "claim_next")
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
        if not resp.is_success:
            _raise_with_details(resp, f"update_segment {segment_id}")

    async def upload_segment_output(
        self, segment_id: uuid.UUID, video_data: bytes, last_frame_data: bytes, result: SegmentResult | None = None
    ) -> None:
        """Upload video + last frame to the API, which stores them in S3."""
        resp = await self.client.post(
            f"/segments/{segment_id}/upload",
            files={
                "video": ("output.mp4", video_data, "video/mp4"),
                "last_frame": ("last_frame.png", last_frame_data, "image/png"),
            },
            timeout=300,
        )
        if not resp.is_success:
            _raise_with_details(resp, f"upload_segment_output {segment_id}")
        logger.info("Uploaded segment output via API for %s", segment_id)

        if result and (result.motion_keywords or result.motion_magnitude):
            await self.update_segment(segment_id, result)

    async def download_file(self, s3_path: str) -> bytes:
        """Download a file from S3 via a presigned URL redirect."""
        large = s3_path.endswith(".safetensors") or s3_path.endswith(".pth")
        timeout = 600 if large else 60
        resp = await self.client.get(
            "/files", params={"path": s3_path}, timeout=timeout,
            follow_redirects=True,
        )
        if not resp.is_success:
            _raise_with_details(resp, f"download_file {s3_path}")
        return resp.content

    # --- Worker registry methods (formerly in RegistryClient) ---

    async def register(
        self,
        friendly_name: str,
        hostname: str,
        ip_address: str,
        comfyui_running: bool,
    ) -> tuple[uuid.UUID, str]:
        resp = await self.client.post(
            "/workers",
            json={
                "friendly_name": friendly_name,
                "hostname": hostname,
                "ip_address": ip_address,
                "comfyui_running": comfyui_running,
            },
        )
        if not resp.is_success:
            _raise_with_details(resp, "register")
        data = resp.json()
        return uuid.UUID(data["id"]), data["friendly_name"]

    async def heartbeat(
        self,
        worker_id: uuid.UUID,
        comfyui_running: bool,
        gpu_stats: dict | None = None,
        sd_scripts_status: dict | None = None,
        a1111_status: dict | None = None,
    ) -> dict:
        """Send heartbeat. Returns full worker data including current friendly_name."""
        payload: dict = {"comfyui_running": comfyui_running}
        if gpu_stats is not None:
            payload["gpu_stats"] = gpu_stats
        if sd_scripts_status is not None:
            payload["sd_scripts"] = sd_scripts_status
        if a1111_status is not None:
            payload["a1111"] = a1111_status
        resp = await self.client.post(
            f"/workers/{worker_id}/heartbeat",
            json=payload,
        )
        if not resp.is_success:
            _raise_with_details(resp, "heartbeat")
        return resp.json()

    async def update_status(self, worker_id: uuid.UUID, status: str):
        resp = await self.client.patch(
            f"/workers/{worker_id}/status",
            json={"status": status},
        )
        if not resp.is_success:
            _raise_with_details(resp, f"update_status {status}")

    async def deregister(self, worker_id: uuid.UUID):
        resp = await self.client.delete(f"/workers/{worker_id}")
        if not resp.is_success:
            _raise_with_details(resp, "deregister")

    async def close(self):
        await self.client.aclose()
