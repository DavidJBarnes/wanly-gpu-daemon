import asyncio
import logging
import uuid
from typing import Optional

import httpx

from daemon.config import settings
from daemon.schemas import SegmentClaim, SegmentResult

logger = logging.getLogger(__name__)


def _redact_url(url) -> str:
    """Host and path only — never the query string.

    Presigned S3 URLs carry AWSAccessKeyId, Signature and x-amz-security-token in the query.
    They are short-lived, but daemon.log is bind-mounted to the host and gets pasted into
    tickets, so they must not be written there at all.
    """
    try:
        return f"{url.scheme}://{url.host}{url.path}"
    except Exception:
        return "<url unavailable>"


def _raise_with_details(resp: httpx.Response, context: str) -> None:
    """Log HTTP error details before raising, without leaking a signed URL.

    httpx builds raise_for_status()'s message from resp.url. After follow_redirects that is the
    presigned S3 URL, so the default exception put the full credential set into the log and into
    every traceback. The message is rebuilt here instead. See #112.
    """
    try:
        body = resp.text[:500]
    except Exception:
        body = "<unreadable>"
    where = _redact_url(resp.url)
    logger.error("%s — HTTP %d for %s: %s", context, resp.status_code, where, body)
    raise httpx.HTTPStatusError(
        f"{context} — HTTP {resp.status_code} for {where}",
        request=resp.request,
        response=resp,
    )


class QueueClient:
    """HTTP client for the wanly-api (queue + worker registry)."""

    def __init__(self):
        self.base_url = settings.queue_url
        headers = {}
        if settings.queue_api_key:
            headers["X-API-Key"] = settings.queue_api_key
        # keepalive_expiry below the far end's idle timeout: httpx otherwise hands out a
        # connection the server has already closed, and only finds out after writing the
        # request — surfacing as "Server disconnected without sending a response". Retiring
        # them locally first removes most of that class. See #105.
        limits = httpx.Limits(max_keepalive_connections=5, keepalive_expiry=30.0)
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=10, headers=headers, limits=limits
        )

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue a request, retrying once on a transport-level failure.

        A reaped keepalive connection fails instantly and succeeds instantly on a fresh socket,
        so one retry is the whole fix — there is no backoff to tune. Deliberately narrow: only
        transport errors retry. An HTTP error status is a real answer and must not be repeated,
        since claiming is not idempotent.
        """
        try:
            return await self.client.request(method, url, **kwargs)
        except (httpx.TransportError, httpx.RemoteProtocolError) as e:
            logger.info(
                "%s %s failed at the transport layer (%s); retrying once on a fresh connection",
                method, url, type(e).__name__,
            )
            return await self.client.request(method, url, **kwargs)

    async def claim_next(
        self, worker_id: uuid.UUID, worker_name: str | None = None, kind: str | None = None
    ) -> Optional[SegmentClaim]:
        """Claim the next available segment. Returns None if no work available.

        kind="gpu" claims generations only, kind="hologram" claims CPU-only hologram carriers
        only (so the two can run concurrently); None claims either.
        """
        params = {"worker_id": str(worker_id), "worker_name": worker_name or settings.friendly_name}
        if kind:
            params["kind"] = kind
        resp = await self._request_with_retry("GET", "/segments/next", params=params)
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

        if result and result.motion_magnitude:
            await self.update_segment(segment_id, result)

    async def upload_hologram_output(
        self, segment_id: uuid.UUID, video_data: bytes, manifest_data: bytes, poster_data: bytes
    ) -> None:
        """Upload AR hologram artifacts (packed color+alpha mp4 + manifest + poster) to the API."""
        resp = await self.client.post(
            f"/segments/{segment_id}/hologram_upload",
            files={
                "video": ("hologram.mp4", video_data, "video/mp4"),
                "manifest": ("hologram.json", manifest_data, "application/json"),
                "poster": ("poster.png", poster_data, "image/png"),
            },
            timeout=300,
        )
        if not resp.is_success:
            _raise_with_details(resp, f"upload_hologram_output {segment_id}")
        logger.info("Uploaded hologram output via API for %s", segment_id)

    async def upload_smashcut_output(self, segment_id: uuid.UUID, video_data: bytes) -> None:
        """Upload the concatenated smashcut video; the API creates the container job's Video."""
        resp = await self.client.post(
            f"/segments/{segment_id}/smashcut_upload",
            files={"video": ("smashcut.mp4", video_data, "video/mp4")},
            timeout=300,
        )
        if not resp.is_success:
            _raise_with_details(resp, f"upload_smashcut_output {segment_id}")
        logger.info("Uploaded smashcut output via API for %s", segment_id)


    async def download_file(self, s3_path: str) -> bytes:
        """Download a file from S3 via a presigned URL redirect.

        Uses a *granular* timeout instead of a flat 600s: a stalled read fails in
        ~60s (the max gap allowed between chunks) rather than pinning an idle GPU
        for 10 minutes, while a legitimately slow-but-steady transfer still
        completes. Transient stalls/network errors are retried a few times with
        backoff; HTTP status errors (4xx/5xx) are not retried and surface at once.
        """
        large = s3_path.endswith(".safetensors") or s3_path.endswith(".pth")
        # connect fast; `read` caps the gap between chunks (stall detector), not the
        # total transfer time, so big files download fine as long as bytes keep flowing.
        timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=15.0)
        attempts = 3 if large else 2
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = await self.client.get(
                    "/files", params={"path": s3_path}, timeout=timeout,
                    follow_redirects=True,
                )
                if not resp.is_success:
                    _raise_with_details(resp, f"download_file {s3_path}")
                return resp.content
            except httpx.TransportError as exc:
                # TransportError covers timeouts + connect/read/network errors.
                last_exc = exc
                logger.warning(
                    "download_file %s attempt %d/%d failed (%s): %s",
                    s3_path, attempt, attempts, type(exc).__name__, exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(2 * attempt)
        raise RuntimeError(
            f"download_file {s3_path} failed after {attempts} attempts: {last_exc}"
        )

    # --- Worker registry methods (formerly in RegistryClient) ---

    async def register(
        self,
        friendly_name: str,
        hostname: str,
        ip_address: str,
        comfyui_running: bool,
    ) -> tuple[uuid.UUID, str]:
        # The pod id is how the console pairs this worker with the RunPod pod behind it.
        # Pairing on name only worked for pods the console launcher created; a pod started from
        # the template has an auto-generated name and showed as "Starting" forever beside the
        # worker it had already become. Empty on anything self-hosted.
        payload = {
            "friendly_name": friendly_name,
            "hostname": hostname,
            "ip_address": ip_address,
            "comfyui_running": comfyui_running,
        }
        if settings.runpod_pod_id:
            payload["runpod_pod_id"] = settings.runpod_pod_id

        resp = await self.client.post(
            "/workers",
            json=payload,
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
        resp = await self._request_with_retry(
            "POST", f"/workers/{worker_id}/heartbeat", json=payload
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
