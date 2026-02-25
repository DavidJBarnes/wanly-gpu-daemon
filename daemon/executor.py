"""Orchestrate full segment execution pipeline.

Steps:
1. Build workflow from segment params
2. Download start_image from API, upload to ComfyUI
3. Submit workflow → get prompt_id
4. Monitor via WebSocket → wait for success/error
5. Poll /history for output filenames
6. Download video from ComfyUI
7. Extract last frame via ffmpeg
8. Upload video + last frame via API (API stores in S3)
9. Done — API marks segment completed
On error → report failure to queue
"""

import asyncio
import io
import logging
import os
import tempfile
import time

import httpx
from PIL import Image

from daemon.comfyui_client import ComfyUIClient, ComfyUIExecutionError
from daemon.lora_sync import ensure_loras_available
from daemon.progress import ProgressLog
from daemon.queue_client import QueueClient
from daemon.schemas import SegmentClaim, SegmentResult
from daemon.workflow_builder import build_workflow

logger = logging.getLogger(__name__)


def _validate_image_data(data: bytes, label: str) -> None:
    """Validate downloaded image data is a real, decodable image.

    Raises RuntimeError if the data is empty, too small, or not a valid image.
    """
    if not data:
        raise RuntimeError(f"{label}: empty data")
    size_kb = len(data) / 1024
    if size_kb < 1:
        raise RuntimeError(f"{label}: too small ({size_kb:.1f} KB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        # Re-open after verify to get dimensions (verify consumes the stream)
        img = Image.open(io.BytesIO(data))
        w, h = img.size
    except Exception as e:
        raise RuntimeError(f"{label}: invalid image — {e}")
    logger.info("  %s: %dx%d, %.0f KB", label, w, h, size_kb)


async def _download_with_retry(coro_factory, label: str, attempts: int = 3, delay: float = 2.0) -> bytes:
    """Retry a download coroutine on transient network errors.

    coro_factory must be a callable that returns a fresh coroutine each call.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as e:
            last_exc = e
            if attempt < attempts:
                logger.warning("%s download failed (attempt %d/%d): %s — retrying in %.0fs", label, attempt, attempts, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("%s download failed after %d attempts: %s", label, attempts, e)
    raise last_exc  # type: ignore[misc]


async def _resolve_start_image(
    segment: SegmentClaim, comfyui: ComfyUIClient, queue: QueueClient
) -> str | None:
    """Resolve the start_image field to a ComfyUI-local filename.

    Returns None for text-to-video (no start image).
    Downloads from API and uploads to ComfyUI if it's an S3 path.
    """
    start_image = segment.start_image
    if not start_image:
        return None

    if start_image.startswith("s3://"):
        logger.info("Downloading start image via API: %s", start_image)
        data = await _download_with_retry(
            lambda: queue.download_file(start_image), "start_image"
        )
        _validate_image_data(data, "start_image")
        ext = os.path.splitext(start_image)[1] or ".png"
        filename = f"segment_{segment.id}{ext}"
        comfyui_filename = await comfyui.upload_image(data, filename)
        logger.info("Uploaded start image to ComfyUI as: %s", comfyui_filename)
        return comfyui_filename

    # Already a ComfyUI filename
    return start_image


async def _resolve_faceswap_image(
    segment: SegmentClaim, comfyui: ComfyUIClient, queue: QueueClient
) -> str | None:
    """If faceswap_image is an S3 path, download via API and upload to ComfyUI."""
    if not segment.faceswap_enabled or not segment.faceswap_image:
        return None

    faceswap_image = segment.faceswap_image
    if faceswap_image.startswith("s3://"):
        logger.info("Downloading faceswap image via API: %s", faceswap_image)
        data = await _download_with_retry(
            lambda: queue.download_file(faceswap_image), "faceswap_image"
        )
        _validate_image_data(data, "faceswap_image")
        ext = os.path.splitext(faceswap_image)[1] or ".png"
        filename = f"faceswap_{segment.id}{ext}"
        comfyui_filename = await comfyui.upload_image(data, filename)
        logger.info("Uploaded faceswap image to ComfyUI as: %s", comfyui_filename)
        return comfyui_filename

    return faceswap_image


async def _extract_last_frame(video_data: bytes) -> bytes:
    """Extract the last frame from a video using ffmpeg.

    Uses -sseof to seek near the end and extract one frame as PNG.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        tmp_video.write(video_data)
        tmp_video_path = tmp_video.name

    tmp_frame_path = tmp_video_path.replace(".mp4", "_last_frame.png")

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-sseof", "-0.1",
            "-i", tmp_video_path,
            "-frames:v", "1",
            "-update", "1",
            tmp_frame_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

        with open(tmp_frame_path, "rb") as f:
            return f.read()
    finally:
        for path in (tmp_video_path, tmp_frame_path):
            try:
                os.unlink(path)
            except OSError:
                pass


async def execute_segment(
    segment: SegmentClaim,
    comfyui: ComfyUIClient,
    queue: QueueClient,
) -> None:
    """Execute a single segment end-to-end."""
    lora_names = ", ".join(l.high_file or l.low_file or "?" for l in (segment.loras or []))
    logger.info(
        "=== Segment %d (job %s) === prompt: %s%s",
        segment.index, str(segment.job_id)[:8],
        segment.prompt[:80],
        f" | loras: {lora_names}" if lora_names else "",
    )

    progress = ProgressLog(segment.id, queue)
    segment_start = time.monotonic()
    step_times: list[tuple[str, float]] = []

    try:
        # Step 1: Resolve start image
        t0 = time.monotonic()
        await progress.log("[1/7] Downloading start image...")
        start_image_filename = await _resolve_start_image(segment, comfyui, queue)
        if start_image_filename:
            await progress.log(f"[1/7] Start image ready: {start_image_filename}")
        else:
            await progress.log("[1/7] No start image (text-to-video)")

        # Step 1b: Resolve initial reference image (identity anchor for PainterLongVideo)
        initial_ref_filename = None
        if segment.initial_reference_image and segment.index > 0:
            ref_image = segment.initial_reference_image
            if ref_image.startswith("s3://"):
                logger.info("Downloading initial reference image via API: %s", ref_image)
                ref_data = await _download_with_retry(
                    lambda: queue.download_file(ref_image), "initial_reference_image"
                )
                _validate_image_data(ref_data, "initial_reference_image")
                ext = os.path.splitext(ref_image)[1] or ".png"
                ref_filename = f"initial_ref_{segment.id}{ext}"
                initial_ref_filename = await comfyui.upload_image(ref_data, ref_filename)
                await progress.log(f"[1/7] Initial reference image ready: {initial_ref_filename}")
            else:
                initial_ref_filename = ref_image

        # Step 1c: Resolve faceswap image
        faceswap_comfyui_filename = await _resolve_faceswap_image(segment, comfyui, queue)
        if faceswap_comfyui_filename:
            await progress.log(f"[1/7] Faceswap image ready: {faceswap_comfyui_filename}")
            segment = segment.model_copy(update={"faceswap_image": faceswap_comfyui_filename})
        step_times.append(("images", time.monotonic() - t0))

        # Step 2: Ensure LoRA files
        t0 = time.monotonic()
        if segment.loras:
            await progress.log("[2/7] Syncing LoRA files...")
            await ensure_loras_available(segment.loras, queue)
            await progress.log("[2/7] LoRAs ready")
        else:
            await progress.log("[2/7] No LoRAs")
        step_times.append(("loras", time.monotonic() - t0))

        # Step 3: Build workflow
        t0 = time.monotonic()
        await progress.log("[3/7] Building workflow...")
        workflow = build_workflow(
            segment,
            start_image_filename=start_image_filename,
            initial_reference_image_filename=initial_ref_filename,
        )
        await progress.log(f"[3/7] Workflow built ({len(workflow)} nodes)")
        step_times.append(("build", time.monotonic() - t0))

        # Step 4: Submit to ComfyUI
        t0 = time.monotonic()
        await progress.log("[4/7] Submitting to ComfyUI...")
        prompt_id, client_id = await comfyui.submit_workflow(workflow)
        await progress.log(f"[4/7] Submitted (prompt_id={prompt_id[:8]})")
        step_times.append(("submit", time.monotonic() - t0))

        # Step 5: Wait for ComfyUI execution
        t0 = time.monotonic()
        await progress.log("[5/7] Waiting for ComfyUI execution...")
        await comfyui.monitor_execution(prompt_id, client_id)
        await progress.log("[5/7] Execution complete")
        step_times.append(("execute", time.monotonic() - t0))

        # Step 6: Download output
        t0 = time.monotonic()
        await progress.log("[6/7] Downloading output video...")
        history = await comfyui.get_history(prompt_id)
        video_info = comfyui.find_video_output(history)
        if not video_info:
            raise RuntimeError("No video output found in ComfyUI history")

        video_data = await comfyui.download_output(
            filename=video_info["filename"],
            subfolder=video_info.get("subfolder", ""),
            output_type=video_info.get("type", "output"),
        )
        video_mb = len(video_data) / (1024 * 1024)
        await progress.log(f"[6/7] Video downloaded: {video_info['filename']} ({video_mb:.1f} MB)")
        step_times.append(("download", time.monotonic() - t0))

        # Step 7: Extract last frame + upload
        t0 = time.monotonic()
        await progress.log("[7/7] Extracting last frame and uploading...")
        last_frame_data = await _extract_last_frame(video_data)
        await queue.upload_segment_output(segment.id, video_data, last_frame_data)
        step_times.append(("upload", time.monotonic() - t0))

        total = time.monotonic() - segment_start
        timing_str = " | ".join(f"{name}={elapsed:.1f}s" for name, elapsed in step_times)
        logger.info(
            "Segment %d complete in %.1fs — %s",
            segment.index, total, timing_str,
        )

    except ComfyUIExecutionError as e:
        error_msg = f"ComfyUI error: {e}"
        if e.node_id:
            error_msg += f" (node {e.node_id} [{e.node_type}])"
        logger.error(error_msg)
        if e.traceback:
            logger.error("Traceback:\n%s", e.traceback)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000], progress_log=progress.text),
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("Segment %s failed", segment.id)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000], progress_log=progress.text),
        )
