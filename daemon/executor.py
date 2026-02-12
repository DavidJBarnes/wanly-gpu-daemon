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
import logging
import os
import tempfile

from daemon.comfyui_client import ComfyUIClient, ComfyUIExecutionError
from daemon.lora_sync import ensure_loras_available
from daemon.queue_client import QueueClient
from daemon.schemas import SegmentClaim, SegmentResult
from daemon.workflow_builder import build_workflow

logger = logging.getLogger(__name__)


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
        data = await queue.download_file(start_image)
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
        data = await queue.download_file(faceswap_image)
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

    # Report processing status
    await queue.update_segment(
        segment.id,
        SegmentResult(status="processing"),
    )

    try:
        # Step 1: Resolve start image
        logger.info("[1/7] Downloading start image...")
        start_image_filename = await _resolve_start_image(segment, comfyui, queue)
        if start_image_filename:
            logger.info("[1/7] Start image ready: %s", start_image_filename)
        else:
            logger.info("[1/7] No start image (text-to-video)")

        # Step 1b: Resolve faceswap image
        faceswap_comfyui_filename = await _resolve_faceswap_image(segment, comfyui, queue)
        if faceswap_comfyui_filename:
            logger.info("[1/7] Faceswap image ready: %s", faceswap_comfyui_filename)
            segment = segment.model_copy(update={"faceswap_image": faceswap_comfyui_filename})

        # Step 2: Ensure LoRA files
        if segment.loras:
            logger.info("[2/7] Syncing LoRA files...")
            await ensure_loras_available(segment.loras, queue)
            logger.info("[2/7] LoRAs ready")
        else:
            logger.info("[2/7] No LoRAs")

        # Step 3: Build workflow
        logger.info("[3/7] Building workflow...")
        workflow = build_workflow(segment, start_image_filename=start_image_filename)
        logger.info("[3/7] Workflow built (%d nodes)", len(workflow))

        # Step 4: Submit to ComfyUI
        logger.info("[4/7] Submitting to ComfyUI...")
        prompt_id, client_id = await comfyui.submit_workflow(workflow)
        logger.info("[4/7] Submitted (prompt_id=%s)", prompt_id[:8])

        # Step 5: Wait for ComfyUI execution
        logger.info("[5/7] Waiting for ComfyUI execution...")
        await comfyui.monitor_execution(prompt_id, client_id)

        # Step 6: Download output
        logger.info("[6/7] Downloading output video...")
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
        logger.info("[6/7] Video downloaded: %s (%.1f MB)", video_info["filename"], video_mb)

        # Step 7: Extract last frame + upload
        logger.info("[7/7] Extracting last frame and uploading...")
        last_frame_data = await _extract_last_frame(video_data)
        await queue.upload_segment_output(segment.id, video_data, last_frame_data)

        logger.info("=== Segment %d complete ===", segment.index)

    except ComfyUIExecutionError as e:
        error_msg = f"ComfyUI error: {e}"
        if e.node_id:
            error_msg += f" (node {e.node_id} [{e.node_type}])"
        logger.error(error_msg)
        if e.traceback:
            logger.error("Traceback:\n%s", e.traceback)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000]),
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("Segment %s failed", segment.id)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000]),
        )
