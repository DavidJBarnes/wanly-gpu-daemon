"""Execute a segment on LTX 2.3, via ltx-engine.

Kept in its own module rather than branching inside executor.py, so that retiring WAN 2.2 is
deleting files rather than unpicking merged functions.

What is DIFFERENT from the WAN path is only the middle: instead of building a ComfyUI graph
and driving ComfyUI over a websocket, this hands the job to ltx-engine, which owns graph
assembly and recipe resolution.

What is the SAME is everything either side of that, and deliberately so — the start image
still comes from the queue, and the finished mp4 still gets its last frame extracted, its
motion measured, its identity scored and its bytes uploaded through exactly the same calls.
That is what makes an LTX render appear in Videos, carry identity chips and accept
observations without any of that machinery knowing which engine produced it.
"""

import asyncio
import logging
import time

from daemon.executor import (
    _download_with_retry,
    _extract_last_frame,
    _score_segment_identity,
    _validate_image_data,
)
from daemon.ltx_client import LtxClient, LtxEngineError
from daemon.motion_analyzer import measure_motion_series
from daemon.progress import ProgressLog
from daemon.queue_client import QueueClient
from daemon.schemas import SegmentClaim, SegmentResult

logger = logging.getLogger(__name__)


async def _start_image_bytes(segment: SegmentClaim, queue: QueueClient) -> bytes | None:
    """The start frame as raw bytes.

    The WAN path uploads it to ComfyUI and passes a filename; the engine takes a data URI in
    the submit payload instead, so there is nothing to upload and no ComfyUI involved.
    """
    ref = segment.start_image
    if not ref:
        return None
    if not ref.startswith("s3://"):
        # A bare ComfyUI filename means a WAN-shaped claim reached the LTX path. There is no
        # ComfyUI upload area to read it back out of, so say so rather than rendering t2v and
        # returning a clip of the wrong person.
        raise LtxEngineError(
            f"start_image {ref!r} is not an s3:// path — the LTX engine needs the image "
            "itself, not a ComfyUI filename"
        )
    data = await _download_with_retry(lambda: queue.download_file(ref), "start_image")
    _validate_image_data(data, "start_image")
    return data


async def execute_ltx_segment(segment: SegmentClaim, queue: QueueClient) -> None:
    """Render one segment on ltx-engine and report the result."""
    recipe = segment.ltx_recipe or {}
    logger.info(
        "=== Segment %d (job %s) on LTX === %s | prompt: %s",
        segment.index, str(segment.job_id)[:8],
        f"recipe={recipe.get('recipe')!r} character={recipe.get('character')!r}"
        if recipe else "free-form",
        segment.prompt[:80],
    )

    progress = ProgressLog(segment.id, queue)
    started = time.monotonic()
    client = LtxClient()

    try:
        await progress.log("[1/6] Downloading start image...")
        image_bytes = await _start_image_bytes(segment, queue)
        await progress.log(
            f"[1/6] Start image ready ({len(image_bytes)} bytes)" if image_bytes
            else "[1/6] No start image (text-to-video)"
        )

        # Frames come from the recipe when there is one. The queue speaks seconds, so fall
        # back to duration x fps — but a recipe's own frame count wins, because it is part of
        # the configuration that was validated.
        num_frames = recipe.get("frames") or round(segment.duration_seconds * segment.fps)

        await progress.log("[2/6] Submitting to ltx-engine...")
        job_id = await client.submit(
            image_bytes=image_bytes,
            prompt=segment.prompt,
            negative_prompt=segment.negative_prompt,
            width=segment.width,
            height=segment.height,
            num_frames=int(num_frames),
            frame_rate=segment.fps,
            seed=segment.seed,
            recipe=recipe or None,
        )
        await progress.log(f"[3/6] Queued as {job_id}")

        job = await client.wait(job_id, progress=progress)

        # The engine's notes carry the recipe name and the resolved graph hash. That hash is
        # the regression trail — it is what makes this render provably the configuration that
        # was signed off — so it goes in the segment's own log, not just the engine's.
        for note in job.get("notes") or []:
            await progress.log(f"[4/6] {note}")

        await progress.log("[5/6] Downloading video...")
        video_data = await client.fetch_video(job_id)
        await progress.log(f"[5/6] Video downloaded ({len(video_data) / 1e6:.1f} MB)")

        await progress.log("[6/6] Extracting last frame and uploading...")
        last_frame_data = await _extract_last_frame(video_data)

        # Identical to the WAN path from here. Motion is synchronous OpenCV over every frame
        # (~38s at 289 frames), so it goes off the event loop or the heartbeat stops with it.
        motion_magnitude, motion_series = await asyncio.to_thread(
            measure_motion_series, video_data
        )
        if motion_magnitude:
            await progress.log(f"[6/6] Motion magnitude: {motion_magnitude:.2f} px/frame")

        # The shared scorer defaults to the WAN path's "[7/7]"; this path has six steps,
        # and a progress log that counts 1..6 then prints 7/7 reads as a missing step.
        identity_fields = await _score_segment_identity(
            segment, video_data, queue, progress, step="[6/6]"
        )
        metrics = identity_fields.get("identity_metrics")
        if isinstance(metrics, dict) and motion_series:
            metrics["series_motion"] = motion_series

        await queue.upload_segment_output(
            segment.id, video_data, last_frame_data,
            SegmentResult(status="completed", motion_magnitude=motion_magnitude,
                          **identity_fields),
        )
        logger.info("Segment %d complete in %.1fs", segment.index, time.monotonic() - started)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("LTX segment %s failed", segment.id)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000],
                          progress_log=progress.text),
        )
    finally:
        await client.close()
