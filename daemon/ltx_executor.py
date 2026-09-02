"""Execute a segment on LTX 2.3, via ltx-engine.

Kept in its own module rather than branching inside executor.py, so that retiring WAN 2.2 is
deleting files rather than unpicking merged functions.

What is DIFFERENT from the WAN path is only the middle: instead of building a ComfyUI graph
and driving ComfyUI over a websocket, this hands the job to ltx-engine, which owns graph
assembly and recipe resolution.

What is the SAME is everything either side of that, and deliberately so — the start image
still comes from the queue, and the finished mp4 still gets its last frame extracted and its
bytes uploaded through exactly the same calls. That is what makes an LTX render appear in
Videos and accept observations without any of that machinery knowing which engine produced
it.
"""

import logging
import time

from daemon.executor import (
    _download_with_retry,
    _extract_last_frame,
    _validate_image_data,
)
from daemon.lora_sync import ensure_named_loras_present
from daemon.ltx_client import LtxClient, LtxEngineError
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

        # A LoRA the pose names may have been published AFTER this worker booted — the boot
        # sync is a snapshot, and the console offers a LoRA the moment it reaches the bucket.
        # Fetching it here turns "this segment fails until someone restarts the pod" into a
        # one-off download. Costs a stat() per LoRA in the normal case, where both are
        # already on disk.
        fetched = await ensure_named_loras_present(
            [recipe.get("char_lora"), recipe.get("content_lora")], queue
        )
        if fetched:
            await progress.log(f"[2/6] Fetched missing LoRA(s): {', '.join(fetched)}")

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

        # No identity scoring, no motion analysis. Measured on a 241-frame render they cost
        # 326s and 15-39s against a 263s render — post-processing outweighed the render it
        # analysed.
        #
        # They existed to compensate for WAN 2.2 drifting: measure the damage, then re-roll
        # or re-anchor against the measurement. LTX holds identity from the character LoRA
        # and the start frame, so there is nothing to compensate for, and the metrics were
        # never trustworthy anyway — expression rewards the mouth-gape artifact it should
        # penalise, and motion scored a 5-rated segment BELOW a 3-rated one. Human ratings
        # are the judgement that counts. See #151.
        await queue.upload_segment_output(
            segment.id, video_data, last_frame_data,
            SegmentResult(status="completed"),
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
