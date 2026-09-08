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
from daemon.ltx_client import LtxClient, LtxEngineError, character_lora_names
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


def _engine_detail(job: dict) -> list[str]:
    """The parts of the engine's job view worth putting in the segment's own record.

    FUSION COVERAGE is the one that matters. A LoRA whose keys do not line up against the
    checkpoint fuses NOTHING and says nothing about it — no error, no warning, the run looks
    completely normal and comes back as the base model with none of the character in it.
    Now that a pose can choose its own base model (console#404), "did it fuse?" is a routine
    question, and 480/480 is exactly as informative as 0/480. Only the zero case was ever
    promoted anywhere a person could see it; both belong on the segment.

    STAGES say what each pass actually ran — read off the resolved graph by the engine, not
    echoed from the request, so a step count the workflow could not express shows up here
    rather than being silently ignored.

    Deliberately quiet when there is nothing to say: no LoRAs means no lines, so this does
    not add noise to renders that have none.
    """
    out: list[str] = []

    for lo in job.get("loras") or []:
        name = lo.get("name", "?")
        fused, targeted = lo.get("fused"), lo.get("targeted")
        s1, s2 = lo.get("strength_stage_1"), lo.get("strength_stage_2")
        at = f" @{s1}/{s2}" if s1 is not None and s2 is not None else ""
        if fused is None or targeted is None:
            # The engine could not read the checkpoint to compare against. Said out loud,
            # because "no coverage line" and "coverage of zero" must not look the same.
            out.append(f"lora {name}{at}: fusion coverage unavailable")
        elif fused == 0:
            out.append(f"lora {name}{at}: FUSED 0/{targeted} — this render carries NONE of it")
        else:
            out.append(f"lora {name}{at}: fused {fused}/{targeted} weights")

    stages = job.get("stages") or []
    if stages:
        parts = []
        for st in stages:
            n = st.get("stage", "?")
            steps = st.get("steps")
            sched = st.get("schedule", "")
            parts.append(f"stage {n}: {steps} steps ({sched})" if steps is not None
                         else f"stage {n}: {sched}")
        out.append("passes — " + ", ".join(parts))

    return out

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
        # progress.log is passed in so the DOWNLOAD is visible on the job page, not just its
        # completion. A 400 MB LoRA takes minutes; without this the segment sits at
        # "[1/6] Start image ready" the whole time and is indistinguishable from a wedged
        # worker. See console#392.
        # content_loraS, plural. Content LoRAs became a stacked list in console#410 and every
        # other reader moved with it -- the API returns content_loras, the console sends it,
        # and ltx_client builds the engine payload from it. This line kept reading the old
        # singular key, so it always saw None and fetched nothing (#176).
        #
        # It went unseen because the 3090 already holds every content LoRA on disk, so the
        # boot sync's "deferred (content, fetched on demand)" promise was never actually
        # tested there. On a fresh pod the deferral is real, and the engine was handed a name
        # for a file nobody had downloaded: 422 no such lora, after the claim.
        content_names = [e.get("name") for e in (recipe.get("content_loras") or [])
                         if isinstance(e, dict)]
        # Every person's LoRA, not just the first: a two-person shot (console#473) names
        # two, and a second one this box has never seen would otherwise 422 at the engine
        # after the claim, exactly as the content LoRAs used to.
        character_names = character_lora_names(recipe)
        fetched = await ensure_named_loras_present(
            [*character_names, *content_names], queue,
            progress=progress.log,
        )
        if fetched:
            await progress.log(f"[2/6] LoRA(s) ready: {', '.join(fetched)}")

        # And the base model, same idea at 70x the size (console#423). Deliberately AFTER
        # the LoRAs: those are seconds each and this can be twenty minutes, so a pose that
        # is going to fail on a missing LoRA should fail before the long download, not after.
        #
        # ComfyUI re-scans diffusion_models on every object_info request -- verified on a
        # live pod mid-render -- so a file that lands here is visible to the render that
        # follows, with no restart.
        from daemon.checkpoint_sync import ensure_checkpoint_present
        if await ensure_checkpoint_present(recipe.get("checkpoint"), queue,
                                           progress=progress.log):
            # Tell the API what this worker holds NOW. Without this the model gate keeps
            # routing around a checkpoint the box has just acquired.
            #
            # Imported here, not at module scope: main imports the executor chain, and the
            # reported-inventory state lives in main. executor.py takes the same approach
            # for the same reason.
            from daemon.main import refresh_reported_checkpoints
            await refresh_reported_checkpoints()

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

        # The engine reports more than its notes, and until now the rest was dropped on the
        # floor — readable only by shelling into the container and reading
        # /workspace/logs/ltx-engine.log. That is how answering "did the character LoRA
        # actually apply on this render?" became a docker exec (console#392).
        for line in _engine_detail(job):
            await progress.log(f"[4/6] {line}")

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

        # Reclaim the engine's local copies now that S3 has them (console#380). AFTER the
        # upload, never before — the local file is the only copy until that call returns.
        #
        # Failure here is logged and swallowed. The segment is already complete and
        # uploaded; turning a successful render into a failure over disk housekeeping would
        # be a much worse outcome than 5 MB left behind, which the next sweep collects
        # anyway.
        try:
            purged = await client.purge(job_id)
            if purged.get("removed"):
                logger.info("Purged %d local file(s), %.1f MB reclaimed",
                            len(purged["removed"]), purged.get("freed_bytes", 0) / 1e6)
        except Exception as e:
            logger.warning("Could not purge engine job %s (%s) — leaving it for the sweep",
                           job_id, e)

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
