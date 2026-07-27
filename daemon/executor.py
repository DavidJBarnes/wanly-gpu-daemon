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
import glob
import io
import json
import logging
import os
import shutil
import tempfile
import time

import httpx
import numpy as np
from PIL import Image

from daemon.comfyui_client import ComfyUIClient, ComfyUIExecutionError
from daemon.lora_sync import ensure_loras_available
from daemon.motion_extractor import _augment_prompt_with_motion, extract_motion_keywords
from daemon.motion_analyzer import measure_motion_magnitude
from daemon.progress import ProgressLog
from daemon.queue_client import QueueClient
from daemon.schemas import SegmentClaim, SegmentResult
from daemon.workflow_builder import (
    GENERATION_FPS,
    build_workflow,
    build_faceswap_workflow,
    build_seed_faceswap_workflow,
    build_vace_workflow,
    vace_num_frames,
)
from daemon.config import settings
from daemon.hologram import (
    GUARD_PX,
    build_manifest,
    chroma_key_despill,
    detect_greenscreen,
    edge_extend_color,
    ensure_rvm_model,
    pack_sbs,
    pack_sbs_depth,
    rvm_matte,
    union_alpha_bbox,
)
from daemon.depth import ensure_depth_model, estimate_depth, normalize_depth_clip
from daemon.face_crop import extract_best_face

# HACK (temp): the auto face-crop for the VACE identity reference is unreliable, so feed VACE a
# known-good canonical face per character LoRA instead, keyed by char lora_id. Remove once the
# crop is fixed (or replaced with insightface).
_HARDCODE_VACE_FACE = {
    "b86c19a9-20aa-44de-85e9-aab2a4ca4809": "s3://wanly-loras/hardcode/k3lly2026_face.png",
    "5427b202-76da-4a15-aa8a-75e375c4d7d0": "s3://wanly-loras/hardcode/k3llydw_face.png",
}

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


def _find_image_output(history: dict) -> dict | None:
    """Find a SaveImage output (node 186) in ComfyUI history. Returns filename info or None."""
    outputs = history.get("outputs", {})
    node = outputs.get("186", {})
    if node.get("images"):
        return node["images"][0]
    for _nid, out in outputs.items():
        for im in out.get("images", []):
            return im
    return None


def _images_differ(a: bytes, b: bytes, threshold: float = 1.0) -> bool:
    """True if two PNGs differ perceptually. FaceFusion re-saves the SAME pixels (lossless
    PNG) when it finds no face, so a near-zero mean pixel diff => no swap happened."""
    import cv2
    import numpy as np
    ia = cv2.imdecode(np.frombuffer(a, np.uint8), cv2.IMREAD_COLOR)
    ib = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if ia is None or ib is None or ia.shape != ib.shape:
        return True
    return float(np.abs(ia.astype(np.int16) - ib.astype(np.int16)).mean()) > threshold


async def _reanchor_seed_frame(
    segment: SegmentClaim,
    last_frame_data: bytes,
    comfyui: ComfyUIClient,
    queue: QueueClient,
    progress: "ProgressLog",
) -> bytes:
    """Re-anchor the continuation seed (last frame) to the canonical identity face before
    it seeds the next i2v segment. FaceFusion passes the frame through unchanged when no
    face is detected, so we diff the result and fall back to the raw seed on no-op/error.
    NEVER raises — always returns a usable seed frame (swapped or raw), and logs which."""
    try:
        source_s3 = next(
            (_HARDCODE_VACE_FACE[l.lora_id] for l in (segment.loras or [])
             if l.lora_id in _HARDCODE_VACE_FACE),
            None,
        )
        if not source_s3:
            logger.info("Seed re-anchor: no canonical face for segment %d LoRAs — kept raw seed", segment.index)
            await progress.log("[seed] No canonical face for these LoRAs — kept raw frame")
            return last_frame_data

        face_data = await _download_with_retry(lambda: queue.download_file(source_s3), "seed_face")
        _validate_image_data(face_data, "seed_face")
        face_fn = await comfyui.upload_image(face_data, f"seedface_{segment.id}.png")
        seed_fn = await comfyui.upload_image(last_frame_data, f"seed_{segment.id}.png")

        seg = segment.model_copy(update={
            "faceswap_image": face_fn,
            "faceswap_method": segment.faceswap_method or "facefusion",
        })
        workflow = build_seed_faceswap_workflow(seg, seed_fn)
        prompt_id, client_id = await comfyui.submit_workflow(workflow)
        await comfyui.monitor_execution(prompt_id, client_id, progress=progress)
        history = await comfyui.get_history(prompt_id)
        img = _find_image_output(history)
        if not img:
            raise RuntimeError("no image output from seed faceswap")
        swapped = await comfyui.download_output(
            img["filename"], img.get("subfolder", ""), img.get("type", "output"),
        )
        _validate_image_data(swapped, "swapped_seed")

        if _images_differ(last_frame_data, swapped):
            logger.info("Seed re-anchor: identity faceswapped onto segment %d seed", segment.index)
            await progress.log("[seed] Re-anchored identity via faceswap")
            return swapped
        logger.info("Seed re-anchor: no face detected on segment %d seed — kept raw", segment.index)
        await progress.log("[seed] No face detected on seed — kept raw frame")
        return last_frame_data
    except Exception as e:  # graceful fallback — a bad swap must never break continuation
        logger.warning("Seed re-anchor failed on segment %d (%s) — kept raw seed", segment.index, e)
        await progress.log(f"[seed] Faceswap errored ({type(e).__name__}) — kept raw frame")
        return last_frame_data


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


async def _run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg (with -y) and raise RuntimeError on failure."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", *args,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-500:]}")


async def _trim_video_start(video_data: bytes, seconds: float) -> bytes:
    """Drop the first `seconds` from a video (frame-accurate, re-encoded). Returns bytes."""
    if seconds <= 0:
        return video_data
    tmpdir = tempfile.mkdtemp(prefix="trim_")
    try:
        src = os.path.join(tmpdir, "in.mp4")
        out = os.path.join(tmpdir, "out.mp4")
        with open(src, "wb") as f:
            f.write(video_data)
        # -ss AFTER -i = frame-accurate output seeking.
        await _run_ffmpeg(["-i", src, "-ss", f"{seconds:.4f}",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", out])
        with open(out, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _build_vace_control(
    video_data: bytes, keep_n: int, num_frames: int, width: int, height: int
) -> tuple[bytes, bytes]:
    """From a previous segment's video, build the VACE control clip (last keep_n frames at
    GENERATION_FPS + grey pad to num_frames) and the mask (black=keep tail, white=generate).
    The wrapper's WanVideoVACEEncode does NOT auto-pad, so both must be full num_frames long.
    Returns (control_mp4_bytes, mask_mp4_bytes)."""
    tmpdir = tempfile.mkdtemp(prefix="vace_")
    try:
        prev = os.path.join(tmpdir, "prev.mp4")
        with open(prev, "wb") as f:
            f.write(video_data)
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir)
        await _run_ffmpeg(["-i", prev, "-vf", f"fps={GENERATION_FPS},scale={width}:{height}",
                           os.path.join(frames_dir, "%05d.png")])
        allf = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not allf:
            raise RuntimeError("no frames extracted from previous segment video")
        keep_n = max(1, min(keep_n, len(allf)))
        tail_dir = os.path.join(tmpdir, "tail")
        os.makedirs(tail_dir)
        for i, src in enumerate(allf[-keep_n:]):
            shutil.copy(src, os.path.join(tail_dir, f"{i:05d}.png"))
        pad = max(num_frames - keep_n, 0)

        tail_clip = os.path.join(tmpdir, "tail.mp4")
        await _run_ffmpeg(["-framerate", str(GENERATION_FPS), "-i", os.path.join(tail_dir, "%05d.png"),
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", tail_clip])
        grey_clip = os.path.join(tmpdir, "grey.mp4")
        await _run_ffmpeg(["-f", "lavfi", "-i", f"color=c=0x808080:s={width}x{height}:r={GENERATION_FPS}",
                           "-frames:v", str(pad), "-c:v", "libx264", "-pix_fmt", "yuv420p", grey_clip])
        control = os.path.join(tmpdir, "control.mp4")
        cc = os.path.join(tmpdir, "cc.txt")
        with open(cc, "w") as f:
            f.write(f"file '{tail_clip}'\nfile '{grey_clip}'\n")
        await _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", cc, "-c:v", "libx264", "-pix_fmt", "yuv420p", control])

        black_clip = os.path.join(tmpdir, "black.mp4")
        await _run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={GENERATION_FPS}",
                           "-frames:v", str(keep_n), "-c:v", "libx264", "-pix_fmt", "yuv420p", black_clip])
        white_clip = os.path.join(tmpdir, "white.mp4")
        await _run_ffmpeg(["-f", "lavfi", "-i", f"color=c=white:s={width}x{height}:r={GENERATION_FPS}",
                           "-frames:v", str(pad), "-c:v", "libx264", "-pix_fmt", "yuv420p", white_clip])
        mask = os.path.join(tmpdir, "mask.mp4")
        mm = os.path.join(tmpdir, "mm.txt")
        with open(mm, "w") as f:
            f.write(f"file '{black_clip}'\nfile '{white_clip}'\n")
        await _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", mm, "-c:v", "libx264", "-pix_fmt", "yuv420p", mask])

        with open(control, "rb") as f:
            control_bytes = f.read()
        with open(mask, "rb") as f:
            mask_bytes = f.read()
        return control_bytes, mask_bytes
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _vace_capable(comfyui: ComfyUIClient) -> bool:
    """True if this worker's ComfyUI has the WanVideoWrapper VACE node + the Fun-VACE modules."""
    try:
        resp = await comfyui.http.get("/object_info/WanVideoVACEModelSelect")
        if resp.status_code != 200:
            return False
        info = resp.json().get("WanVideoVACEModelSelect", {})
        spec = info.get("input", {}).get("required", {}).get("vace_model", [[]])
        choices = spec[0] if spec and isinstance(spec[0], list) else []
        return settings.vace_module_high in choices and settings.vace_module_low in choices
    except Exception as e:
        logger.warning("VACE capability check failed (%s) — treating as not capable", e)
        return False


async def _execute_vace_continuation(
    segment: SegmentClaim,
    comfyui: ComfyUIClient,
    queue: QueueClient,
) -> None:
    """Generate a segment as a VACE video-conditioned continuation of the previous
    segment's tail (seamless), instead of the traditional single-frame i2v handoff."""
    logger.info("=== VACE continuation segment %d (job %s) === overlap=%d",
                segment.index, str(segment.job_id)[:8], segment.vace_overlap_frames)
    progress = ProgressLog(segment.id, queue)
    segment_start = time.monotonic()
    try:
        if not segment.previous_output_path:
            raise RuntimeError("VACE continuation requires previous_output_path")
        if segment.loras:
            await ensure_loras_available(segment.loras, queue)

        # 1. Download previous segment video (tail source)
        await progress.log("[1/7] Downloading previous segment video...")
        prev_data = await _download_with_retry(
            lambda: queue.download_file(segment.previous_output_path), "prev_video"
        )
        await progress.log(f"[1/7] Downloaded ({len(prev_data) / (1024 * 1024):.1f} MB)")

        # 2. Build control (kept tail + grey pad) + mask (keep + generate)
        await progress.log("[2/7] Building VACE control from tail...")
        base_frames = vace_num_frames(segment)
        keep_n = max(1, min(segment.vace_overlap_frames, base_frames - 4))
        # Generate keep_n extra (reconstructed lead-in) frames, trimmed after decode, so the
        # segment keeps its intended duration AND butts seamlessly onto the previous one.
        num_frames = ((base_frames + keep_n - 1) // 4) * 4 + 1
        control_data, mask_data = await _build_vace_control(
            prev_data, keep_n, num_frames, segment.width, segment.height
        )

        # 3. Reference image (identity anchor). Canonical seg0 face crop, persisted on the job and
        # reused for every downstream segment so identity re-anchors at each cut. On the FIRST
        # continuation the job has no crop yet (ref_source is the job's start image), so derive it
        # from seg0 (= prev_data here) and persist it for the segments that follow.
        ref_source = segment.initial_reference_image
        is_persisted_crop = bool(ref_source) and "identity_reference.png" in ref_source
        ref_data = None
        # HACK: if the segment uses a character LoRA we have a canonical face for, use that face
        # as the VACE identity reference (bypasses the unreliable auto-crop).
        hardcode_uri = next(
            (_HARDCODE_VACE_FACE[l.lora_id] for l in (segment.loras or [])
             if l.lora_id in _HARDCODE_VACE_FACE),
            None,
        )
        if hardcode_uri:
            await progress.log("[3/7] Using hardcoded canonical identity face (VACE ref override)")
            ref_data = await _download_with_retry(
                lambda: queue.download_file(hardcode_uri), "vace_ref_hardcode"
            )
        elif not is_persisted_crop:
            face_png = await asyncio.to_thread(extract_best_face, prev_data)
            if face_png:
                await progress.log("[3/7] Cropped canonical identity face from seg0")
                try:
                    await queue.upload_identity_reference(str(segment.job_id), face_png)
                except Exception:
                    logger.exception("identity reference upload failed (non-fatal)")
                ref_data = face_png
        if ref_data is None:
            if ref_source and ref_source.startswith("s3://"):
                ref_data = await _download_with_retry(
                    lambda: queue.download_file(ref_source), "vace_reference"
                )
            else:
                ref_data = await _extract_last_frame(prev_data)

        # 4. Upload control assets to ComfyUI
        await progress.log("[3/7] Uploading control assets to ComfyUI...")
        control_fn = await comfyui.upload_video(control_data, f"vace_control_{segment.id}.mp4")
        mask_fn = await comfyui.upload_video(mask_data, f"vace_mask_{segment.id}.mp4")
        ref_fn = await comfyui.upload_image(ref_data, f"vace_ref_{segment.id}.png")

        # 5. Build + submit
        await progress.log("[4/7] Building VACE workflow...")
        workflow = build_vace_workflow(segment, control_fn, mask_fn, ref_fn, num_frames)
        prompt_id, client_id = await comfyui.submit_workflow(workflow)
        await progress.log(f"[5/7] Submitted (prompt_id={prompt_id[:8]}, {num_frames} frames)")

        # 6. Monitor
        await progress.log("[6/7] Waiting for VACE execution...")
        await comfyui.monitor_execution(prompt_id, client_id, progress=progress)

        # 7. Download output, extract last frame, upload
        await progress.log("[7/7] Downloading result and uploading...")
        history = await comfyui.get_history(prompt_id)
        video_info = comfyui.find_video_output(history)
        if not video_info:
            raise RuntimeError("No video output found in ComfyUI history")
        result_data = await comfyui.download_output(
            filename=video_info["filename"],
            subfolder=video_info.get("subfolder", ""),
            output_type=video_info.get("type", "output"),
        )
        # Do NOT trim the reconstructed lead-in here: those keep_n frames are the smooth
        # bridge to the previous segment (the generation follows the reconstruction, which
        # drifts slightly from the previous segment's actual tail — dropping the bridge and
        # jumping to the first generated frame left a boundary hitch). Instead report the
        # overlap so stitch trims the *previous* segment's tail, letting this reconstruction
        # replace it (consecutive frames = seamless).
        last_frame_data = await _extract_last_frame(result_data)
        await queue.upload_segment_output(
            segment.id, result_data, last_frame_data,
            SegmentResult(status="completed", vace_overlap_seconds=keep_n / GENERATION_FPS),
        )
        logger.info("VACE continuation segment %d complete in %.1fs",
                    segment.index, time.monotonic() - segment_start)

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
        logger.exception("VACE continuation segment %s failed", segment.id)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000], progress_log=progress.text),
        )


async def _execute_faceswap_reprocess(
    segment: SegmentClaim,
    comfyui: ComfyUIClient,
    queue: QueueClient,
) -> None:
    """Apply faceswap to an existing completed video without regenerating it."""
    logger.info(
        "=== Faceswap reprocess segment %d (job %s) === method: %s",
        segment.index, str(segment.job_id)[:8], segment.faceswap_method or "facefusion",
    )

    progress = ProgressLog(segment.id, queue)
    segment_start = time.monotonic()

    try:
        if not segment.output_path:
            raise RuntimeError("No output_path on segment — cannot reprocess without existing video")
        # Step 1: Download existing video from S3
        await progress.log("[1/5] Downloading existing video...")
        video_data = await _download_with_retry(
            lambda: queue.download_file(segment.output_path), "existing_video"
        )
        video_mb = len(video_data) / (1024 * 1024)
        await progress.log(f"[1/5] Video downloaded ({video_mb:.1f} MB)")

        # Step 2: Upload video + faceswap image to ComfyUI
        await progress.log("[2/5] Uploading to ComfyUI...")
        video_filename = f"reprocess_{segment.id}.mp4"
        comfyui_video = await comfyui.upload_video(video_data, video_filename)
        logger.info("Uploaded existing video to ComfyUI as: %s", comfyui_video)

        faceswap_comfyui_filename = await _resolve_faceswap_image(segment, comfyui, queue)
        if faceswap_comfyui_filename:
            segment = segment.model_copy(update={"faceswap_image": faceswap_comfyui_filename})
        await progress.log("[2/5] Files ready in ComfyUI")

        # Step 3: Build and submit faceswap-only workflow
        await progress.log("[3/5] Building faceswap workflow...")
        workflow = build_faceswap_workflow(segment, comfyui_video)
        prompt_id, client_id = await comfyui.submit_workflow(workflow)
        await progress.log(f"[3/5] Submitted (prompt_id={prompt_id[:8]})")

        # Step 4: Wait for execution
        await progress.log("[4/5] Waiting for faceswap execution...")
        await comfyui.monitor_execution(prompt_id, client_id, progress=progress)
        await progress.log("[4/5] Execution complete")

        # Step 5: Download output, extract last frame, upload
        await progress.log("[5/5] Downloading result and uploading...")
        history = await comfyui.get_history(prompt_id)
        video_info = comfyui.find_video_output(history)
        if not video_info:
            raise RuntimeError("No video output found in ComfyUI history")

        result_data = await comfyui.download_output(
            filename=video_info["filename"],
            subfolder=video_info.get("subfolder", ""),
            output_type=video_info.get("type", "output"),
        )
        last_frame_data = await _extract_last_frame(result_data)

        segment_result = SegmentResult(status="completed")
        await queue.upload_segment_output(segment.id, result_data, last_frame_data, segment_result)

        total = time.monotonic() - segment_start
        logger.info("Faceswap reprocess segment %d complete in %.1fs", segment.index, total)

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
        if hasattr(e, "response") and hasattr(e.response, "text"):
            error_msg += f"\nResponse: {e.response.text[:500]}"
        logger.exception("Faceswap reprocess segment %s failed", segment.id)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000], progress_log=progress.text),
        )


def _holo_process_frames_sync(
    frame_files: list[str], packed_dir: str, key_color: str,
    subject_height_m: float, rvm_model_path: str, fps: float,
    flavor: str = "2d_matte", depth_model_path: str = "", depth_model_url: str = "",
    depth_scale_m: float = 0.12,
) -> tuple[str, int, int, bytes, dict]:
    """CPU-heavy hologram work: load -> matte (chroma/RVM) -> [depth] -> crop -> pack -> poster + manifest.

    Runs in a worker thread (asyncio.to_thread) so the daemon event loop stays free for the GPU
    loop + heartbeat while a hologram builds. Writes packed frames to packed_dir; returns
    (matte_mode, packed_w, packed_h, poster_png_bytes, manifest_dict). For "2.5d_depth" each frame
    additionally carries a depth region (bright = near) for the player's displaced mesh.
    """
    arrs = [np.asarray(Image.open(ff).convert("RGB")) for ff in frame_files]
    if detect_greenscreen(arrs[0], key_color):
        mode = "chroma"
        rgbs: list[np.ndarray] = []
        alphas: list[np.ndarray] = []
        for arr in arrs:
            rgb, alpha = chroma_key_despill(arr, key_color)
            rgbs.append(rgb)
            alphas.append(alpha)
    else:
        mode = "rvm"
        rgbs, alphas = rvm_matte(arrs, ensure_rvm_model(rvm_model_path))

    x, y, cw, ch = union_alpha_bbox(alphas)

    depth_u8: list[np.ndarray] = []
    if flavor == "2.5d_depth":
        mode += "+depth"
        dpath = ensure_depth_model(depth_model_path, depth_model_url)
        # Depth on the full frame with the backdrop neutralized via the matte (a flat chroma
        # plate flattens the model's subject disparity); normalize_depth_clip crops to the bbox.
        raw_depths = [estimate_depth(arr, dpath, alpha=a) for arr, a in zip(arrs, alphas)]
        depth_u8 = normalize_depth_clip(raw_depths, alphas, (x, y, cw, ch))

    packed_w = packed_h = 0
    for i, (rgb, alpha) in enumerate(zip(rgbs, alphas)):
        crgb = edge_extend_color(rgb[y:y + ch, x:x + cw], alpha[y:y + ch, x:x + cw])
        acrop = alpha[y:y + ch, x:x + cw]
        if flavor == "2.5d_depth":
            packed = pack_sbs_depth(crgb, acrop, depth_u8[i])
        else:
            packed = pack_sbs(crgb, acrop)
        if packed_w == 0:
            packed_h, packed_w = packed.shape[0], packed.shape[1]
        Image.fromarray(packed).save(os.path.join(packed_dir, f"{i:05d}.png"))

    pal = (np.clip(alphas[0][y:y + ch, x:x + cw], 0.0, 1.0) * 255).astype(np.uint8)
    poster_rgba = np.dstack([rgbs[0][y:y + ch, x:x + cw], pal])
    poster_buf = io.BytesIO()
    Image.fromarray(poster_rgba, "RGBA").save(poster_buf, format="PNG")

    manifest = build_manifest(
        packed_w, packed_h, cw, GUARD_PX, fps, (x, y, cw, ch), subject_height_m,
        flavor=flavor, depth_scale_m=depth_scale_m,
    )
    return mode, packed_w, packed_h, poster_buf.getvalue(), manifest


async def _execute_ar_hologram(
    segment: SegmentClaim,
    comfyui: ComfyUIClient,
    queue: QueueClient,
) -> None:
    """Matte a finalized clip into a packed color+alpha 'hologram' + manifest + poster.

    Tier-0 (flat/mono), chroma-key only in v1. Pure ffmpeg/numpy/cv2 — no ComfyUI. The
    source is the job's finalized stitched video (hologram_source_path), NOT this carrier
    segment's own output.
    """
    key_color = segment.hologram_key_color or "0x00b140"
    subject_height_m = segment.hologram_subject_height_m or 1.70
    flavor = segment.hologram_flavor or "2d_matte"
    depth_scale_m = segment.hologram_depth_scale_m or settings.depth_scale_m
    logger.info(
        "=== AR hologram segment %d (job %s) === key=%s height=%.2fm flavor=%s depth_scale=%.2fm",
        segment.index, str(segment.job_id)[:8], key_color, subject_height_m, flavor, depth_scale_m,
    )
    progress = ProgressLog(segment.id, queue)
    t0 = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix="holo_")
    try:
        if not segment.hologram_source_path:
            raise RuntimeError("No hologram_source_path — job has no finalized video to source from")
        fps = float(segment.fps) or 30.0

        await progress.log("[1/6] Downloading source video...")
        video_data = await _download_with_retry(
            lambda: queue.download_file(segment.hologram_source_path), "hologram_source"
        )
        src = os.path.join(tmpdir, "src.mp4")
        with open(src, "wb") as f:
            f.write(video_data)

        await progress.log("[2/6] Extracting frames...")
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir)
        await _run_ffmpeg(["-i", src, "-vsync", "0", os.path.join(frames_dir, "%05d.png")])
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not frame_files:
            raise RuntimeError("No frames extracted from source video")

        # Heavy CPU work (load -> matte -> crop -> pack -> poster + manifest) runs in a thread so
        # this hologram track runs concurrently with GPU generation without stalling the event loop.
        await progress.log(f"[3/6] Matting + cropping + packing {len(frame_files)} frames...")
        packed_dir = os.path.join(tmpdir, "packed")
        os.makedirs(packed_dir)
        mode, packed_w, packed_h, poster_bytes, manifest = await asyncio.to_thread(
            _holo_process_frames_sync, frame_files, packed_dir, key_color,
            subject_height_m, settings.rvm_model_path, fps,
            flavor, settings.depth_model_path, settings.depth_model_url, depth_scale_m,
        )
        await progress.log(f"[4/6] Matted ({mode}) + packed {len(frame_files)} frames")

        await progress.log("[5/6] Encoding packed mp4...")
        packed_mp4 = os.path.join(tmpdir, "hologram.mp4")
        # Explicit limited-range + bt709 tags: untagged output makes some decoders (Quest
        # included) guess the range, which crushes/clips the depth + alpha luma regions.
        await _run_ffmpeg([
            "-framerate", f"{fps}", "-i", os.path.join(packed_dir, "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-color_range", "tv", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart", packed_mp4,
        ])
        with open(packed_mp4, "rb") as f:
            packed_bytes = f.read()
        manifest_bytes = json.dumps(manifest).encode("utf-8")

        await progress.log("[6/6] Uploading hologram...")
        await queue.upload_hologram_output(segment.id, packed_bytes, manifest_bytes, poster_bytes)
        logger.info("AR hologram segment %d complete in %.1fs", segment.index, time.monotonic() - t0)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("AR hologram segment %s failed", segment.id)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=error_msg[:2000], progress_log=progress.text),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _generate_black(path: str, duration: float, width: int, height: int, fps: int) -> None:
    """Generate a black mp4 clip for dip-to-black smashcut transitions."""
    await _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])


async def _execute_smashcut_concat(segment: SegmentClaim, queue: QueueClient) -> None:
    """Concatenate hand-picked clips (same resolution) into one montage. Pure ffmpeg, no ComfyUI.

    seamless = stream-copy concat (fast, lossless); black = insert black dips + re-encode concat.
    """
    paths = segment.smashcut_clip_paths or []
    black = segment.smashcut_transition == "black"
    logger.info("=== Smashcut concat %s: %d clips, transition=%s ===",
                str(segment.id)[:8], len(paths), segment.smashcut_transition)
    progress = ProgressLog(segment.id, queue)
    start = time.monotonic()
    try:
        if len(paths) < 2:
            raise RuntimeError("smashcut needs at least 2 clips")

        with tempfile.TemporaryDirectory() as tmp:
            await progress.log(f"[1/3] Downloading {len(paths)} clips...")
            local = []
            for i, p in enumerate(paths):
                data = await _download_with_retry(lambda p=p: queue.download_file(p), "smashcut_clip")
                fn = os.path.join(tmp, f"clip_{i:03d}.mp4")
                with open(fn, "wb") as f:
                    f.write(data)
                local.append(fn)

            await progress.log("[2/3] Concatenating...")
            items = []
            for i, fn in enumerate(local):
                items.append(fn)
                if black and i < len(local) - 1:
                    bp = os.path.join(tmp, f"black_{i:03d}.mp4")
                    await _generate_black(bp, 0.4, segment.width, segment.height, segment.fps)
                    items.append(bp)

            # concat demuxer resolves relative paths against the list file's dir → basenames in tmp.
            list_path = os.path.join(tmp, "concat.txt")
            with open(list_path, "w") as f:
                for fn in items:
                    f.write(f"file '{os.path.basename(fn)}'\n")

            out = os.path.join(tmp, "smashcut.mp4")
            if black:
                await _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_path,
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out])
            else:
                await _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out])

            with open(out, "rb") as f:
                result = f.read()

            await progress.log("[3/3] Uploading...")
            await queue.upload_smashcut_output(segment.id, result)
        logger.info("Smashcut %s complete in %.1fs", str(segment.id)[:8], time.monotonic() - start)
    except Exception as e:
        logger.exception("Smashcut concat failed: %s", e)
        await queue.update_segment(
            segment.id,
            SegmentResult(status="failed", error_message=str(e)[:2000], progress_log=progress.text),
        )


async def execute_segment(
    segment: SegmentClaim,
    comfyui: ComfyUIClient,
    queue: QueueClient,
) -> None:
    """Execute a single segment end-to-end."""
    if segment.reprocess_type == "faceswap":
        return await _execute_faceswap_reprocess(segment, comfyui, queue)

    if segment.reprocess_type == "ar_hologram":
        return await _execute_ar_hologram(segment, comfyui, queue)

    if segment.reprocess_type == "smashcut_concat":
        return await _execute_smashcut_concat(segment, queue)

    # VACE video-conditioned continuation (resolved API-side). Capability-gated: a worker
    # without the Fun-VACE models/nodes silently falls through to the traditional i2v path.
    if segment.continuation_mode == "vace":
        if await _vace_capable(comfyui):
            return await _execute_vace_continuation(segment, comfyui, queue)
        logger.info(
            "Segment %d requested VACE but worker is not VACE-capable — falling back to traditional i2v",
            segment.index,
        )

    lora_names = ", ".join(l.high_file or l.low_file or "?" for l in (segment.loras or []))

    augmented_prompt = segment.prompt
    if segment.previous_motion_keywords:
        original_prompt = segment.prompt
        augmented_prompt = _augment_prompt_with_motion(
            original_prompt, segment.previous_motion_keywords
        )
        logger.info("Augmented prompt with motion continuity: %s -> %s", original_prompt[:50], augmented_prompt[:50])
        segment = segment.model_copy(update={"prompt": augmented_prompt})

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
        reference_frames_filenames = []
        if segment.reference_frames:
            for ref_s3_path in segment.reference_frames:
                ref_data = await _download_with_retry(
                    lambda: queue.download_file(ref_s3_path), f"reference_frame"
                )
                _validate_image_data(ref_data, "reference_frame")
                ext = os.path.splitext(ref_s3_path)[1] or ".png"
                ref_filename = f"ref_{len(reference_frames_filenames)}_{segment.id}{ext}"
                comfyui_filename = await comfyui.upload_image(ref_data, ref_filename)
                reference_frames_filenames.append(comfyui_filename)
                logger.info("Uploaded reference frame to ComfyUI: %s", comfyui_filename)
        # Set ComfyUI's activation-memory estimate for this run (read by the model_base
        # memory_required patch). CFG>1 runs the uncond pass (~2x activations) and must offload
        # the idle expert -> 0.015; distilled CFG-1 runs can stay resident and fast -> 0.003.
        # This is what prevents the de-distilled / long-video OOM.
        _cfg_hi = segment.cfg_high if segment.cfg_high is not None else settings.cfg_high
        _cfg_lo = segment.cfg_low if segment.cfg_low is not None else settings.cfg_low
        _cfg_active = _cfg_hi > 1 or _cfg_lo > 1
        _estimate = 0.015 if _cfg_active else 0.003
        try:
            with open("/tmp/wanly_estimate", "w") as _f:
                _f.write(str(_estimate))
            logger.info("wanly_estimate=%s (cfg=%.1f/%.1f -> %s)", _estimate, _cfg_hi, _cfg_lo,
                        "offload" if _cfg_active else "resident")
        except Exception as _e:
            logger.warning("Could not write /tmp/wanly_estimate: %s", _e)

        # Pass previous segment's motion magnitude for motion matching
        workflow = build_workflow(
            segment,
            start_image_filename=start_image_filename,
            initial_reference_image_filename=initial_ref_filename,
            reference_frame_filenames=reference_frames_filenames if reference_frames_filenames else None,
            previous_motion_magnitude=segment.previous_motion_magnitude,
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
        await comfyui.monitor_execution(prompt_id, client_id, progress=progress)
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

        # Step 7: Extract last frame + measure motion + upload
        t0 = time.monotonic()
        await progress.log("[7/7] Extracting last frame and uploading...")
        last_frame_data = await _extract_last_frame(video_data)

        # Seed re-anchor: faceswap the last frame to the canonical identity before it seeds
        # the next segment (only fires when API resolved seed_faceswap=True). Never raises.
        if getattr(segment, "seed_faceswap", False):
            last_frame_data = await _reanchor_seed_frame(
                segment, last_frame_data, comfyui, queue, progress,
            )

        # Measure motion magnitude using optical flow
        motion_magnitude = measure_motion_magnitude(video_data)
        if motion_magnitude:
            await progress.log(f"[7/7] Motion magnitude: {motion_magnitude:.2f} px/frame")

        motion_keywords = await extract_motion_keywords(segment.prompt, video_data)
        if motion_keywords:
            await progress.log(f"[7/7] Motion detected: {', '.join(motion_keywords)}")
        segment_result = SegmentResult(
            status="completed",
            motion_keywords=motion_keywords if motion_keywords else None,
            motion_magnitude=motion_magnitude,
        )
        await queue.upload_segment_output(segment.id, video_data, last_frame_data, segment_result)
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
