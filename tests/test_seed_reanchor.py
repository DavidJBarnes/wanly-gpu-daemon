"""Tests for the continuation seed re-anchor.

This path shipped non-functional and stayed that way through 26 multi-segment jobs, because
the API gated it on "a later segment already exists at claim time" — which a job created
with segment 0 and continued afterwards can never satisfy. It had zero test coverage, which
is why nobody noticed. These tests pin the behaviour that matters:

  - the face comes from the SEGMENT (not a hardcoded per-character map)
  - a missing face degrades to the raw frame instead of raising
  - FaceFusion's "no face found" no-op is detected and falls back
  - nothing in here may ever raise: a failed re-anchor must not fail the segment
"""

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from daemon.executor import _images_differ, _reanchor_seed_frame
from daemon.workflow_builder import build_seed_faceswap_workflow
from tests.conftest import make_segment


def png_bytes(seed: int, size: tuple[int, int] = (128, 128)) -> bytes:
    """Deterministic noise, not a flat colour.

    _validate_image_data rejects anything under 1 KB, and a solid-colour PNG of this size
    compresses to ~0.2 KB — so flat fixtures get rejected as corrupt before the code under
    test is reached. Noise keeps the encoded size realistic.
    """
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


RAW_SEED = png_bytes(1)
SWAPPED = png_bytes(2)


class FakeProgress:
    def __init__(self):
        self.lines: list[str] = []

    async def log(self, msg: str) -> None:
        self.lines.append(msg)


class FakeComfy:
    """Records uploads and returns a canned swapped image."""

    def __init__(self, output: bytes | None = SWAPPED, raise_on_submit: bool = False):
        self.uploads: list[str] = []
        self.submitted: dict | None = None
        self.output = output
        self.raise_on_submit = raise_on_submit

    async def upload_image(self, data: bytes, filename: str) -> str:
        self.uploads.append(filename)
        return filename

    async def submit_workflow(self, workflow: dict):
        if self.raise_on_submit:
            raise RuntimeError("comfy exploded")
        self.submitted = workflow
        return "prompt-1", "client-1"

    async def monitor_execution(self, prompt_id, client_id, progress=None):
        return None

    async def get_history(self, prompt_id):
        return {"outputs": {"186": {"images": [{"filename": "seed_swap_0001.png"}]}}}

    async def download_output(self, filename, subfolder="", type_="output"):
        return self.output


class FakeQueue:
    def __init__(self, file_bytes: bytes = png_bytes(3)):
        self.downloaded: list[str] = []
        self.file_bytes = file_bytes

    async def download_file(self, uri: str) -> bytes:
        self.downloaded.append(uri)
        return self.file_bytes


class TestFaceSource:
    """The whole point of the rework: the face is per-segment, not per-character."""

    @pytest.mark.asyncio
    async def test_uses_the_segments_own_faceswap_face(self):
        seg = make_segment(
            index=0, seed_faceswap=True,
            faceswap_enabled=True, faceswap_image="faceswap_abc.png",
        )
        comfy, queue, prog = FakeComfy(), FakeQueue(), FakeProgress()
        out = await _reanchor_seed_frame(seg, RAW_SEED, comfy, queue, prog)

        assert out == SWAPPED
        # node 188 is the swap SOURCE image; it must be the segment's face
        assert comfy.submitted["188"]["inputs"]["image"] == "faceswap_abc.png"
        assert any("Re-anchored" in line for line in prog.lines)

    @pytest.mark.asyncio
    async def test_any_character_works_not_just_the_two_hardcoded_ones(self):
        """Previously the face came from a lora_id -> S3 map with exactly two entries, so an
        unknown LoRA silently did nothing. Identity now travels with the segment."""
        seg = make_segment(
            seed_faceswap=True, loras=None,
            faceswap_enabled=True, faceswap_image="someone_else.png",
        )
        comfy = FakeComfy()
        out = await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), FakeProgress())
        assert out == SWAPPED
        assert comfy.submitted["188"]["inputs"]["image"] == "someone_else.png"

    @pytest.mark.asyncio
    async def test_unresolved_s3_uri_is_downloaded_and_uploaded(self):
        """Covers a face selected while video faceswap is off, so _resolve_faceswap_image
        never converted it to a ComfyUI filename."""
        seg = make_segment(
            seed_faceswap=True, faceswap_enabled=False,
            faceswap_image="s3://wanly-loras/faces/kelly.png",
        )
        comfy, queue = FakeComfy(), FakeQueue()
        out = await _reanchor_seed_frame(seg, RAW_SEED, comfy, queue, FakeProgress())

        assert out == SWAPPED
        assert queue.downloaded == ["s3://wanly-loras/faces/kelly.png"]
        assert any(f.startswith("seedface_") for f in comfy.uploads)


class TestAnchorsToTheScoredReference:
    """Two bugs found on 2026-08-04 after a real re-anchor moved identity by +0.014.

    The seed came out at 0.677 against ground truth where the raw frame was 0.663 — a full
    face swap that accomplished nothing. Cause was a mismatch on BOTH ends: it swapped the
    wrong face IN, onto the wrong face in the frame."""

    @pytest.mark.asyncio
    async def test_prefers_the_job_ground_truth_over_a_separate_portrait(self):
        """Identity is scored against the job's starting image. Anchoring to a different
        portrait pulls the seed toward a face nobody is measuring."""
        seg = make_segment(
            seed_faceswap=True, faceswap_enabled=True,
            faceswap_image="Kelly_young_driveway.jpg",
            identity_ground_truth="seg0_start.png",
        )
        comfy = FakeComfy()
        out = await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), FakeProgress())
        assert out == SWAPPED
        assert comfy.submitted["188"]["inputs"]["image"] == "seg0_start.png"

    @pytest.mark.asyncio
    async def test_falls_back_to_faceswap_face_without_ground_truth(self):
        seg = make_segment(
            seed_faceswap=True, faceswap_enabled=True,
            faceswap_image="portrait.png", identity_ground_truth=None,
        )
        comfy = FakeComfy()
        await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), FakeProgress())
        assert comfy.submitted["188"]["inputs"]["image"] == "portrait.png"

    @pytest.mark.asyncio
    async def test_forces_facefusion_even_when_the_video_uses_reactor(self):
        """ReActor picks the target face by POSITION (faces_index 0, left-right). On a
        two-person frame that is the leftmost face — frequently the wrong person. It still
        rewrites pixels, so the caller's diff reads success while her face is untouched.
        FaceFusion selects by reference, matching the source face, which is the only correct
        semantics here."""
        seg = make_segment(
            seed_faceswap=True, faceswap_enabled=True, faceswap_image="f.png",
            faceswap_method="reactor", faceswap_faces_index="0",
            faceswap_faces_order="left-right",
        )
        comfy = FakeComfy()
        await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), FakeProgress())
        node = comfy.submitted["183"]
        assert node["class_type"] == "AdvancedSwapFaceImage", node["class_type"]
        assert node["inputs"]["face_selector_mode"] == "reference"

    @pytest.mark.asyncio
    async def test_seed_uses_a_looser_reference_threshold_than_video(self):
        """The seed frame is the DRIFTED one — that is why it is being re-anchored. A face
        at 0.663 cosine can fall outside the video path's 0.8 threshold, in which case
        FaceFusion swaps nothing and the fix no-ops precisely when it matters."""
        seg = make_segment(seed_faceswap=True, faceswap_enabled=True, faceswap_image="f.png")
        comfy = FakeComfy()
        await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), FakeProgress())
        assert comfy.submitted["183"]["inputs"]["reference_face_distance"] == 1.0

    def test_video_faceswap_keeps_the_validated_08_threshold(self):
        """Scoped change: the seed path loosens, the video path David has already validated
        must not move."""
        from daemon.workflow_builder import _add_faceswap
        wf: dict = {}
        seg = make_segment(faceswap_enabled=True, faceswap_image="f.png",
                           faceswap_method="facefusion")
        _add_faceswap(wf, seg)
        assert wf["183"]["inputs"]["reference_face_distance"] == 0.8


class TestDegradesToRawSeed:
    """A failed re-anchor must cost nothing: the raw frame still seeds the next segment."""

    @pytest.mark.asyncio
    async def test_no_face_configured_keeps_raw_frame(self):
        seg = make_segment(seed_faceswap=True, faceswap_enabled=False, faceswap_image=None)
        comfy, prog = FakeComfy(), FakeProgress()
        out = await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), prog)

        assert out == RAW_SEED
        assert comfy.submitted is None, "must not submit a workflow with no source face"
        assert any("No faceswap face" in line for line in prog.lines)

    @pytest.mark.asyncio
    async def test_facefusion_noop_is_detected_and_falls_back(self):
        """FaceFusion re-saves identical pixels when it finds no face in the frame. That is
        indistinguishable from success unless the result is diffed."""
        seg = make_segment(seed_faceswap=True, faceswap_enabled=True, faceswap_image="f.png")
        comfy = FakeComfy(output=RAW_SEED)   # same pixels back
        prog = FakeProgress()
        out = await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), prog)

        assert out == RAW_SEED
        assert any("No face detected" in line for line in prog.lines)

    @pytest.mark.asyncio
    async def test_never_raises_when_comfyui_fails(self):
        """A broken re-anchor must not fail an otherwise-good 30-minute generation."""
        seg = make_segment(seed_faceswap=True, faceswap_enabled=True, faceswap_image="f.png")
        comfy = FakeComfy(raise_on_submit=True)
        out = await _reanchor_seed_frame(seg, RAW_SEED, comfy, FakeQueue(), FakeProgress())
        assert out == RAW_SEED


class TestNoopDetection:
    def test_identical_images_do_not_differ(self):
        assert _images_differ(RAW_SEED, RAW_SEED) is False

    def test_different_images_differ(self):
        assert _images_differ(RAW_SEED, SWAPPED) is True

    def test_mismatched_shapes_count_as_different(self):
        assert _images_differ(RAW_SEED, png_bytes(1, size=(64, 64))) is True


class TestWorkflowShape:
    def test_seed_workflow_loads_the_frame_and_saves_one_image(self):
        seg = make_segment(faceswap_enabled=True, faceswap_image="face.png",
                           faceswap_method="facefusion")
        wf = build_seed_faceswap_workflow(seg, "seed_frame.png")
        assert wf["400"]["class_type"] == "LoadImage"
        assert wf["400"]["inputs"]["image"] == "seed_frame.png"
        assert wf["186"]["class_type"] == "SaveImage"
        # single still, not a video: no VHS combine anywhere in the graph
        assert not any(n.get("class_type", "").startswith("VHS_") for n in wf.values())


class TestFaceSelection:
    """With two people in frame the seed swap must be told WHICH face to replace.

    _add_faceswap defaults to faces_order="left-right", faces_index="0" — the leftmost
    face. In a two-person scene that can be the wrong person entirely, so these values
    have to survive from the console onto the seed workflow.
    """

    def test_faces_order_and_index_reach_the_seed_workflow(self):
        seg = make_segment(
            faceswap_enabled=False, seed_faceswap=True, faceswap_image="face.png",
            faceswap_method="reactor",
            faceswap_faces_order="right-left", faceswap_faces_index="1",
        )
        wf = build_seed_faceswap_workflow(seg, "seed.png")
        opts = next(n for n in wf.values() if n["class_type"] == "ReActorOptions")
        assert opts["inputs"]["input_faces_order"] == "right-left"
        assert opts["inputs"]["input_faces_index"] == "1"

    def test_defaults_pick_the_leftmost_face_when_unset(self):
        """Documents the fallback: fine for one face, WRONG for two."""
        seg = make_segment(
            faceswap_enabled=False, seed_faceswap=True, faceswap_image="face.png",
            faceswap_method="reactor",
            faceswap_faces_order=None, faceswap_faces_index=None,
        )
        wf = build_seed_faceswap_workflow(seg, "seed.png")
        opts = next(n for n in wf.values() if n["class_type"] == "ReActorOptions")
        assert opts["inputs"]["input_faces_order"] == "left-right"
        assert opts["inputs"]["input_faces_index"] == "0"

    def test_method_reaches_the_seed_workflow(self):
        seg = make_segment(
            faceswap_enabled=False, seed_faceswap=True, faceswap_image="face.png",
            faceswap_method="reactor",
        )
        wf = build_seed_faceswap_workflow(seg, "seed.png")
        assert any(n["class_type"] == "ReActorOptions" for n in wf.values())


class TestFlagPlumbing:
    def test_seed_faceswap_defaults_off(self):
        assert make_segment().seed_faceswap is False

    def test_flag_round_trips_from_the_claim(self):
        assert make_segment(seed_faceswap=True).seed_faceswap is True

    def test_flag_is_independent_of_video_faceswap(self):
        """The seed re-anchor is a separate decision from swapping the whole clip."""
        seg = make_segment(seed_faceswap=True, faceswap_enabled=False)
        assert seg.seed_faceswap is True and seg.faceswap_enabled is False


class TestSwapperTunables:
    """face_swapper_model and pixel_boost were hardcoded, which made the swap stage
    untunable - and a controlled sweep put the remaining headroom there, not in generation
    (no LoRA at all scores mean 0.695, same as a purpose-trained one)."""

    def test_defaults_preserve_the_validated_recipe(self):
        from daemon.workflow_builder import _add_faceswap
        wf: dict = {}
        seg = make_segment(faceswap_enabled=True, faceswap_image="f.png",
                           faceswap_method="facefusion")
        _add_faceswap(wf, seg)
        assert wf["183"]["inputs"]["face_swapper_model"] == "inswapper_128"
        assert wf["183"]["inputs"]["pixel_boost"] == "512x512"

    def test_segment_can_override_both(self):
        from daemon.workflow_builder import _add_faceswap
        wf: dict = {}
        seg = make_segment(faceswap_enabled=True, faceswap_image="f.png",
                           faceswap_method="facefusion",
                           faceswap_model="hyperswap_1c_256",
                           faceswap_pixel_boost="256x256")
        _add_faceswap(wf, seg)
        assert wf["183"]["inputs"]["face_swapper_model"] == "hyperswap_1c_256"
        assert wf["183"]["inputs"]["pixel_boost"] == "256x256"


class TestCleanContinuationSeed:
    """The deliverable keeps the swap; the frame that seeds the NEXT segment should not.

    Measured on a real 2x5s chain: face detail ran 233 -> 130 across seg0, fell to 101 once
    that swapped frame became conditioning, then 101 -> 79 through seg1 — 66% of facial detail
    gone. A swap costs ~18% because inswapper rebuilds the face from a 512-d embedding, and
    seeding from a swapped frame compounds that every segment. Identity does not need to ride
    on the seed: the next segment's own faceswap restores it on every output frame."""

    def test_workflow_saves_the_pre_swap_frame_when_swapping(self):
        from daemon.workflow_builder import build_workflow
        seg = make_segment(faceswap_enabled=True, faceswap_image="f.png",
                           faceswap_method="facefusion", duration_seconds=3.0)
        wf = build_workflow(seg, start_image_filename="start.png")
        assert "205" in wf and wf["205"]["class_type"] == "ImageFromBatch"
        # node 87 is the raw VAEDecode batch - 183 would be the swapped one
        assert wf["205"]["inputs"]["image"] == ["87", 0]
        assert wf["206"]["inputs"]["images"] == ["205", 0]

    def test_seed_index_is_pre_rife(self):
        """Node 87 holds wan_frames, not the interpolated total; an out-of-range index would
        silently clamp to the wrong frame."""
        from daemon.workflow_builder import build_workflow, _calculate_generation_params
        seg = make_segment(faceswap_enabled=True, faceswap_image="f.png",
                           faceswap_method="facefusion", duration_seconds=3.0)
        wf = build_workflow(seg, start_image_filename="start.png")
        gen = _calculate_generation_params(seg.fps, seg.duration_seconds, seg.speed)
        assert wf["205"]["inputs"]["batch_index"] == gen["wan_frames"] - 1

    def test_no_clean_seed_nodes_without_faceswap(self):
        """Without a swap the video frames ARE the clean frames - the extra save would just
        cost time."""
        from daemon.workflow_builder import build_workflow
        seg = make_segment(faceswap_enabled=False, duration_seconds=3.0)
        wf = build_workflow(seg, start_image_filename="start.png")
        assert "205" not in wf and "206" not in wf

    @pytest.mark.asyncio
    async def test_clean_seed_returns_none_when_absent(self):
        """Falls back to extracting from the finished video rather than failing a completed
        30-minute generation."""
        from daemon.executor import _clean_seed_frame
        assert await _clean_seed_frame({"outputs": {}}, FakeComfy()) is None

    @pytest.mark.asyncio
    async def test_clean_seed_downloads_node_206(self):
        from daemon.executor import _clean_seed_frame
        comfy = FakeComfy(output=SWAPPED)
        hist = {"outputs": {"206": {"images": [{"filename": "clean_seed_0001.png"}]}}}
        assert await _clean_seed_frame(hist, comfy) == SWAPPED

    @pytest.mark.asyncio
    async def test_clean_seed_never_raises_on_download_failure(self):
        from daemon.executor import _clean_seed_frame

        class Boom(FakeComfy):
            async def download_output(self, *a, **k):
                raise RuntimeError("s3 exploded")

        hist = {"outputs": {"206": {"images": [{"filename": "x.png"}]}}}
        assert await _clean_seed_frame(hist, Boom()) is None
