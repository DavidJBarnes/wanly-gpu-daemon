"""The hologram encode carries the source's audio (console#475).

The packed pipeline decoded the source to silent PNGs and re-encoded from an image
sequence, so no hologram ever had an audio stream — by construction, invisibly. Now the
encode maps the source's audio alongside the packed frames, and the manifest records
whether a stream was actually there.
"""

import os
import shutil
import subprocess

import pytest

from daemon.executor import _packed_encode_args, _probe_has_audio
from daemon.hologram import build_manifest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _ffmpeg_encoders() -> str:
    if not FFMPEG:
        return ""
    try:
        return subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
    except Exception:
        return ""


# The encode argv itself demands libx264 (the daemon image and CI runners have it; this
# workstation may not), so tests that RUN ffmpeg skip on absence. The argv-shape tests
# above need nothing and always run.
LIBX264 = "libx264" in _ffmpeg_encoders()
NEEDS_REAL_FFMPEG = pytest.mark.skipif(
    not FFMPEG or not FFPROBE or not LIBX264,
    reason="no ffmpeg/ffprobe with libx264 installed",
)

_manifest_kwargs = dict(
    packed_w=1920, packed_h=640, color_w=640, guard_px=8, fps=24.0,
    crop_rect=(0, 0, 640, 640), subject_height_m=1.70,
)


class TestPackedEncodeArgs:
    def test_frames_and_source_are_separate_inputs(self):
        """Packed frames are input 0 — the matted-and-packed video; the source rides in as
        input 1 purely for its audio. One input would leave nothing to map."""
        args = _packed_encode_args(24.0, "/tmp/packed", "/tmp/src.mp4", "/tmp/out.mp4")
        inputs = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
        assert inputs == ["/tmp/packed/%05d.png", "/tmp/src.mp4"]

    def test_audio_is_mapped_with_an_optional_stream(self):
        """`?` is the backward-compat: an old final.mp4 has no audio stream, and without
        `?` the optional map would fail the whole encode because of the source's origins,
        not the hologram's settings."""
        args = _packed_encode_args(24.0, "/tmp/packed", "/tmp/src.mp4", "/tmp/out.mp4")
        assert args[args.index("-map") + 1] == "0:v"
        assert args[args.index("-map") + 3] == "1:a?"

    def test_audio_copies_instead_of_reencoding(self):
        """The source audio is already mp4-compatible AAC — measured 48 kHz stereo on a
        real 2026-09-08 render. Re-encoding it would spend CPU for nothing and drift."""
        args = _packed_encode_args(24.0, "/tmp/packed", "/tmp/src.mp4", "/tmp/out.mp4")
        assert args[args.index("-c:a") + 1] == "copy"

    def test_no_an(self):
        """The regression this suite exists for: `-an` here is what made every hologram
        silent, exactly like the image-sequence encode it replaced."""
        args = _packed_encode_args(24.0, "/tmp/packed", "/tmp/src.mp4", "/tmp/out.mp4")
        assert "-an" not in args

    def test_shortest_ties_the_tail_to_the_packed_frames(self):
        args = _packed_encode_args(24.0, "/tmp/packed", "/tmp/src.mp4", "/tmp/out.mp4")
        assert "-shortest" in args

    def test_output_path_is_last(self):
        args = _packed_encode_args(24.0, "/tmp/packed", "/tmp/src.mp4", "/tmp/out.mp4")
        assert args[-1] == "/tmp/out.mp4"


@NEEDS_REAL_FFMPEG
class TestRealEncode:
    @pytest.fixture
    def audible_source(self, tmp_path):
        """A 1-second h264+AAC clip — the shape a real stitched final has."""
        src = str(tmp_path / "src.mp4")
        os.system(
            "ffmpeg -y -v error "
            "-f lavfi -i 'color=c=green:s=64x64:d=1' "
            "-f lavfi -i 'sine=frequency=440:duration=1' "
            "-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest " + src
        )
        return src

    @pytest.fixture
    def silent_source(self, tmp_path):
        """An old-style silent clip (pre-audio VAe finals looked like this)."""
        src = str(tmp_path / "silent.mp4")
        os.system(
            "ffmpeg -y -v error "
            "-f lavfi -i 'color=c=green:s=64x64:d=1' "
            "-c:v libx264 -pix_fmt yuv420p " + src
        )
        return src

    @pytest.fixture
    def packed_frames(self, tmp_path):
        (tmp_path / "packed").mkdir()
        import numpy as np
        from PIL import Image
        Image.fromarray(np.full((640, 640, 3), 128, dtype=np.uint8)).save(
            tmp_path / "packed" / "00000.png")
        return str(tmp_path / "packed")

    def test_an_audible_source_produces_an_audio_stream(self, audible_source, packed_frames, tmp_path):
        out = str(tmp_path / "holo.mp4")
        import subprocess
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", *_packed_encode_args(24.0, packed_frames, audible_source, out)])
        assert proc.returncode == 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=codec_name", "-of", "csv=p=0", out],
            capture_output=True, text=True)
        assert probe.stdout.strip() == "aac"

    def test_a_silent_source_still_encodes(self, silent_source, packed_frames, tmp_path):
        """The `?`: pre-audio sources must keep encoding, with no audio stream in the
        result — and the manifest, driven by the probe, says so."""
        out = str(tmp_path / "holo.mp4")
        import subprocess
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", *_packed_encode_args(24.0, packed_frames, silent_source, out)])
        assert proc.returncode == 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", out],
            capture_output=True, text=True)
        assert probe.stdout.strip() == ""


class TestProbeHasAudio:
    @pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="no ffmpeg/ffprobe installed")
    @pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="no ffmpeg/ffprobe installed")
    async def test_an_audible_file_yes_a_silent_file_no(self, tmp_path):
        audible = str(tmp_path / "a.mp4")
        assert await _probe_has_audio(audible) is False  # also: missing file is False, not an error
        os.system(
            "ffmpeg -y -v error -f lavfi -i 'sine=frequency=440:duration=1' -c:a aac " + audible)
        assert await _probe_has_audio(audible) is True
        silent = str(tmp_path / "s.mp4")
        # mpeg4, not libx264: a stock distro ffmpeg lacks x264, and the probe does not care.
        os.system(
            "ffmpeg -y -v error -f lavfi -i 'color=c=black:s=16x16:d=1' -c:v mpeg4 " + silent)
        assert await _probe_has_audio(silent) is False


class TestManifestHasAudio:
    def test_defaults_false(self):
        """Callers that predate the field — old call sites, tests — build the same manifest
        they always did, with the field present and honest about its silence."""
        m = build_manifest(**_manifest_kwargs)
        assert m["has_audio"] is False

    def test_true_when_the_source_had_audio(self):
        m = build_manifest(**_manifest_kwargs, has_audio=True)
        assert m["has_audio"] is True

    def test_coexists_with_the_25d_fields(self):
        m = build_manifest(**_manifest_kwargs, flavor="2.5d_depth", depth_scale_m=0.30, has_audio=True)
        assert m["has_audio"] is True
        assert m["region_depth_uv"]["w"] > 0
        assert m["depth_scale_m"] == 0.30
