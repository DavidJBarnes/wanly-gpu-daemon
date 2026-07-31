"""Tests for per-clip smashcut retiming.

The fps choice is the load-bearing decision here. Every clip must be encoded at one rate
before the concat demuxer will splice them, so a montage mixing 0.5x and 2x still has to
resolve to a single number — and the naive choice (scale by each clip's own speed) produces
streams that will not concatenate at all.
"""

import pytest

from daemon.executor import SMASHCUT_MAX_FPS, _retime_clip, smashcut_output_fps


class TestOutputFps:
    def test_no_retiming_keeps_the_source_rate(self):
        assert smashcut_output_fps(30, []) == 30
        assert smashcut_output_fps(30, [1.0, 1.0]) == 30

    def test_speeding_up_raises_the_rate_instead_of_dropping_frames(self):
        """Sources are RIFE-interpolated, so 2x on 30fps still fits every original frame."""
        assert smashcut_output_fps(30, [2.0]) == 60

    def test_slowing_down_never_drops_below_the_source_rate(self):
        """The regression this guards: base*0.5 would emit 15fps and make slow-motion
        choppier than the clip it was made from. Frames are held longer instead."""
        assert smashcut_output_fps(30, [0.5]) == 30
        assert smashcut_output_fps(30, [0.25, 0.5]) == 30

    def test_the_fastest_clip_sets_the_rate(self):
        """A mixed montage resolves to one rate; the slow clips just repeat frames."""
        assert smashcut_output_fps(30, [0.5, 2.0, 1.0]) == 60

    def test_rate_is_capped(self):
        """4x on 30fps would be 120fps — past what the cap allows for a montage."""
        assert smashcut_output_fps(30, [4.0]) == SMASHCUT_MAX_FPS
        assert smashcut_output_fps(60, [2.0]) == SMASHCUT_MAX_FPS

    def test_fractional_rates_are_rounded_to_an_integer(self):
        """-r takes an int here; 24 * 1.5 must not reach ffmpeg as 36.0."""
        result = smashcut_output_fps(24, [1.5])
        assert result == 36
        assert isinstance(result, int)


class TestRetimeCommand:
    @pytest.fixture
    def captured(self, monkeypatch):
        calls = []

        async def fake(args):
            calls.append(args)

        monkeypatch.setattr("daemon.executor._run_ffmpeg", fake)
        return calls

    async def test_setpts_divides_by_the_speed(self, captured):
        """setpts=PTS/2 halves the duration. Inverting this silently reverses the control."""
        await _retime_clip("in.mp4", "out.mp4", 2.0, 60)
        assert "setpts=PTS/2.0" in captured[0]

    async def test_slow_motion_stretches(self, captured):
        await _retime_clip("in.mp4", "out.mp4", 0.5, 30)
        assert "setpts=PTS/0.5" in captured[0]

    async def test_output_rate_is_forced(self, captured):
        """Without -r the clips keep their source rates and the concat demuxer rejects them."""
        await _retime_clip("in.mp4", "out.mp4", 2.0, 60)
        args = captured[0]
        assert args[args.index("-r") + 1] == "60"

    async def test_audio_is_dropped(self, captured):
        """Generated clips have no audio; a stray stream would survive setpts and drift."""
        await _retime_clip("in.mp4", "out.mp4", 2.0, 60)
        assert "-an" in captured[0]

    async def test_source_and_destination_are_not_swapped(self, captured):
        await _retime_clip("src.mp4", "dst.mp4", 1.5, 45)
        args = captured[0]
        assert args[args.index("-i") + 1] == "src.mp4"
        assert args[-1] == "dst.mp4"
