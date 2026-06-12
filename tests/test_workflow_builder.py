"""Tests for daemon.workflow_builder: frame/fps math, sampler schedule
injection, shift injection, and the de-distilled lightx2v rewire."""

import pytest

from daemon.config import settings
from daemon.schemas import LoraItem
from daemon.workflow_builder import (
    _calculate_generation_params,
    _snap_to_4n_plus_1,
    build_faceswap_workflow,
    build_workflow,
)


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (75, 77),
        (80, 81),
        (81, 81),
        (3, 5),
        (5, 5),
        (78, 77),
    ],
)
def test_snap_to_4n_plus_1(frames, expected):
    assert _snap_to_4n_plus_1(frames) == expected


class TestCalculateGenerationParams:
    @pytest.fixture(autouse=True)
    def _gen_fps_16(self, monkeypatch):
        monkeypatch.setattr(settings, "generation_fps", 16)

    @pytest.mark.parametrize("duration", [1.0, 2.5, 3.3, 5.0, 8.0, 10.7])
    @pytest.mark.parametrize("speed", [0.25, 0.5, 1.0, 1.5, 2.0])
    def test_wan_frames_always_on_4n_plus_1_grid(self, duration, speed):
        gen = _calculate_generation_params(30, duration, speed)
        assert (gen["wan_frames"] - 1) % 4 == 0
        assert gen["wan_frames"] >= 5

    @pytest.mark.parametrize(
        ("target_fps", "expected_multiplier"),
        [
            (30, 2),
            (60, 4),
            (16, 1),
        ],
    )
    def test_rife_multiplier(self, target_fps, expected_multiplier):
        gen = _calculate_generation_params(target_fps, 5.0)
        assert gen["rife_multiplier"] == expected_multiplier

    def test_output_fps_fits_duration(self):
        gen = _calculate_generation_params(30, 5.0)
        expected_total = gen["wan_frames"] * gen["rife_multiplier"]
        assert gen["output_fps"] == round(expected_total / 5.0)


class TestScheduleInjection:
    def test_custom_schedule_split(self, make_segment):
        segment = make_segment(steps_total=8, high_noise_steps=4)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert workflow["86"]["inputs"]["steps"] == 8
        assert workflow["86"]["inputs"]["start_at_step"] == 0
        assert workflow["86"]["inputs"]["end_at_step"] == 4
        assert workflow["85"]["inputs"]["steps"] == 8
        assert workflow["85"]["inputs"]["start_at_step"] == 4
        assert workflow["85"]["inputs"]["end_at_step"] == 8

    def test_high_noise_steps_clamped_below_steps_total(self, make_segment):
        segment = make_segment(steps_total=4, high_noise_steps=10)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert workflow["86"]["inputs"]["end_at_step"] == 3
        assert workflow["85"]["inputs"]["start_at_step"] == 3

    def test_high_noise_steps_clamped_above_zero(self, make_segment):
        segment = make_segment(steps_total=4, high_noise_steps=0)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert workflow["86"]["inputs"]["end_at_step"] == 1
        assert workflow["85"]["inputs"]["start_at_step"] == 1


class TestShiftInjection:
    def test_shift_values_reach_model_sampling_nodes(self, make_segment):
        segment = make_segment(shift_high=7.0, shift_low=4.5)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert workflow["104"]["inputs"]["shift"] == 7.0
        assert workflow["103"]["inputs"]["shift"] == 4.5


class TestOverridePrecedence:
    def test_segment_overrides_win_over_settings(self, make_segment, monkeypatch):
        monkeypatch.setattr(settings, "steps_total", 6)
        monkeypatch.setattr(settings, "shift_high", 5.0)
        segment = make_segment(steps_total=8, high_noise_steps=4, shift_high=7.0)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert workflow["86"]["inputs"]["steps"] == 8
        assert workflow["104"]["inputs"]["shift"] == 7.0

    def test_none_falls_back_to_settings(self, make_segment, monkeypatch):
        monkeypatch.setattr(settings, "steps_total", 6)
        monkeypatch.setattr(settings, "high_noise_steps", 3)
        monkeypatch.setattr(settings, "shift_high", 6.5)
        monkeypatch.setattr(settings, "shift_low", 4.0)
        segment = make_segment()  # all sampler overrides default to None
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert workflow["86"]["inputs"]["steps"] == 6
        assert workflow["86"]["inputs"]["end_at_step"] == 3
        assert workflow["85"]["inputs"]["start_at_step"] == 3
        assert workflow["85"]["inputs"]["end_at_step"] == 6
        assert workflow["104"]["inputs"]["shift"] == 6.5
        assert workflow["103"]["inputs"]["shift"] == 4.0


class TestDeDistillRewire:
    def test_high_strength_zero_removes_node_101(self, make_segment):
        segment = make_segment(lightx2v_strength_high=0.0)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert "101" not in workflow
        # 104 is spliced onto 101's upstream model: the high-noise UNET (95)
        assert workflow["104"]["inputs"]["model"] == ["95", 0]
        # Low side untouched
        assert "102" in workflow
        assert workflow["103"]["inputs"]["model"] == ["102", 0]

    def test_high_strength_positive_keeps_node_101(self, make_segment):
        segment = make_segment(lightx2v_strength_high=2.0)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert "101" in workflow
        assert workflow["101"]["inputs"]["strength_model"] == 2.0
        assert workflow["104"]["inputs"]["model"] == ["101", 0]

    def test_low_strength_zero_removes_node_102(self, make_segment):
        segment = make_segment(lightx2v_strength_low=0.0)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert "102" not in workflow
        assert workflow["103"]["inputs"]["model"] == ["96", 0]
        # High side untouched
        assert "101" in workflow
        assert workflow["104"]["inputs"]["model"] == ["101", 0]

    def test_low_strength_positive_keeps_node_102(self, make_segment):
        segment = make_segment(lightx2v_strength_low=1.0)
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert "102" in workflow
        assert workflow["102"]["inputs"]["strength_model"] == 1.0
        assert workflow["103"]["inputs"]["model"] == ["102", 0]

    def test_rewire_composes_with_user_lora(self, make_segment):
        segment = make_segment(
            lightx2v_strength_high=0.0,
            loras=[LoraItem(high_file="user_high.safetensors", high_weight=0.8)],
        )
        workflow = build_workflow(segment, start_image_filename="start.png")

        assert "101" not in workflow
        # 104 must point at the user-LoRA node (118), not the bare UNET (95)
        assert workflow["104"]["inputs"]["model"] == ["118", 0]
        assert workflow["118"]["inputs"]["model"] == ["95", 0]


class TestFaceswapWorkflowFps:
    def test_force_rate_follows_generation_fps(self, make_segment, monkeypatch):
        monkeypatch.setattr(settings, "generation_fps", 16)
        segment = make_segment(faceswap_enabled=True, faceswap_image="face.png")
        workflow = build_faceswap_workflow(segment, "existing.mp4")

        assert workflow["400"]["inputs"]["force_rate"] == 16.0
        assert "16fps" in workflow["400"]["_meta"]["title"]
        # target fps 30 at gen fps 16 → RIFE 2x (round, not floor)
        assert workflow["200"]["inputs"]["multiplier"] == 2
