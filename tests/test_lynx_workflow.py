"""Tests for the Lynx identity-preserving graph builder."""

import json
from pathlib import Path

import pytest

from daemon.config import settings
from daemon.schemas import LoraItem
from daemon.workflow_builder import (
    LYNX_ARMS,
    LYNX_RESOLUTIONS,
    WANVIDEO_SCHEDULERS,
    LynxValidationError,
    _lynx_arm,
    _pick,
    _validate_lynx,
    build_lynx_workflow,
    lynx_num_frames,
)
from tests.conftest import make_segment

GOLDEN = Path(__file__).parent / "golden" / "lynx_832x480_81f.json"


class TestPick:
    """Per-job-override -> settings-default precedence."""

    def test_none_falls_back_to_default(self):
        assert _pick(None, 0.7) == 0.7

    @pytest.mark.parametrize("override", [0, 0.0, "", False, 1.5, "lcm"])
    def test_falsy_but_set_overrides_win(self, override):
        # 0 / "" / False are legitimate overrides, not "unset" — only None defers.
        assert _pick(override, "default") == override


class TestLynxArm:
    @pytest.mark.parametrize("name,expected", [
        ("Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors", "lite"),
        ("lynx_full_resampler_fp32.safetensors", "full"),
        ("LYNX_FULL_RESAMPLER.safetensors", "full"),
    ])
    def test_detects_arm(self, name, expected):
        assert _lynx_arm(name) == expected

    @pytest.mark.parametrize("name", [
        "some_other_adapter.safetensors",          # no marker
        "lynx_lite_full_mixed.safetensors",        # ambiguous: both markers
    ])
    def test_unclassifiable_returns_none(self, name):
        assert _lynx_arm(name) is None

    def test_arms_constant(self):
        assert LYNX_ARMS == ("lite", "full")


class TestLynxNumFrames:
    def test_lands_on_4n_plus_1(self):
        assert lynx_num_frames(make_segment(duration_seconds=5.4)) == 81

    @pytest.mark.parametrize("duration", [0.1, 0.25, 1.0, 2.0, 3.7, 8.0])
    def test_always_on_grid_and_minimum(self, duration):
        n = lynx_num_frames(make_segment(duration_seconds=duration))
        assert n >= 5
        assert (n - 1) % 4 == 0


class TestValidation:
    """Constraints raise with the offending value named — never silently clamp."""

    def test_missing_subject_rejected(self):
        with pytest.raises(LynxValidationError, match="subject_image"):
            _validate_lynx(make_segment(), "", 81, "lite_ip.safetensors", "lite_res.safetensors")

    @pytest.mark.parametrize("w,h", [(512, 512), (1920, 1080), (480, 832), (1024, 576)])
    def test_off_bucket_resolution_rejected(self, w, h):
        seg = make_segment(width=w, height=h)
        with pytest.raises(LynxValidationError, match=f"{w}x{h}"):
            _validate_lynx(seg, "s.png", 81, "lite_ip.safetensors", "lite_res.safetensors")

    @pytest.mark.parametrize("w,h", sorted(LYNX_RESOLUTIONS))
    def test_native_buckets_accepted(self, w, h):
        seg = make_segment(width=w, height=h)
        _validate_lynx(seg, "s.png", 81, "lite_ip.safetensors", "lite_res.safetensors")

    @pytest.mark.parametrize("frames", [0, 3, 4, 80, 82, 100])
    def test_off_grid_frame_count_rejected(self, frames):
        with pytest.raises(LynxValidationError, match="4n\\+1"):
            _validate_lynx(make_segment(), "s.png", frames, "lite_ip.safetensors", "lite_res.safetensors")

    @pytest.mark.parametrize("frames", [5, 9, 81, 121])
    def test_on_grid_frame_count_accepted(self, frames):
        _validate_lynx(make_segment(), "s.png", frames, "lite_ip.safetensors", "lite_res.safetensors")

    @pytest.mark.parametrize("ip,res", [
        ("lynx_lite_ip.safetensors", "lynx_full_resampler.safetensors"),   # mixed arms
        ("lynx_full_ip.safetensors", "lynx_lite_resampler.safetensors"),   # mixed arms
        ("unmarked_ip.safetensors", "lynx_full_resampler.safetensors"),    # unclassifiable
    ])
    def test_mismatched_adapter_pair_rejected(self, ip, res):
        with pytest.raises(LynxValidationError, match="same arm"):
            _validate_lynx(make_segment(), "s.png", 81, ip, res)

    def test_build_surfaces_validation(self):
        # The public builder must validate, not just the private helper.
        with pytest.raises(LynxValidationError):
            build_lynx_workflow(make_segment(width=512, height=512), "s.png", 81)


class TestGraphTopology:
    """The wiring that makes Lynx identity-preserving, asserted structurally."""

    def test_node_set(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        # 700 = the distill LoRA (on by default); no 670 because fps 15 needs no RIFE.
        assert set(wf) == {
            "600", "601", "602", "610", "620", "621", "622",
            "630", "631", "632", "633", "640", "641", "650", "660", "680", "700",
        }

    def test_adapters_chain_into_the_loader(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["601"]["inputs"]["extra_model"] == settings.lynx_ip_layers
        assert wf["602"]["inputs"]["extra_model"] == settings.lynx_ref_layers
        # ref selector chains onto the ip selector, and only the tail reaches the loader
        assert wf["602"]["inputs"]["prev_model"] == ["601", 0]
        assert wf["610"]["inputs"]["extra_model"] == ["602", 0]

    def test_subject_feeds_both_adapters(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["630"]["inputs"]["image"] == "subject.png"
        # crop output 0 (112px ArcFace crop) -> resampler; output 1 (256px) -> ref adapter
        assert wf["631"]["inputs"]["image"] == ["630", 0]
        assert wf["633"]["inputs"]["ip_image"] == ["631", 0]
        assert wf["641"]["inputs"]["ref_image"] == ["631", 1]
        assert wf["633"]["inputs"]["resampler"] == ["632", 0]

    def test_subject_is_not_a_start_frame(self, segment):
        """T2V conditioned on a subject — the image must never become frame 0."""
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["640"]["class_type"] == "WanVideoEmptyEmbeds"
        assert not any(n["class_type"] == "WanVideoImageToVideoEncode" for n in wf.values())
        # nothing feeds LoadImage into the latent/embeds path
        assert wf["640"]["inputs"] == {"width": 832, "height": 480, "num_frames": 81}

    def test_ref_path_supplies_its_own_text_embed_and_vae(self, segment):
        """WanVideoAddLynxEmbeds raises if ref_image comes without ref_text_embed + vae."""
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["641"]["inputs"]["ref_text_embed"] == ["622", 0]
        assert wf["641"]["inputs"]["vae"] == ["620", 0]
        assert wf["622"]["inputs"]["positive_prompt"] == settings.lynx_ref_prompt
        # the ref-pass embed is distinct from the main prompt embed
        assert wf["621"]["inputs"]["positive_prompt"] == segment.prompt

    def test_text_encoders_use_cached_offload_path(self, segment):
        """umt5-xxl residency has OOM'd this box; both encodes use the cached node."""
        wf = build_lynx_workflow(segment, "subject.png", 81)
        for nid in ("621", "622"):
            assert wf[nid]["class_type"] == "WanVideoTextEncodeCached"
            assert wf[nid]["inputs"]["model_name"] == settings.lynx_t5_model

    def test_block_swap_and_offload(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["610"]["inputs"]["load_device"] == "offload_device"
        assert wf["610"]["inputs"]["block_swap_args"] == ["600", 0]
        assert wf["600"]["inputs"]["blocks_to_swap"] == settings.lynx_blocks_to_swap

    def test_does_not_requantise_an_already_scaled_fp8_checkpoint(self, segment):
        """Regression: the base file is already _scaled_KJ fp8.

        Quantizing it again leaves the fp16 Lynx adapters meeting fp8 base weights and
        WanVideoSampler dies with "self and mat2 must have the same dtype, but got Half
        and Float8_e4m3fn". Kijai's reference workflow pairs fp16_fast with disabled.
        """
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["610"]["inputs"]["quantization"] == "disabled"
        assert wf["610"]["inputs"]["base_precision"] == "fp16_fast"

    def test_loader_precision_is_configurable(self, monkeypatch, segment):
        monkeypatch.setattr(settings, "lynx_base_precision", "fp16")
        monkeypatch.setattr(settings, "lynx_quantization", "fp8_e4m3fn_scaled")
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["610"]["inputs"]["base_precision"] == "fp16"
        assert wf["610"]["inputs"]["quantization"] == "fp8_e4m3fn_scaled"

    def test_sampler_consumes_lynx_embeds_not_raw_latents(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["650"]["inputs"]["image_embeds"] == ["641", 0]
        assert wf["641"]["inputs"]["embeds"] == ["640", 0]


class TestParameterPrecedence:
    def test_defaults_come_from_settings(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        embeds = wf["641"]["inputs"]
        assert embeds["ip_scale"] == settings.lynx_ip_scale
        assert embeds["ref_scale"] == settings.lynx_ref_scale
        assert embeds["lynx_cfg_scale"] == settings.lynx_lynx_cfg_scale
        assert wf["650"]["inputs"]["steps"] == settings.lynx_steps

    def test_per_job_overrides_win(self):
        seg = make_segment(
            lynx_ip_scale=1.5, lynx_ref_scale=1.0, lynx_cfg_scale=3.0,
            lynx_start_percent=0.1, lynx_end_percent=0.9,
            lynx_steps=12, lynx_cfg=3.5, lynx_shift=5.0,
        )
        wf = build_lynx_workflow(seg, "subject.png", 81)
        embeds, sampler = wf["641"]["inputs"], wf["650"]["inputs"]
        assert (embeds["ip_scale"], embeds["ref_scale"]) == (1.5, 1.0)
        assert embeds["lynx_cfg_scale"] == 3.0
        assert (embeds["start_percent"], embeds["end_percent"]) == (0.1, 0.9)
        assert (sampler["steps"], sampler["cfg"], sampler["shift"]) == (12, 3.5, 5.0)

    def test_zero_ip_scale_is_honoured_not_treated_as_unset(self):
        """0.0 disables the ID adapter — it must not fall back to the 0.7 default."""
        wf = build_lynx_workflow(make_segment(lynx_ip_scale=0.0), "subject.png", 81)
        assert wf["641"]["inputs"]["ip_scale"] == 0.0

    def test_ab_arm_override_swaps_both_adapter_and_resampler(self):
        seg = make_segment(
            lynx_ip_layers="Wan2_1-T2V-14B-Lynx_full_ip_layers_fp16.safetensors",
            lynx_resampler="lynx_full_resampler_fp32.safetensors",
        )
        wf = build_lynx_workflow(seg, "subject.png", 81)
        assert "full_ip_layers" in wf["601"]["inputs"]["extra_model"]
        assert "full_resampler" in wf["632"]["inputs"]["model_name"]

    def test_ref_blocks_omitted_when_empty(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert "ref_blocks_to_use" not in wf["641"]["inputs"]

    def test_ref_blocks_included_when_set(self):
        seg = make_segment(lynx_ref_blocks_to_use="0-20, 25, 35-39")
        wf = build_lynx_workflow(seg, "subject.png", 81)
        assert wf["641"]["inputs"]["ref_blocks_to_use"] == "0-20, 25, 35-39"

    @pytest.mark.parametrize("scheduler", sorted(WANVIDEO_SCHEDULERS))
    def test_valid_wanvideo_schedulers_pass_through(self, scheduler):
        wf = build_lynx_workflow(make_segment(lynx_scheduler=scheduler), "subject.png", 81)
        assert wf["650"]["inputs"]["scheduler"] == scheduler

    @pytest.mark.parametrize("scheduler", ["simple", "karras", "normal", "bogus"])
    def test_ksampler_scheduler_names_fall_back(self, scheduler):
        """KSampler names would 400 in ComfyUI — fall back rather than submit a bad graph."""
        wf = build_lynx_workflow(make_segment(lynx_scheduler=scheduler), "subject.png", 81)
        assert wf["650"]["inputs"]["scheduler"] == settings.lynx_scheduler


class TestLoraStacking:
    def test_distill_lora_present_by_default(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert wf["700"]["inputs"]["lora"] == settings.lynx_distill_lora
        assert wf["700"]["inputs"]["strength"] == settings.lynx_distill_strength
        assert wf["610"]["inputs"]["lora"] == ["700", 0]

    def test_zero_distill_strength_drops_the_lora(self):
        """De-distilled path: no Lightning LoRA at all, not strength 0."""
        wf = build_lynx_workflow(make_segment(lynx_distill_strength=0.0), "subject.png", 81)
        assert "700" not in wf
        assert "lora" not in wf["610"]["inputs"]

    def test_character_lora_stacks_on_top_of_distill(self, lora):
        wf = build_lynx_workflow(make_segment(loras=[lora]), "subject.png", 81)
        assert wf["700"]["inputs"]["lora"] == settings.lynx_distill_lora
        assert wf["701"]["inputs"]["lora"] == "k3lly_high.safetensors"
        assert wf["701"]["inputs"]["strength"] == 0.8
        assert wf["701"]["inputs"]["prev_lora"] == ["700", 0]
        assert wf["610"]["inputs"]["lora"] == ["701", 0]

    def test_character_lora_without_distill_starts_the_chain(self, lora):
        wf = build_lynx_workflow(
            make_segment(loras=[lora], lynx_distill_strength=0.0), "subject.png", 81
        )
        assert "prev_lora" not in wf["700"]["inputs"]
        assert wf["700"]["inputs"]["lora"] == "k3lly_high.safetensors"

    def test_low_file_used_when_no_high_file(self):
        """Wan2.1 T2V is single-expert; a low-only LoRA still applies."""
        item = LoraItem(lora_id="x", low_file="only_low.safetensors", low_weight=0.5)
        wf = build_lynx_workflow(make_segment(loras=[item]), "subject.png", 81)
        assert wf["701"]["inputs"]["lora"] == "only_low.safetensors"
        assert wf["701"]["inputs"]["strength"] == 0.5

    def test_empty_lora_entry_skipped(self):
        wf = build_lynx_workflow(make_segment(loras=[LoraItem(lora_id="x")]), "subject.png", 81)
        assert "701" not in wf

    def test_multiple_character_loras_chain(self, lora):
        second = LoraItem(lora_id="y", high_file="motion.safetensors", high_weight=1.0)
        wf = build_lynx_workflow(make_segment(loras=[lora, second]), "subject.png", 81)
        assert wf["702"]["inputs"]["prev_lora"] == ["701", 0]
        assert wf["610"]["inputs"]["lora"] == ["702", 0]


class TestOutputStage:
    def test_no_rife_at_generation_fps(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        assert "670" not in wf
        assert wf["680"]["inputs"]["images"] == ["660", 0]

    def test_rife_inserted_for_higher_target_fps(self):
        wf = build_lynx_workflow(make_segment(fps=30), "subject.png", 81)
        assert wf["670"]["inputs"]["multiplier"] == 2
        assert wf["680"]["inputs"]["images"] == ["670", 0]


class TestStructuredLogging:
    def test_graph_build_emits_correlated_json(self, segment, caplog):
        with caplog.at_level("INFO"):
            build_lynx_workflow(segment, "subject.png", 81)
        records = [json.loads(r.message) for r in caplog.records
                   if r.message.startswith("{")]
        built = [r for r in records if r["event"] == "lynx.graph_built"]
        assert len(built) == 1
        assert built[0]["correlation_id"] == str(segment.id)
        assert built[0]["num_frames"] == 81
        assert built[0]["arm"] == "lite"
        assert built[0]["resolution"] == "832x480"


class TestGoldenFile:
    """A fixed param set must serialise to a byte-stable ComfyUI graph."""

    def test_matches_golden(self, segment):
        wf = build_lynx_workflow(segment, "subject.png", 81)
        actual = json.dumps(wf, indent=2, sort_keys=True)
        if not GOLDEN.exists():  # pragma: no cover - regeneration path
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(actual)
        assert actual == GOLDEN.read_text(), (
            "Lynx graph drifted from the golden file. If intentional, delete "
            f"{GOLDEN.name} and re-run to regenerate."
        )
