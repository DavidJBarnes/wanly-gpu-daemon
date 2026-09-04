"""What the engine already knows, put where a person can read it (console#392).

`GET /job/{id}` on ltx-engine returns `stages` and `loras` with fusion coverage. The daemon
read only `notes`, so the rest existed solely in /workspace/logs/ltx-engine.log inside the
container — which is how answering "did the character LoRA actually apply?" became a
docker exec.

The fusion count is the one that matters. A LoRA whose keys do not line up against the
checkpoint fuses NOTHING and says nothing about it: no error, no warning, the run looks
normal and comes back as the base model with none of the character in it.
"""
from daemon.ltx_executor import _engine_detail


def test_a_full_fuse_is_reported_not_just_a_failure():
    """480/480 is exactly as informative as 0/480 when comparing base models. Only the zero
    case was ever visible, which is half an answer."""
    out = _engine_detail({"loras": [
        {"name": "k3lly2026_v2.safetensors", "strength_stage_1": 0.8,
         "strength_stage_2": 1.5, "fused": 480, "targeted": 480}]})
    assert any("fused 480/480" in l for l in out)
    assert any("@0.8/1.5" in l for l in out)


def test_a_zero_fuse_is_stated_in_terms_of_what_it_means():
    """"fused 0/480" is a number. "carries NONE of it" is what the number means, and the
    person reading a disappointing render needs the second one."""
    out = _engine_detail({"loras": [
        {"name": "k3lly2026_v2.safetensors", "fused": 0, "targeted": 480}]})
    assert any("FUSED 0/480" in l and "carries NONE" in l for l in out)


def test_unavailable_coverage_is_not_silently_the_same_as_zero():
    """The engine could not read the checkpoint to compare against. "no line" and "zero"
    must not look alike — one is a missing measurement, the other is a broken render."""
    out = _engine_detail({"loras": [
        {"name": "x.safetensors", "fused": None, "targeted": None}]})
    assert any("unavailable" in l for l in out)
    assert not any("0/" in l for l in out)


def test_stages_describe_what_each_pass_ran():
    out = _engine_detail({"stages": [
        {"stage": 1, "steps": 8, "schedule": "distilled sigma table"},
        {"stage": 2, "steps": 3, "schedule": "distilled sigma table"}]})
    line = next(l for l in out if l.startswith("passes"))
    assert "stage 1: 8 steps" in line and "stage 2: 3 steps" in line


def test_it_is_silent_when_there_is_nothing_to_say():
    """A render with no LoRAs must not gain empty lines. This runs on every segment, so
    noise here is noise on everything."""
    assert _engine_detail({}) == []
    assert _engine_detail({"loras": [], "stages": []}) == []
