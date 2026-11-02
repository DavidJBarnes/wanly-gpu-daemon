"""The Settings stage 2 schedule, from claim to engine request (console#717)."""
from daemon.ltx_client import build_submit_payload
from tests.test_identity_ref import _run


def _payload(recipe):
    return build_submit_payload(
        image_bytes=None, prompt="p", negative_prompt=None, width=1216, height=832,
        num_frames=241, frame_rate=24, seed=1, recipe=recipe)


def test_forwarded_when_the_claim_carries_it():
    assert _payload({"recipe": "r", "stage2_sigmas": "0.725, 0.4219, 0.0"})[
        "stage2_sigmas"] == "0.725, 0.4219, 0.0"


def test_absent_when_not_set():
    assert "stage2_sigmas" not in _payload({"recipe": "r"})


def test_an_engine_without_the_feature_is_refused_not_ignored(monkeypatch):
    seen = _run(monkeypatch, None, features=(), recipe_extra={"stage2_sigmas": "0.7, 0.0"})
    assert "submit" not in seen
    assert "does not support stage 2 sigmas" in seen["failed"]


def test_a_capable_engine_gets_it_through_the_real_submit_path(monkeypatch):
    seen = _run(monkeypatch, None, features=("stage2_sigmas",),
                recipe_extra={"stage2_sigmas": "0.7, 0.0"})
    assert seen["submit"]["recipe"]["stage2_sigmas"] == "0.7, 0.0"


def test_nothing_set_asks_the_engine_nothing(monkeypatch):
    assert _run(monkeypatch, None)["features_asked"] == 0
