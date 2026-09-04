"""What the daemon forwards to ltx-engine for a pose's content LoRA.

The content LoRA is the POSE's — motion and act — and is chained ahead of the character
LoRA, which is identity. They are different axes and the payload has to keep them apart:
the engine reads `loras[0]` as the CHARACTER LoRA on this path, so a content LoRA appended
to that list would be loaded as a character.

Nothing here would raise if it were wrong. Every failure mode in this file produces a
render that succeeds at the wrong configuration.
"""
from daemon.ltx_client import build_submit_payload


def _payload(**recipe):
    base = {"recipe": "Doggystyle Side", "character": "k3lly2026",
            "char_lora": "k3lly2026_v2", "char_s1": 0.8, "char_s2": 1.5}
    base.update(recipe)
    return build_submit_payload(
        image_bytes=None, prompt="p", negative_prompt=None,
        width=832, height=1216, num_frames=241, frame_rate=24, seed=1,
        recipe=base,
    )


def test_a_pose_without_a_content_lora_sends_none():
    """Every pose today. The engine then renders on the checkpoint plus the character."""
    assert "content_loras" not in _payload()


def test_none_in_any_spelling_is_not_sent_as_a_filename():
    """"none" is how a pose says off. Sent through, the engine would look for
    'none.safetensors' and the segment would die ten minutes in."""
    for spelling in ("none", "NONE", " None ", ""):
        assert "content_loras" not in _payload(content_loras=[{"name": spelling}])


def test_the_content_lora_is_its_own_field_not_a_second_entry_in_loras():
    """`loras[0]` is the CHARACTER LoRA on this path. A content LoRA appended there would
    be read as a character — both load, the render succeeds, and it is the wrong person."""
    p = _payload(content_loras=[{"name": "sfbehind_LTX2_3_v0_1"}])
    assert p["content_loras"][0]["name"] == "sfbehind_LTX2_3_v0_1.safetensors"
    assert len(p["loras"]) == 1
    assert p["loras"][0]["name"] == "k3lly2026_v2.safetensors"


def test_the_extension_is_added_because_the_engine_matches_files_exactly():
    """A real render died on `no such lora 'pay_v2_e05'` while the .safetensors sat in the
    very list the error printed."""
    p = _payload(content_loras=[{"name": "sfbehind_LTX2_3_v0_1"}])
    assert p["content_loras"][0]["name"].endswith(".safetensors")
    already = _payload(content_loras=[{"name": "sfbehind_LTX2_3_v0_1.safetensors"}])
    assert already["content_loras"][0]["name"] == "sfbehind_LTX2_3_v0_1.safetensors"  # not doubled


def test_both_strengths_are_forwarded_independently():
    p = _payload(content_loras=[{"name": "sfbehind_LTX2_3_v0_1", "s1": 0.3, "s2": 1.1}])
    assert p["content_loras"][0]["s1"] == 0.3
    assert p["content_loras"][0]["s2"] == 1.1


def test_a_strength_of_zero_is_forwarded_and_not_dropped():
    """The `is not None` rule, same as img_compression.

    0 loads the LoRA and gives it no weight — how you measure what it contributes. Dropped,
    the engine applies its own 0.6 default and the measurement is of another configuration
    entirely, with nothing in any log to say so.
    """
    p = _payload(content_loras=[{"name": "sfbehind_LTX2_3_v0_1", "s1": 0.0, "s2": 0.0}])
    assert p["content_loras"][0]["s1"] == 0.0
    assert p["content_loras"][0]["s2"] == 0.0


def test_the_order_of_the_list_is_preserved():
    """Order is part of the configuration: the same LoRAs applied in a different order are a
    different render, so the payload must not reorder or deduplicate."""
    p = _payload(content_loras=[{"name": "first"}, {"name": "second"}, {"name": "third"}])
    assert [c["name"] for c in p["content_loras"]] == [
        "first.safetensors", "second.safetensors", "third.safetensors"]


def test_a_none_entry_is_dropped_without_disturbing_the_rest():
    """A pose may carry an off entry among live ones. Skipping it must not reorder what
    remains."""
    p = _payload(content_loras=[{"name": "a"}, {"name": "none"}, {"name": "b"}])
    assert [c["name"] for c in p["content_loras"]] == ["a.safetensors", "b.safetensors"]


def test_the_pose_checkpoint_is_forwarded():
    """Without this the engine falls back to its default and every render uses one base
    model — which is exactly the state that made sulphur-vs-10Eros impossible to compare."""
    p = _payload(checkpoint="10Eros_v1.5_bf16")
    assert p["checkpoint"] == "10Eros_v1.5_bf16"


def test_no_checkpoint_means_the_engine_default():
    """Every pose today. The key must be absent rather than null, so the engine applies its
    own default rather than being handed one to interpret."""
    assert "checkpoint" not in _payload()


def test_a_character_lora_of_none_is_not_sent():
    """"none" means render without a character LoRA (console#412).

    It must be filtered here rather than left to the engine: `if lora` alone passes, because
    the STRING "none" is truthy, and the entry would then be normalised to
    "none.safetensors" — no longer the literal "none" the engine's want_char check looks
    for. It would sail past that and 422 on a file that does not exist, ten minutes into a
    claimed segment.
    """
    for spelling in ("none", "NONE", " None "):
        p = _payload(char_lora=spelling)
        assert "loras" not in p or p["loras"] == [], f"{spelling!r} was forwarded as a LoRA"


def test_a_real_character_lora_is_still_sent():
    """The guard must not swallow the normal case."""
    p = _payload(char_lora="k3lly2026_v2")
    assert p["loras"][0]["name"] == "k3lly2026_v2.safetensors"
