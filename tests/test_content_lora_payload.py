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
    p = _payload()
    assert "content_lora" not in p
    assert "content_s1" not in p and "content_s2" not in p


def test_none_in_any_spelling_is_not_sent_as_a_filename():
    """"none" is how the stack says off. Sent through, the engine would look for
    'none.safetensors' and the segment would die ten minutes in."""
    for spelling in ("none", "NONE", " None ", ""):
        assert "content_lora" not in _payload(content_lora=spelling)


def test_the_content_lora_is_its_own_field_not_a_second_entry_in_loras():
    """`loras[0]` is the CHARACTER LoRA on this path. A content LoRA appended there would
    be read as a character — both load, the render succeeds, and it is the wrong person."""
    p = _payload(content_lora="sfbehind_LTX2_3_v0_1")
    assert p["content_lora"] == "sfbehind_LTX2_3_v0_1.safetensors"
    assert len(p["loras"]) == 1
    assert p["loras"][0]["name"] == "k3lly2026_v2.safetensors"


def test_the_extension_is_added_because_the_engine_matches_files_exactly():
    """A real render died on `no such lora 'pay_v2_e05'` while the .safetensors sat in the
    very list the error printed."""
    assert _payload(content_lora="sfbehind_LTX2_3_v0_1")["content_lora"].endswith(".safetensors")
    already = _payload(content_lora="sfbehind_LTX2_3_v0_1.safetensors")["content_lora"]
    assert already == "sfbehind_LTX2_3_v0_1.safetensors"   # not doubled


def test_both_strengths_are_forwarded_independently():
    p = _payload(content_lora="sfbehind_LTX2_3_v0_1", content_s1=0.3, content_s2=1.1)
    assert p["content_s1"] == 0.3
    assert p["content_s2"] == 1.1


def test_a_strength_of_zero_is_forwarded_and_not_dropped():
    """The `is not None` rule, same as img_compression.

    0 loads the LoRA and gives it no weight — how you measure what it contributes. Dropped,
    the engine applies its own 0.6 default and the measurement is of another configuration
    entirely, with nothing in any log to say so.
    """
    p = _payload(content_lora="sfbehind_LTX2_3_v0_1", content_s1=0.0, content_s2=0.0)
    assert p["content_s1"] == 0.0
    assert p["content_s2"] == 0.0


def test_strengths_without_a_lora_are_not_sent_alone():
    """They would configure a LoRA that is not being loaded — noise in the payload, and a
    misleading record of what ran."""
    p = _payload(content_s1=0.3, content_s2=1.1)
    assert "content_s1" not in p and "content_s2" not in p
