"""What the daemon asks ltx-engine for.

Pure-logic tests over build_submit_payload. The interesting failures here are all silent:
every one of them still returns a perfectly plausible video, just not the one that was
validated — which is exactly the class of bug the graph hash exists to catch, and which is
worth catching one layer earlier.
"""

from daemon.ltx_client import build_submit_payload

RECIPE = {
    "recipe": "Missionary POV",
    "character": "k3lly2026",
    "char_lora": "k3lly2026_v2",
    "char_s1": 0.8,
    "char_s2": 1.5,
    "frames": 241,
    "graph_sha256": "85649768667ba700fdd75643be3ba031",
}


def _payload(**over):
    args = dict(
        image_bytes=None, prompt="a prompt", negative_prompt="bad things",
        width=768, height=1344, num_frames=241, frame_rate=24, seed=42, recipe=RECIPE,
    )
    args.update(over)
    return build_submit_payload(**args)


def test_character_lora_is_sent_per_stage_not_flat():
    """0.8/1.5 is the validated split; one number for both stages is a different config.

    Stage 1 generates at half size from noise and stage 2 refines the 2x-upscaled latent, so
    the two are not interchangeable. Collapsing them renders something that looks fine and is
    not the signed-off configuration.
    """
    lora = _payload()["loras"][0]
    assert lora["name"] == "k3lly2026_v2"
    assert lora["strength_stage_1"] == 0.8
    assert lora["strength_stage_2"] == 1.5


def test_graph_hash_is_never_sent_back_to_the_engine():
    """graph_sha256 is a record of what ran, not an input.

    Sending it back would offer the engine a hash to honour rather than one to produce,
    inverting the point of hashing the resolved graph — the check would then confirm that we
    told it what to say.
    """
    assert "graph_sha256" not in _payload()


def test_recipe_and_character_are_passed_through():
    p = _payload()
    assert p["recipe"] == "Missionary POV"
    assert p["character"] == "k3lly2026"


def test_seed_is_narrowed_to_what_the_engine_accepts():
    """The queue derives 63-bit seeds; the engine takes signed 32-bit.

    Narrowed here so the engine never has to reject or silently wrap it — a wrapped seed
    reproduces a different clip from the number recorded beside it.
    """
    assert _payload(seed=2**62 + 7)["seed"] < 2**31 - 1


def test_free_form_render_sends_no_recipe_fields():
    p = _payload(recipe=None)
    assert "recipe" not in p and "character" not in p and "loras" not in p
    assert p["prompt"] == "a prompt"


def test_partial_recipe_does_not_produce_a_half_configured_lora():
    """A recipe missing a strength must send NO lora, not one with a guessed strength.

    Defaulting the missing half would render at a strength nobody chose and report success.
    """
    assert "loras" not in _payload(recipe={**RECIPE, "char_s2": None})
    assert "loras" not in _payload(recipe={**RECIPE, "char_lora": None})


def test_start_frame_becomes_a_data_uri_keyframe():
    p = _payload(image_bytes=b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    assert p["keyframes"][0]["image"].startswith("data:image/png;base64,")


def test_no_start_frame_is_an_empty_keyframe_list():
    assert _payload(image_bytes=None)["keyframes"] == []
