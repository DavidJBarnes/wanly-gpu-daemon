"""What the daemon asks ltx-engine for.

Pure-logic tests over build_submit_payload. The interesting failures here are all silent:
every one of them still returns a perfectly plausible video, just not the one that was
validated — which is exactly the class of bug the graph hash exists to catch, and which is
worth catching one layer earlier.
"""

from daemon.ltx_client import build_submit_payload, character_entries

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
    assert lora["name"] == "k3lly2026_v2.safetensors"
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


def test_strengths_arrive_as_strings_from_the_sheet_and_are_coerced():
    """GET /recipes returns char_s1/char_s2 as STRINGS — '0.8', not 0.8.

    Verified against the live engine on 2026-08-31. The recipe sheet is the source of truth
    and its cells serialise as text, so the blob stored on the segment carries strings. Sent
    unconverted they would reach the engine as strings too; float() here is load-bearing, not
    defensive tidying.
    """
    lora = _payload(recipe={**RECIPE, "char_s1": "0.8", "char_s2": "1.5"})["loras"][0]
    assert lora["strength_stage_1"] == 0.8
    assert lora["strength_stage_2"] == 1.5
    assert isinstance(lora["strength_stage_2"], float)


def test_payload_keys_match_the_engines_request_model():
    """Every key must exist on ltx-engine's JobRequest.

    Checked against the live /openapi.json on 2026-08-31: 27 fields, only `keyframes`
    required, and nothing the daemon sends is rejected. This encodes the field NAMES so a
    rename on either side shows up here rather than as a 422 ten minutes into a render.
    """
    engine_job_request_fields = {
        "prompt", "negative_prompt", "loras", "workflow", "checkpoint", "cfg",
        "multimodal_guidance", "stg", "rescale", "stg_blocks", "cfg_stage_1", "cfg_stage_2",
        "distilled_lora_strength", "steps_stage_1", "steps_stage_2", "recipe", "character",
        "keyframes", "width", "height", "num_frames", "frame_rate", "seed", "snap_indices",
    }
    assert set(_payload()) <= engine_job_request_fields


def test_lora_name_is_sent_with_its_file_extension():
    """The engine matches LoRAs by exact filename.

    A real render failed with `no such lora 'pay_v2_e05'` while
    'pay_v2_e05.safetensors' sat in the list the error itself printed. Names arrive bare
    because that is how the recipe sheet wrote them and how the console displays them.
    """
    assert _payload()["loras"][0]["name"] == "k3lly2026_v2.safetensors"


def test_an_already_qualified_name_is_not_double_suffixed():
    lora = _payload(recipe={**RECIPE, "char_lora": "k3lly2026_v2.safetensors"})["loras"][0]
    assert lora["name"] == "k3lly2026_v2.safetensors"


# ---------------------------------------------------------------- two people (console#473)

TWO = dict(RECIPE, characters=[
    {"name": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05", "s1": 0.8, "s2": 1.5},
    {"name": "Me", "trigger": "d@vid", "char_lora": "david_v1_final", "s1": 0.7, "s2": 1.2},
])


def test_two_characters_become_two_lora_entries_in_order():
    """On the recipe path the engine reads every `loras` entry as a person, slot 0 first."""
    loras = _payload(recipe=TWO)["loras"]
    assert [l["name"] for l in loras] == ["pay_v2_e05.safetensors", "david_v1_final.safetensors"]
    assert (loras[1]["strength_stage_1"], loras[1]["strength_stage_2"]) == (0.7, 1.2)


def test_the_scalar_shape_still_builds_exactly_one_entry():
    """A blob from before console#473, byte for byte."""
    assert _payload()["loras"] == [{
        "name": "k3lly2026_v2.safetensors",
        "strength": 0.8, "strength_stage_1": 0.8, "strength_stage_2": 1.5,
    }]


def test_characters_wins_over_the_mirrored_scalars():
    """The scalars mirror characters[0]; if a blob disagrees, the list is the record."""
    blob = dict(TWO, char_lora="something_else")
    assert _payload(recipe=blob)["loras"][0]["name"] == "pay_v2_e05.safetensors"


def test_a_none_second_character_is_dropped_not_sent():
    blob = dict(TWO)
    blob["characters"] = [TWO["characters"][0], {"char_lora": "none", "s1": 1, "s2": 1}]
    assert len(_payload(recipe=blob)["loras"]) == 1


def test_a_half_configured_second_slot_is_dropped_like_the_first_always_was():
    blob = dict(TWO)
    blob["characters"] = [TWO["characters"][0], {"char_lora": "david_v1_final"}]
    assert len(_payload(recipe=blob)["loras"]) == 1


def test_an_empty_list_falls_back_to_the_scalars():
    assert _payload(recipe=dict(RECIPE, characters=[]))["loras"][0]["name"] == "k3lly2026_v2.safetensors"
