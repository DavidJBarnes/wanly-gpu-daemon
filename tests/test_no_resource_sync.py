"""The WAN-era resource sync is gone, and must not come back (#175).

It carried a one-entry manifest -- rife49.pth for ComfyUI-Frame-Interpolation, a node pack
removed with WAN -- fetched from `s3://wanly-resources/...`, a bucket that no longer exists:

    $ aws s3 ls s3://wanly-resources/comfyui/models/rife/
    NoSuchBucket: The specified bucket does not exist

And main() treated a sync failure as FATAL: "Resource sync failed. Exiting.", which
restart-loops the container. It was dormant only because settings.comfyui_path is empty,
which made sync_resources return True without doing anything -- so the hard failure was one
environment variable away from every LTX worker, permanently.

Nothing under LTX needs it: ltx-engine owns its own ComfyUI and validates its own models,
and character LoRAs arrive through LORA_CACHE_DIR.
"""
import importlib

import pytest


def test_the_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daemon.resource_sync")


def test_startup_does_not_import_it():
    """A re-added import is how this would return, and it would be invisible until someone
    set COMFYUI_PATH."""
    import inspect

    from daemon import main as main_mod
    src = inspect.getsource(main_mod)
    assert "resource_sync" not in src
    assert "sync_resources" not in src


def test_no_reference_to_the_deleted_bucket():
    """`wanly-resources` does not exist. Any code still naming it can only fail."""
    import pathlib
    root = pathlib.Path(__file__).parent.parent / "daemon"
    hits = [p.name for p in root.rglob("*.py") if "wanly-resources" in p.read_text()]
    assert hits == [], f"still referencing the deleted bucket: {hits}"


def test_setting_comfyui_path_no_longer_arms_a_fatal_startup_step():
    """The acceptance criterion, as close as a unit test gets.

    Previously, a non-empty comfyui_path turned the dormant sync live and a 404 exited the
    daemon. With the module gone there is no longer any startup step that both reads
    comfyui_path and can terminate the daemon.
    """
    import inspect

    from daemon import main as main_mod
    src = inspect.getsource(main_mod.main) if hasattr(main_mod, "main") else inspect.getsource(main_mod)
    # The node check also reads comfyui_path and CAN exit -- that one is deliberate and
    # unrelated (it guards against a broken ComfyUI), so it is named here rather than
    # asserted away, to keep this test honest about what it does and does not cover.
    assert "Resource sync failed" not in src
