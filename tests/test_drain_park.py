"""A drained worker on a box parks; only a pod exits.

On 2026-09-08 the drained render worker on 3090.zero exited as designed, the container's
restart policy brought it straight back, it registered a fresh row with no drain, the update
timer then recreated it on a new image (it looked idle), and it claimed a render beside a
55-minute training run. The box hard-reset under both loads and the training was lost.
"""
import inspect

from daemon import main as mod


def test_off_runpod_the_poll_loop_parks_instead_of_shutting_down():
    src = inspect.getsource(mod.job_poll_loop)
    park = src.index("if drain_event.is_set():")
    block = src[park:park + 900]
    assert "if settings.runpod_pod_id:" in block
    assert "shutdown_event.set()" in block
    assert "await comfyui.free_memory()" in block
    assert "continue" in block


def test_parking_unloads_the_models_once():
    """The trainer waits for VRAM, not for a status; an engine holding 13 GB idle is not free."""
    src = inspect.getsource(mod.job_poll_loop)
    assert "parked = False" in src and "parked = True" in src
    assert "if not parked:" in src


def test_a_finished_segment_does_not_shut_a_box_down():
    src = inspect.getsource(mod.job_poll_loop)
    tail = src[src.index("executing_event.clear()"):]
    assert "if settings.runpod_pod_id:" in tail
    assert "parking" in tail


def test_the_heartbeat_releases_the_drain_when_the_row_is_no_longer_draining():
    src = inspect.getsource(mod.heartbeat_loop)
    assert 'data.get("status") != "draining"' in src
    assert "drain_event.clear()" in src
    # and never on a pod, which drains to terminate
    assert "not on_runpod" in src


def test_no_status_push_while_parked():
    """Pushing online-idle from our side would undo the drain, the heartbeat would read the
    row as released, and the worker would claim beside the trainer."""
    src = inspect.getsource(mod.heartbeat_loop)
    assert "if is_busy != last_busy_state and not (drain_event.is_set() and not on_runpod):" in src


def test_free_memory_posts_comfyuis_free():
    from daemon.comfyui_client import ComfyUIClient
    src = inspect.getsource(ComfyUIClient.free_memory)
    assert '"/free"' in src and '"unload_models": True' in src
