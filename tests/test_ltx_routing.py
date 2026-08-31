"""Which engine a worker drives, and when it declines to claim.

ENGINE is a property of the WORKER — which models and services its container actually has —
not of the job. Deciding per segment instead would let a worker with no ltx-engine claim an
LTX render on the strength of a field and then fail it, and a failed segment is not free: it
needs a human to notice and retry.
"""

import ast
from pathlib import Path


def _source(path: str) -> str:
    return Path(path).read_text()


def test_ltx_route_is_selected_by_config_not_by_the_segment():
    """The branch must read settings.engine, never segment.ltx_recipe.

    Routing on the presence of a recipe would also misroute a free-form LTX render, which
    legitimately has no recipe at all.
    """
    src = _source("daemon/executor.py")
    tree = ast.parse(src)
    dispatch = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "execute_segment"
    )
    branch = next(
        (n for n in ast.walk(dispatch)
         if isinstance(n, ast.Compare)
         and isinstance(n.left, ast.Attribute) and n.left.attr == "engine"),
        None,
    )
    assert branch is not None, "execute_segment does not branch on settings.engine"


def test_default_engine_is_wan_so_merging_this_changes_no_running_worker():
    """start.sh git-pulls the daemon on EVERY boot.

    A default of "ltx" here would therefore retarget any existing worker the moment it
    restarted, pointing it at an ltx-engine its container does not run. The LTX image sets
    ENGINE explicitly instead (wanly-gpu-docker#41).
    """
    from daemon.config import Settings
    # The FIELD default, not Settings(): instantiating reads .env, so this asserted whatever
    # the local worker happened to be configured as. It passed in CI and failed on any box
    # with a real .env — a test that measures the environment instead of the code.
    assert Settings.model_fields["engine"].default == "wan22"


def test_ltx_worker_does_not_claim_while_the_engine_is_down():
    """Claiming work this worker cannot render burns the segment."""
    src = _source("daemon/main.py")
    assert "_ltx_healthy" in src
    poll = src.split("async def job_poll_loop")[1]
    gate = poll.index("_ltx_healthy()")
    claim = poll.index("queue.claim_next")
    assert gate < claim, "the ltx-engine health gate must precede the claim, not follow it"


def test_wan_model_validation_is_skipped_for_an_ltx_worker():
    """MODEL_CHECKS describes the WAN 2.2 model set.

    Running it against an LTX worker would refuse a perfectly good one at boot. Replacing the
    set is wanly-gpu-docker#41; until then it is skipped, and the health gate above is what
    keeps an unready LTX worker from claiming.
    """
    src = _source("daemon/main.py")
    assert 'settings.engine == "ltx"' in src
    head = src.index('if settings.engine == "ltx":')
    assert src.index("validate_models(comfyui)", head) > head


def test_an_ltx_worker_does_not_clear_comfyuis_queue_at_startup():
    """ltx-engine owns ComfyUI's queue on the LTX path.

    The daemon clearing it at startup would kill whatever the engine is rendering — there is
    one GPU and one queue, and a wholesale clear takes the in-flight job with it. The damage
    is invisible from the daemon's side too: the engine reports its job failed, and nothing
    connects that to a worker restart.

    Under WAN the daemon DID own the queue, so the clear was correct there and stays.
    """
    src = _source("daemon/main.py")
    tree = ast.parse(src)
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # the clear_queue call has to sit under a test that mentions the engine
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if "clear_queue" not in body:
            continue
        if "engine" in ast.dump(node.test):
            guarded = True
    assert guarded, (
        "clear_queue at startup is not gated on settings.engine — an LTX worker restarting "
        "mid-render would clear the engine's job out from under it"
    )
