"""The post-sampling analysis must not block the daemon's event loop.

Reported live: a RunPod worker showed `offline` on the Workers page while it was plainly still
working a segment. Its last heartbeat was 240s old; the pod was RUNNING.

Cause: motion analysis (OpenCV optical flow, ~38s measured) and identity scoring (insightface
over 289 frames — 9s at 480p, 126s at 720p, 3m43s on one continuation) both ran inline on the
event loop. While they ran, nothing else on that loop could — including the 30s heartbeat.

Three consequences, in increasing severity:

  1. The worker reads as offline while working. Confusing.
  2. The API's offline sweep overwrites a `draining` status, so a pending drain is silently
     lost and the worker resumes claiming once heartbeats resume.
  3. The stale-claim reaper is documented as safe because "a healthy worker mid-render is still
     heartbeating". That premise was false, so a segment still being finished could be reclaimed
     by another worker.

**Both analyses were removed entirely in #151** — they cost more wall clock than the render
they measured, and they existed to compensate for WAN 2.2 drifting. So the stall cannot happen
today for want of a subject.

This file stays because the LESSON outlives the code: any heavy CPU work added to the
execution path has to go off the event loop, and the third consequence above means the cost of
forgetting is not cosmetic. These tests now guard the removal, so the stall cannot come back by
the same door.
"""

import ast
import inspect
from pathlib import Path

from daemon import executor, ltx_executor

REMOVED = ["measure_motion_series", "identity_score", "_score_segment_identity"]


def _sources() -> dict[str, str]:
    return {
        "executor": Path(inspect.getfile(executor)).read_text(),
        "ltx_executor": Path(inspect.getfile(ltx_executor)).read_text(),
    }


def test_the_blocking_analyses_are_gone_from_both_executors():
    for name, src in _sources().items():
        tree = ast.parse(src)
        called = {
            n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        for gone in REMOVED:
            assert gone not in called, (
                f"{name} calls {gone}, removed in #151. If heavy analysis is ever "
                f"reintroduced it must be offloaded with asyncio.to_thread — inline it "
                f"blocks the heartbeat, and the stale-claim reaper assumes heartbeats "
                f"continue mid-render."
            )


def test_the_modules_are_not_imported_anywhere_in_the_daemon():
    """An import is how a removal quietly grows back."""
    root = Path(inspect.getfile(executor)).parent
    for f in root.glob("*.py"):
        src = f.read_text()
        assert "import identity_score" not in src, f"{f.name} imports identity_score"
        assert "import motion_analyzer" not in src, f"{f.name} imports motion_analyzer"
        assert "from daemon.motion_analyzer" not in src, f"{f.name} imports motion_analyzer"


def test_uploading_a_result_is_not_gated_on_any_metric():
    """The segment's status update must not depend on a measurement existing.

    It used to read `if result and result.motion_magnitude:` — so a render whose motion
    measured 0.0 never had its result sent, and the segment sat un-updated. Removing the
    metrics turned that latent bug into an AttributeError on every completed segment.

    Whether a metric was produced has nothing to do with whether the segment finished.
    """
    import inspect
    import textwrap
    from daemon import queue_client

    # Parsed, not grepped: the explanatory comment in that function names the very attribute
    # this asserts absent, so a text search matches its own documentation.
    src = textwrap.dedent(inspect.getsource(queue_client.QueueClient.upload_segment_output))
    fn = ast.parse(src).body[0]
    guards = [n.test for n in ast.walk(fn) if isinstance(n, ast.If)]
    for test in guards:
        names = {
            getattr(a, "attr", "") for a in ast.walk(test) if isinstance(a, ast.Attribute)
        }
        assert not (names & {"motion_magnitude"}), (
            "the result upload is gated on a metric again — status has nothing to do with "
            "whether a measurement was produced"
        )
        assert not any(n.startswith("identity_") for n in names), (
            "the result upload is gated on an identity metric"
        )
