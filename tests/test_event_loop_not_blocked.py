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
"""

import ast
import inspect
from pathlib import Path

from daemon import executor

BLOCKING_CALLS = ["measure_motion_magnitude", "identity_score.score_video"]


def _executor_source() -> str:
    return Path(inspect.getfile(executor)).read_text()


class TestOffloaded:
    def test_heavy_analysis_runs_in_a_thread(self):
        src = _executor_source()
        for call in BLOCKING_CALLS:
            # The call must appear as an argument to to_thread, never invoked directly.
            assert f"asyncio.to_thread(\n            {call}" in src or f"asyncio.to_thread({call}" in src, (
                f"{call} must be offloaded with asyncio.to_thread — inline it blocks the "
                f"heartbeat for its whole duration"
            )

    def test_no_direct_invocation_remains(self):
        """A leftover direct call would reintroduce the stall even with a threaded one present."""
        tree = ast.parse(_executor_source())
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name not in BLOCKING_CALLS:
                continue
            # Fine when it is the *argument* to to_thread (that is a Name/Attribute, not a Call),
            # so any Call node with these names is a direct invocation.
            offenders.append(name)
        assert not offenders, f"called directly on the event loop: {offenders}"
