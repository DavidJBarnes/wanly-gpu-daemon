"""What code this worker is actually running (wanly-gpu-docker#72).

Two workers ran different code for 14 hours and nothing said so. It surfaced as a 422 on a
content LoRA that looked random: a RunPod pod fetched it correctly while the 3090 could not,
because the pod's daemon was current and the 3090's had been cloned before the fix existed.
Working that out meant SSHing into both boxes and reading git.

TWO VALUES, because there are two update channels and they drift separately:

    daemon commit   this checkout, re-cloned from main by start.sh at every container boot
    image ref       start.sh, download_models.sh and the engine, baked into the image and
                    only changed by a pull + recreate

`docker restart` moves the first and not the second. The 3090 spent 37 hours in exactly that
state, and a single "version" string could not have expressed it.

Read ONCE at import. Both are fixed for the life of the process -- the checkout cannot change
under a running daemon and neither can the image -- so re-reading them on every heartbeat
would spend a subprocess every 30 seconds to re-answer a settled question.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Baked into the image at build time by the Docker build (ARG -> ENV). Absent when the daemon
# runs outside the image, which is a real case for local development and must not look like
# an image whose ref is the string "None".
_IMAGE_ENV = "WANLY_IMAGE_REF"


def _read_commit() -> str | None:
    """The short sha of the daemon checkout, or None if this is not a git checkout.

    None is a real answer: it means "cannot tell", which is different from a commit and must
    not be dressed up as one. start.sh clones with --depth 1, so only HEAD is available --
    describe/tags would fail on a shallow clone and are deliberately not attempted.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as e:  # noqa: BLE001 - git missing, or a hung filesystem
        logger.debug("could not read the daemon commit: %s", e)
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip() or None


def _read_image() -> str | None:
    value = (os.environ.get(_IMAGE_ENV) or "").strip()
    return value or None


DAEMON_COMMIT: str | None = _read_commit()
IMAGE_REF: str | None = _read_image()


def describe() -> str:
    """One line for the boot log. Says "unknown" rather than omitting a field, because a
    missing line reads as "nothing to report" and an unknown one reads as a question."""
    return (f"daemon={DAEMON_COMMIT or 'unknown'} image={IMAGE_REF or 'unknown'}")
