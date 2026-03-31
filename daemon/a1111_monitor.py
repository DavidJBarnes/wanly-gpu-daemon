"""Monitor A1111 Stable Diffusion via its systemd unit (sd)."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def _is_unit_active(unit: str) -> bool:
    """Return True if the given systemd unit is active (running)."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _unit_exists(unit: str) -> bool:
    """Return True if the systemd unit file exists (installed)."""
    try:
        result = subprocess.run(
            ["systemctl", "cat", unit],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_status() -> dict:
    """Return A1111 status for heartbeat payload."""
    installed = _unit_exists("sd")
    running = _is_unit_active("sd") if installed else False
    return {
        "a1111_installed": installed,
        "a1111_running": running,
    }
