"""Monitor sd-scripts (kohya-ss) LoRA training status.

Checks whether the sd-scripts git repo exists on disk and whether
a training process (train_network.py) is currently running.
"""

import os

from daemon.config import settings

# Substrings that identify an sd-scripts training process in /proc cmdline.
TRAINING_SIGNATURES = ("train_network.py",)


def _resolve_path() -> str | None:
    """Resolve and validate the sd-scripts path. Returns None if not found."""
    path = os.path.expanduser(settings.sd_scripts_path)
    if os.path.isdir(path):
        return path
    return None


def check_installed() -> bool:
    """Return True if the sd-scripts directory exists."""
    return _resolve_path() is not None


def check_training_active() -> dict | None:
    """Scan /proc for a running sd-scripts training process.

    Returns a dict with process info if found, None otherwise.
    """
    my_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace")
        except (OSError, PermissionError):
            continue

        if not any(sig in cmdline for sig in TRAINING_SIGNATURES):
            continue

        # Skip child processes (torch compile workers, accelerate launcher)
        # — only match the actual training script invocation
        if "compile_worker" in cmdline or "accelerate" in cmdline:
            continue

        # Extract output_name from cmdline if present
        output_name = None
        parts = cmdline.split("\x00")
        for i, part in enumerate(parts):
            if part == "--output_name" and i + 1 < len(parts):
                output_name = parts[i + 1]
                break
            if part.startswith("--output_name="):
                output_name = part.split("=", 1)[1]
                break

        return {
            "pid": pid,
            "output_name": output_name,
        }

    return None


def get_status() -> dict:
    """Return full sd-scripts status for heartbeat payload."""
    installed = check_installed()
    training = check_training_active() if installed else None
    return {
        "sd_scripts_installed": installed,
        "sd_scripts_training": training is not None,
        "sd_scripts_training_info": training,
    }
