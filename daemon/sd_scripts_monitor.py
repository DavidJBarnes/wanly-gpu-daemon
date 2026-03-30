"""Monitor sd-scripts (kohya-ss) LoRA training status.

Checks whether the sd-scripts git repo exists on disk and whether
a training process (train_network.py) is currently running.  When a
training process is found, its log output is parsed (via the file
descriptor in /proc) to extract real-time progress metrics.
"""

import logging
import os
import re

from daemon.config import settings

logger = logging.getLogger(__name__)

# Substrings that identify an sd-scripts training process in /proc cmdline.
TRAINING_SIGNATURES = ("train_network.py",)

# --- Log-parsing regexes (same patterns as musubi-tuner-ui) ---
# tqdm: " 45%|████      | 9/20 [02:15<02:45, 15.00s/it]"
_TQDM_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)")
# epoch header: "epoch 1/10"
_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)")
# average loss: "avr_loss=0.0812"
_AVR_LOSS_RE = re.compile(r"avr_loss=([\d.]+)")


def _resolve_path() -> str | None:
    """Resolve and validate the sd-scripts path. Returns None if not found."""
    path = os.path.expanduser(settings.sd_scripts_path)
    if os.path.isdir(path):
        return path
    return None


def check_installed() -> bool:
    """Return True if the sd-scripts directory exists."""
    return _resolve_path() is not None


def _extract_cmdline_arg(parts: list[str], flag: str) -> str | None:
    """Extract a --flag value from a NUL-split cmdline."""
    for i, part in enumerate(parts):
        if part == flag and i + 1 < len(parts):
            return parts[i + 1]
        if part.startswith(f"{flag}="):
            return part.split("=", 1)[1]
    return None


def _find_log_path(pid: int, cmdline_parts: list[str] | None = None) -> str | None:
    """Find the training log file for a process.

    Checks, in order:
    1. /proc/{pid}/fd/1 (stdout) — used when musubi-tuner-ui redirects output
    2. /proc/{pid}/fd/2 (stderr) — tqdm writes to stderr
    3. The --output_dir from cmdline for any .log files
    """
    # Check stdout and stderr file descriptors
    for fd in (1, 2):
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
            if os.path.isfile(target):
                return target
        except (OSError, PermissionError):
            pass

    # Fall back to looking for log files in the output directory
    if cmdline_parts:
        output_dir = _extract_cmdline_arg(cmdline_parts, "--output_dir")
        if output_dir and os.path.isdir(output_dir):
            # Find the most recently modified .log file
            log_files = []
            try:
                for f in os.listdir(output_dir):
                    if f.endswith(".log"):
                        full = os.path.join(output_dir, f)
                        log_files.append((os.path.getmtime(full), full))
            except OSError:
                pass
            if log_files:
                log_files.sort(reverse=True)
                return log_files[0][1]

    return None


def _parse_log_tail(log_path: str) -> dict:
    """Read the tail of a training log and extract progress metrics."""
    result: dict = {}
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 32768))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return result

    # tqdm uses \r for in-place updates, so split on both \r and \n
    lines = re.split(r"[\r\n]+", tail)

    for line in lines:
        tm = _TQDM_RE.search(line)
        if tm:
            result["current_step"] = int(tm.group(2))
            result["total_steps"] = int(tm.group(3))

        em = _EPOCH_RE.search(line)
        if em:
            result["current_epoch"] = int(em.group(1))
            result["max_epochs"] = int(em.group(2))

        lm = _AVR_LOSS_RE.search(line)
        if lm:
            result["current_loss"] = float(lm.group(1))

    # Calculate percentage
    current = result.get("current_step", 0)
    total = result.get("total_steps", 0)
    if total > 0:
        result["pct_complete"] = round(current / total * 100, 2)
    else:
        result["pct_complete"] = 0.0

    return result


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

        # Skip DataLoader/compile worker children of the main training process.
        # The real trainer's parent is the accelerate launcher (which we
        # already filter above), so only skip when the parent is itself a
        # direct training invocation (no "accelerate" in its cmdline).
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat_fields = f.read().decode().split()
                ppid = int(stat_fields[3])
            try:
                with open(f"/proc/{ppid}/cmdline", "rb") as f:
                    parent_cmd = f.read().decode("utf-8", errors="replace")
                if (
                    any(sig in parent_cmd for sig in TRAINING_SIGNATURES)
                    and "accelerate" not in parent_cmd
                ):
                    continue
            except (OSError, PermissionError):
                pass
        except (OSError, PermissionError, IndexError, ValueError):
            pass

        parts = cmdline.split("\x00")

        # Extract output_name from cmdline if present
        output_name = _extract_cmdline_arg(parts, "--output_name")

        info: dict = {
            "pid": pid,
            "output_name": output_name,
        }

        # Try to parse the training log for live progress metrics
        log_path = _find_log_path(pid, parts)
        if log_path:
            progress = _parse_log_tail(log_path)
            info.update(progress)
        else:
            # Fall back to cmdline args for max_epochs if no log available
            max_epochs_str = _extract_cmdline_arg(parts, "--max_train_epochs")
            if max_epochs_str and max_epochs_str.isdigit():
                info["max_epochs"] = int(max_epochs_str)

        return info

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
