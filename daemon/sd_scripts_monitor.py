"""Monitor sd-scripts (kohya-ss) LoRA training status.

Checks whether the sd-scripts git repo exists on disk and whether
a training process (train_network.py) is currently running.  When a
training process is found, its log output is parsed (via the file
descriptor in /proc) to extract real-time progress metrics.
"""

import logging
import os
import re
import struct

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


def _find_tfevents_file(logging_dir: str) -> str | None:
    """Find the most recent TensorBoard events file under logging_dir.

    sd-scripts writes to: {logging_dir}/{timestamp}/{subdir}/events.out.tfevents.*
    """
    best: tuple[float, str] | None = None
    try:
        for root, _dirs, files in os.walk(logging_dir):
            for f in files:
                if f.startswith("events.out.tfevents."):
                    full = os.path.join(root, f)
                    mtime = os.path.getmtime(full)
                    if best is None or mtime > best[0]:
                        best = (mtime, full)
    except OSError:
        pass
    return best[1] if best else None


def _parse_tfevents_tail(path: str) -> dict:
    """Read the tail of a TensorBoard events file for loss and step.

    Uses raw struct parsing of the TFRecord format + minimal protobuf
    wire-format decoding — no external dependencies needed.

    TFRecord format per record:
      8 bytes: uint64 length
      4 bytes: masked crc32c of length
      `length` bytes: data (serialized tf.Event protobuf)
      4 bytes: masked crc32c of data

    Event protobuf fields we care about:
      field 2 (int64): step
      field 5 (Summary → repeated Value):
        Value field 1 (string): tag
        Value field 2 (float): simple_value
    """
    result: dict = {}
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # Read last 64KB — enough for recent events
            f.seek(max(0, size - 65536))
            data = f.read()
    except OSError:
        return result

    # Scan for TFRecord boundaries and parse each record
    last_step = 0
    last_loss = None
    pos = 0
    # If we seeked into the middle of the file, find the first valid record
    # by scanning for plausible record lengths
    while pos + 12 < len(data):
        try:
            length = struct.unpack("<Q", data[pos : pos + 8])[0]
        except struct.error:
            pos += 1
            continue
        record_end = pos + 12 + length + 4
        if length > 100000 or record_end > len(data):
            pos += 1
            continue

        record_data = data[pos + 12 : pos + 12 + length]
        step, loss = _decode_event(record_data)
        if step is not None and step > last_step:
            last_step = step
        if loss is not None:
            last_loss = loss
        pos = record_end

    if last_step > 0:
        result["current_step"] = last_step
    if last_loss is not None:
        result["current_loss"] = round(last_loss, 4)

    return result


def _decode_event(data: bytes) -> tuple[int | None, float | None]:
    """Minimal protobuf decoder for tf.Event — extracts step and loss/avr_loss."""
    step = None
    loss = None
    pos = 0
    while pos < len(data):
        try:
            tag_byte = data[pos]
            field_num = tag_byte >> 3
            wire_type = tag_byte & 0x07
            pos += 1
        except IndexError:
            break

        if wire_type == 0:  # varint
            val, pos = _decode_varint(data, pos)
            if val is None:
                break
            if field_num == 2:  # step
                step = val
        elif wire_type == 1:  # 64-bit (double)
            pos += 8
        elif wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            if length is None or pos + length > len(data):
                break
            if field_num == 5:  # summary
                loss = _decode_summary_loss(data[pos : pos + length])
            pos += length
        elif wire_type == 5:  # 32-bit (float)
            pos += 4
        else:
            break  # unknown wire type, stop

    return step, loss


def _decode_summary_loss(data: bytes) -> float | None:
    """Decode a tf.Summary message and extract loss value."""
    # Summary has repeated Value (field 1, length-delimited)
    # Value has tag (field 1, string) and simple_value (field 2, float32)
    loss = None
    pos = 0
    while pos < len(data):
        try:
            tag_byte = data[pos]
            field_num = tag_byte >> 3
            wire_type = tag_byte & 0x07
            pos += 1
        except IndexError:
            break

        if wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            if length is None or pos + length > len(data):
                break
            if field_num == 1:  # Value message
                tag_str, value = _decode_summary_value(data[pos : pos + length])
                if tag_str and value is not None and ("loss" in tag_str):
                    loss = value
            pos += length
        else:
            break
    return loss


def _decode_summary_value(data: bytes) -> tuple[str | None, float | None]:
    """Decode a tf.Summary.Value — extract tag string and simple_value float."""
    tag_str = None
    simple_value = None
    pos = 0
    while pos < len(data):
        try:
            tag_byte = data[pos]
            field_num = tag_byte >> 3
            wire_type = tag_byte & 0x07
            pos += 1
        except IndexError:
            break

        if wire_type == 2:  # length-delimited (string)
            length, pos = _decode_varint(data, pos)
            if length is None or pos + length > len(data):
                break
            if field_num == 1:  # tag
                try:
                    tag_str = data[pos : pos + length].decode("utf-8", errors="replace")
                except Exception:
                    pass
            pos += length
        elif wire_type == 5:  # 32-bit float
            if pos + 4 <= len(data) and field_num == 2:
                simple_value = struct.unpack("<f", data[pos : pos + 4])[0]
            pos += 4
        elif wire_type == 0:  # varint
            _, pos = _decode_varint(data, pos)
        elif wire_type == 1:  # 64-bit
            pos += 8
        else:
            break
    return tag_str, simple_value


def _decode_varint(data: bytes, pos: int) -> tuple[int | None, int]:
    """Decode a protobuf varint starting at pos. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            break
    return None, pos


def _progress_from_cmdline_and_checkpoints(parts: list[str], output_name: str | None) -> dict:
    """Estimate progress from cmdline args, checkpoint files, and TensorBoard events.

    Used when no log file is available (terminal-launched training).
    """
    result: dict = {}

    max_epochs_str = _extract_cmdline_arg(parts, "--max_train_epochs")
    if max_epochs_str and max_epochs_str.isdigit():
        result["max_epochs"] = int(max_epochs_str)

    output_dir = _extract_cmdline_arg(parts, "--output_dir")
    if output_dir and output_name and os.path.isdir(output_dir):
        # Checkpoint files look like: {output_name}-000001.safetensors
        prefix = f"{output_name}-"
        try:
            epoch_nums = []
            for f in os.listdir(output_dir):
                if f.startswith(prefix) and f.endswith(".safetensors"):
                    num_part = f[len(prefix):].replace(".safetensors", "")
                    if num_part.isdigit():
                        epoch_nums.append(int(num_part))
            if epoch_nums:
                result["current_epoch"] = max(epoch_nums)
        except OSError:
            pass

    # Parse TensorBoard events for step-level loss and step count
    logging_dir = _extract_cmdline_arg(parts, "--logging_dir")
    if logging_dir:
        logging_dir = os.path.expanduser(logging_dir)
        tfevents = _find_tfevents_file(logging_dir)
        if tfevents:
            tb_data = _parse_tfevents_tail(tfevents)
            if "current_step" in tb_data:
                result["current_step"] = tb_data["current_step"]
            if "current_loss" in tb_data:
                result["current_loss"] = tb_data["current_loss"]

    # Calculate pct_complete from epochs (step-level % needs total_steps which
    # we don't have from cmdline alone)
    current_epoch = result.get("current_epoch", 0)
    max_epochs = result.get("max_epochs", 0)
    if max_epochs > 0:
        result["pct_complete"] = round(current_epoch / max_epochs * 100, 2)

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
            # No log file (terminal-launched training).
            # Extract what we can from cmdline and checkpoint files.
            info.update(_progress_from_cmdline_and_checkpoints(parts, output_name))

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
