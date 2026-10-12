"""What this worker is doing RIGHT NOW, for the heartbeat (wanly-gpu-daemon#189).

A busy worker must never look offline. Before this, the daemon synced the whole LoRA
catalogue BEFORE it registered, so a 650 MB LoRA published since the last sync took the box
off the Workers page for (size / ~0.5 MB/s) -- ten-plus minutes, twice on 2026-10-12 -- and the
only way to know why was `docker logs`. Now the daemon registers first, syncs in the
background, and says what it is doing on every heartbeat:

    {"phase": "syncing_loras", "detail": "Downloading Payton_v2_final (358 / 650 MB)",
     "file": "Payton_v2_final.safetensors", "done_bytes": ..., "total_bytes": ...,
     "rate_bps": ..., "eta_s": ..., "files_done": 0, "files_total": 1, "since": "..."}

The vocabulary is shared with wanly-gpu-docker's supervisor, which reports `switching_mode`
and `booting` the same way while it holds the row: booting, switching_mode, syncing_loras,
fetching_lora, ready. `detail` is always a full human sentence so a reader that does not know
a phase (an older console) still has something true to show.

A plain module-level dict, not a class: there is one daemon per process and one phase per
daemon, and the heartbeat only ever wants a copy.
"""
from __future__ import annotations

from datetime import datetime, timezone

READY = "ready"

_STATE: dict = {"phase": "booting", "detail": "Starting the render daemon"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_phase(phase: str, detail: str, **fields) -> None:
    """Enter a phase. Clears every progress field the previous phase carried."""
    global _STATE
    since = _STATE.get("since") if _STATE.get("phase") == phase else None
    _STATE = {"phase": phase, "detail": detail, "since": since or _now(), **fields}


def update(detail: str | None = None, **fields) -> None:
    """Progress within the current phase: keeps `since`, overwrites the fields given."""
    if detail is not None:
        _STATE["detail"] = detail
    _STATE.update(fields)


def ready() -> None:
    set_phase(READY, "Ready")


def current() -> str:
    return _STATE.get("phase") or READY


def snapshot() -> dict:
    return dict(_STATE)


def mb(n: float | int | None) -> str:
    return f"{(n or 0) / (1024 * 1024):.0f}"
