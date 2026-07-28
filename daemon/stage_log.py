"""Structured JSON stage logging with correlation IDs and VRAM peaks.

The daemon's baseline logging is human-readable text (see ``main.py`` basicConfig).
Engine stages additionally emit one JSON object per event through the same logger,
so pod logs stay greppable and can be aggregated by correlation ID without changing
global log configuration.

VRAM caveat: the daemon is a *separate process* from ComfyUI, so
``torch.cuda.max_memory_allocated()`` would report the daemon's own (empty)
allocator, not the one doing the work. We therefore sample device-level usage via
``nvidia-smi`` on a background thread and report the peak observed across the
stage. That figure includes anything else resident on the card, which is the
number that actually matters for "does this fit in 24 GB".
"""

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from daemon.gpu_stats import get_gpu_stats

# Seconds between VRAM samples while a stage is running. nvidia-smi costs ~10ms,
# so this is cheap relative to a multi-minute diffusion stage.
VRAM_POLL_INTERVAL = 0.5


def log_event(
    logger: logging.Logger,
    event: str,
    correlation_id: str,
    level: int = logging.INFO,
    **fields: Any,
) -> dict[str, Any]:
    """Emit one structured JSON log record and return the payload.

    The payload is returned so callers (and tests) can assert on it without
    having to parse the emitted log line back out.
    """
    payload: dict[str, Any] = {"event": event, "correlation_id": correlation_id}
    payload.update(fields)
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))
    return payload


class _VramSampler:
    """Polls device VRAM on a background thread, retaining the peak.

    Degrades to ``None`` peaks when nvidia-smi is unavailable (CPU-only dev boxes,
    unit tests) rather than failing the stage it is measuring.
    """

    def __init__(self, interval: float = VRAM_POLL_INTERVAL) -> None:
        self.interval = interval
        self.peak_mb: int | None = None
        self.total_mb: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        stats = get_gpu_stats()
        if stats is None:
            return
        used = int(stats["vram_used_mb"])
        self.total_mb = int(stats["vram_total_mb"])
        if self.peak_mb is None or used > self.peak_mb:
            self.peak_mb = used

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._sample()  # baseline, so very short stages still report a figure
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 4)
        self._sample()  # final sample catches a late peak


@contextmanager
def stage(
    logger: logging.Logger,
    name: str,
    correlation_id: str,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage, logging start/end (or failure) as JSON.

    Yields a mutable dict; anything the caller puts in it is merged into the
    terminating log record, which lets a stage report values it only learns while
    running (node counts, frame counts, ...).
    """
    extra: dict[str, Any] = {}
    log_event(logger, f"{name}.start", correlation_id, **fields)
    sampler = _VramSampler()
    sampler.start()
    started = time.monotonic()
    try:
        yield extra
    except Exception as exc:
        sampler.stop()
        log_event(
            logger,
            f"{name}.failed",
            correlation_id,
            level=logging.ERROR,
            duration_sec=round(time.monotonic() - started, 3),
            vram_peak_mb=sampler.peak_mb,
            vram_total_mb=sampler.total_mb,
            error=str(exc),
            error_type=type(exc).__name__,
            **{**fields, **extra},
        )
        raise
    sampler.stop()
    log_event(
        logger,
        f"{name}.end",
        correlation_id,
        duration_sec=round(time.monotonic() - started, 3),
        vram_peak_mb=sampler.peak_mb,
        vram_total_mb=sampler.total_mb,
        **{**fields, **extra},
    )
