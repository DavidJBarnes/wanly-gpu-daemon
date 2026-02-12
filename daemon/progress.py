"""Accumulates timestamped log lines and PATCHes them to the API."""

import logging
from datetime import datetime
from uuid import UUID

from daemon.schemas import SegmentResult

logger = logging.getLogger(__name__)


class ProgressLog:
    def __init__(self, segment_id: UUID, queue):
        self._segment_id = segment_id
        self._queue = queue
        self._lines: list[str] = []

    @property
    def text(self) -> str:
        return "\n".join(self._lines)

    async def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._lines.append(f"[{ts}] {message}")
        logger.info(message)
        try:
            await self._queue.update_segment(
                self._segment_id,
                SegmentResult(status="processing", progress_log=self.text),
            )
        except Exception:
            logger.debug("Failed to PATCH progress_log (non-fatal)")
