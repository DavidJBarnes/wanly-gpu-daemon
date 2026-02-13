import asyncio
import json
import logging
import uuid
from typing import Any, Optional

import httpx
import websockets

from daemon.config import settings

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 1800  # 30 minutes


class ComfyUIExecutionError(Exception):
    """Raised when ComfyUI reports an execution error."""

    def __init__(self, message: str, node_id: str = "", node_type: str = "", traceback: str = ""):
        self.node_id = node_id
        self.node_type = node_type
        self.traceback = traceback
        super().__init__(message)


class ComfyUIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or settings.comfyui_url).rstrip("/")
        self.api_key = api_key or settings.comfyui_api_key
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=30, headers=headers)

    async def check_health(self) -> bool:
        """Check if ComfyUI is running via GET /system_stats."""
        try:
            resp = await self.http.get("/system_stats")
            return resp.status_code == 200
        except Exception:
            return False

    async def check_queue_busy(self) -> bool:
        """Check if ComfyUI has an active prompt via GET /queue."""
        try:
            resp = await self.http.get("/queue")
            if resp.status_code != 200:
                return False
            data = resp.json()
            running = data.get("queue_running", [])
            return len(running) > 0
        except Exception:
            return False

    async def get_system_info(self) -> dict | None:
        """Get system stats from ComfyUI (VRAM, RAM). Best-effort, returns None on failure."""
        try:
            resp = await self.http.get("/system_stats")
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data
        except Exception:
            return None

    async def upload_image(self, data: bytes, filename: str) -> str:
        """Upload an image to ComfyUI's input folder. Returns the stored filename."""
        resp = await self.http.post(
            "/upload/image",
            files={"image": (filename, data, "image/png")},
            data={"overwrite": "true"},
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("name", filename)

    async def submit_workflow(self, workflow: dict) -> tuple[str, str]:
        """Submit a workflow to ComfyUI. Returns (prompt_id, client_id)."""
        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        resp = await self.http.post("/prompt", json=payload)
        if resp.status_code != 200:
            logger.error("ComfyUI rejected workflow (%d): %s", resp.status_code, resp.text)
            resp.raise_for_status()
        result = resp.json()
        prompt_id = result["prompt_id"]
        logger.info("Submitted workflow, prompt_id=%s", prompt_id)
        return prompt_id, client_id

    async def monitor_execution(self, prompt_id: str, client_id: str) -> dict[str, Any]:
        """Monitor workflow execution via WebSocket with a timeout.

        Returns output data from 'executed' messages.
        Raises ComfyUIExecutionError on timeout or execution failure.
        """
        try:
            return await asyncio.wait_for(
                self._monitor_ws(prompt_id, client_id),
                timeout=EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise ComfyUIExecutionError(
                f"Execution timed out after {EXECUTION_TIMEOUT}s"
            )

    async def _monitor_ws(self, prompt_id: str, client_id: str) -> dict[str, Any]:
        """Inner WebSocket monitoring loop."""
        ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws?clientId={client_id}"
        if self.api_key:
            ws_url += f"&token={self.api_key}"
        outputs: dict[str, Any] = {}
        current_node = None
        last_progress_pct = -1

        async with websockets.connect(ws_url) as ws:
            async for raw_message in ws:
                if isinstance(raw_message, bytes):
                    continue  # Skip binary preview frames

                message = json.loads(raw_message)
                msg_type = message.get("type")
                msg_data = message.get("data", {})

                # Filter to our prompt only
                if msg_data.get("prompt_id") and msg_data["prompt_id"] != prompt_id:
                    continue

                if msg_type == "execution_start":
                    logger.info("       ComfyUI execution started")

                elif msg_type == "executing":
                    node = msg_data.get("node")
                    if node is None:
                        logger.info("       All nodes executed")
                    else:
                        current_node = node
                        last_progress_pct = -1

                elif msg_type == "progress":
                    value = msg_data.get("value", 0)
                    max_val = msg_data.get("max", 0)
                    if max_val > 0:
                        pct = int(value / max_val * 100)
                        # Only log at 0%, 25%, 50%, 75%, 100% to reduce noise
                        for threshold in (0, 25, 50, 75, 100):
                            if pct >= threshold > last_progress_pct:
                                node_label = f" (node {current_node})" if current_node else ""
                                logger.info("       Step %d/%d (%d%%)%s", value, max_val, pct, node_label)
                                last_progress_pct = threshold
                                break

                elif msg_type == "executed":
                    node_id = msg_data.get("node")
                    output = msg_data.get("output", {})
                    if output:
                        outputs[node_id] = output

                elif msg_type == "execution_success":
                    logger.info("       ComfyUI execution complete")
                    break

                elif msg_type == "execution_error":
                    error_msg = msg_data.get("exception_message", "Unknown error")
                    node_id = msg_data.get("node_id", "")
                    node_type = msg_data.get("node_type", "")
                    tb = "\n".join(msg_data.get("traceback", []))
                    raise ComfyUIExecutionError(
                        error_msg,
                        node_id=node_id,
                        node_type=node_type,
                        traceback=tb,
                    )

        return outputs

    async def get_history(self, prompt_id: str, retries: int = 5, delay: float = 1.0) -> dict:
        """Get execution history for a prompt. Retries because outputs may lag after execution_success."""
        for attempt in range(retries):
            resp = await self.http.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            data = resp.json()
            history = data.get(prompt_id)
            if history and history.get("outputs"):
                return history
            if attempt < retries - 1:
                logger.debug("History not ready, retry %d/%d", attempt + 1, retries)
                await asyncio.sleep(delay)
        return data.get(prompt_id, {})

    async def download_output(self, filename: str, subfolder: str = "", output_type: str = "output") -> bytes:
        """Download an output file from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": output_type}
        resp = await self.http.get("/view", params=params, timeout=120)
        resp.raise_for_status()
        return resp.content

    def find_video_output(self, history: dict) -> Optional[dict]:
        """Find the video file info from history outputs.

        Searches for VHS_VideoCombine (node 186) output, which contains gifs/videos.
        Returns dict with 'filename', 'subfolder', 'type' or None.
        """
        outputs = history.get("outputs", {})

        # Check node 186 (VHS_VideoCombine) first
        node_186 = outputs.get("186", {})
        gifs = node_186.get("gifs", [])
        if gifs:
            return gifs[0]

        # Fallback: search all outputs for video files
        for node_id, output in outputs.items():
            for gif in output.get("gifs", []):
                if gif.get("filename", "").endswith(".mp4"):
                    return gif

        return None

    async def close(self):
        await self.http.aclose()
