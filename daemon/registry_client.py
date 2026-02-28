import uuid

import httpx

from daemon.config import settings


class RegistryClient:
    def __init__(self):
        self.base_url = settings.registry_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=10)

    async def register(
        self,
        friendly_name: str,
        hostname: str,
        ip_address: str,
        comfyui_running: bool,
    ) -> tuple[uuid.UUID, str]:
        resp = await self.client.post(
            "/workers",
            json={
                "friendly_name": friendly_name,
                "hostname": hostname,
                "ip_address": ip_address,
                "comfyui_running": comfyui_running,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return uuid.UUID(data["id"]), data["friendly_name"]

    async def heartbeat(
        self,
        worker_id: uuid.UUID,
        comfyui_running: bool,
        gpu_stats: dict | None = None,
    ) -> dict:
        """Send heartbeat. Returns full worker data including current friendly_name."""
        payload: dict = {"comfyui_running": comfyui_running}
        if gpu_stats is not None:
            payload["gpu_stats"] = gpu_stats
        resp = await self.client.post(
            f"/workers/{worker_id}/heartbeat",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def update_status(self, worker_id: uuid.UUID, status: str):
        resp = await self.client.patch(
            f"/workers/{worker_id}/status",
            json={"status": status},
        )
        resp.raise_for_status()

    async def deregister(self, worker_id: uuid.UUID):
        resp = await self.client.delete(f"/workers/{worker_id}")
        resp.raise_for_status()

    async def close(self):
        await self.client.aclose()
