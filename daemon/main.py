import asyncio
import logging
import os
import signal
import socket
import sys

from daemon.comfyui_client import ComfyUIClient
from daemon.config import settings
from daemon.executor import execute_segment
from daemon.model_validator import cleanup_partial_downloads, validate_models
from daemon.node_checker import check_and_install_nodes
from daemon.queue_client import QueueClient
from daemon.gpu_stats import get_gpu_stats
from daemon.resource_sync import sync_resources
from daemon.sd_scripts_monitor import get_status as get_sd_scripts_status
from daemon.a1111_monitor import get_status as get_a1111_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy httpx request-level logging (every /system_stats ping)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_ip_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


def _log_system_info(system_info: dict | None) -> None:
    """Log GPU/VRAM/RAM info from ComfyUI system stats."""
    if not system_info:
        logger.warning("Could not retrieve ComfyUI system info")
        return

    devices = system_info.get("devices", [])
    for dev in devices:
        name = dev.get("name", "unknown")
        vram_total = dev.get("vram_total", 0)
        vram_free = dev.get("vram_free", 0)
        vram_total_gb = vram_total / (1024**3) if vram_total else 0
        vram_free_gb = vram_free / (1024**3) if vram_free else 0
        logger.info("GPU: %s — VRAM: %.1f GB total, %.1f GB free", name, vram_total_gb, vram_free_gb)

    system = system_info.get("system", {})
    ram = system.get("ram", {})
    ram_total = ram.get("total", 0)
    ram_free = ram.get("free", 0)
    if ram_total:
        logger.info("RAM: %.1f GB total, %.1f GB free", ram_total / (1024**3), ram_free / (1024**3))


async def register_with_retry(client, *, friendly_name, hostname, ip_address, comfyui_running, shutdown_event):
    """Attempt to register with the API, retrying every 10s until success or shutdown.
    Returns (worker_id, friendly_name) — friendly_name may differ from config if renamed via console."""
    attempt = 0
    while not shutdown_event.is_set():
        attempt += 1
        try:
            worker_id, registered_name = await client.register(
                friendly_name=friendly_name,
                hostname=hostname,
                ip_address=ip_address,
                comfyui_running=comfyui_running,
            )
            logger.info("Registered as %s (id=%s)", registered_name, worker_id)
            return worker_id, registered_name
        except Exception as e:
            logger.error("Failed to register with API at %s (attempt %d): %s", settings.queue_url, attempt, e)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
    return None, None


async def heartbeat_loop(queue, comfyui, worker_id, friendly_name_ref, shutdown_event, drain_event):
    """Send heartbeats every heartbeat_interval seconds."""
    beat_count = 0
    last_busy_state = None
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                shutdown_event.wait(), timeout=settings.heartbeat_interval
            )
        except asyncio.TimeoutError:
            pass

        if shutdown_event.is_set():
            break

        comfyui_running = await comfyui.check_health()
        comfyui_busy = await comfyui.check_queue_busy() if comfyui_running else False

        # Collect GPU stats via nvidia-smi (works whether ComfyUI is running or not)
        gpu_stats = get_gpu_stats()

        # Collect sd-scripts training status
        sd_scripts_status = get_sd_scripts_status()
        sd_scripts_training = sd_scripts_status.get("sd_scripts_training", False)

        # Collect A1111 status
        a1111_status = get_a1111_status()

        # Worker is busy if either ComfyUI is processing or sd-scripts is training
        is_busy = comfyui_busy or sd_scripts_training

        try:
            data = await queue.heartbeat(worker_id, comfyui_running, gpu_stats, sd_scripts_status, a1111_status)
            beat_count += 1

            # Pick up renames from the registry
            new_name = data.get("friendly_name")
            if new_name and new_name != friendly_name_ref[0]:
                logger.info("Friendly name updated: %s → %s", friendly_name_ref[0], new_name)
                friendly_name_ref[0] = new_name

            # Check if API signals drain
            if data.get("status") == "draining" and not drain_event.is_set():
                logger.info("Drain requested — will stop after current work")
                drain_event.set()

            # Push status update when busy state changes
            if is_busy != last_busy_state:
                new_status = "online-busy" if is_busy else "online-idle"
                try:
                    await queue.update_status(worker_id, new_status)
                except Exception as e:
                    logger.error("Failed to update status to %s: %s", new_status, e)

                # Log what changed
                reasons = []
                if comfyui_busy:
                    reasons.append("ComfyUI busy")
                if sd_scripts_training:
                    info = sd_scripts_status.get("sd_scripts_training_info") or {}
                    name = info.get("output_name", "")
                    reasons.append(f"sd-scripts training{f' ({name})' if name else ''}")
                if reasons:
                    logger.info("Status: %s — %s", new_status, ", ".join(reasons))
                else:
                    logger.info("Status: %s", new_status)
                last_busy_state = is_busy
            elif beat_count % 5 == 0:
                logger.debug("Heartbeat OK (beat #%d)", beat_count)
        except Exception as e:
            logger.error("Heartbeat failed: %s: %s", type(e).__name__, e)


async def _ltx_healthy() -> bool:
    """Is ltx-engine up and answering? Never raises — a probe failure means 'not now'."""
    from daemon.ltx_client import LtxClient
    client = LtxClient()
    try:
        await client.health()
        return True
    except Exception:
        return False
    finally:
        await client.close()


async def job_poll_loop(queue, comfyui, worker_id, friendly_name_ref, shutdown_event, executing_event, drain_event):
    """Poll the queue for segments and execute them one at a time."""
    poll_count = 0
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                shutdown_event.wait(), timeout=settings.poll_interval
            )
        except asyncio.TimeoutError:
            pass

        if shutdown_event.is_set():
            break

        # If draining and not executing, trigger shutdown
        if drain_event.is_set():
            logger.info("Drain active and no work in progress — shutting down")
            shutdown_event.set()
            break

        # Don't claim work if ComfyUI isn't running
        if not await comfyui.check_health():
            if poll_count == 0 or poll_count % 60 == 0:
                logger.warning("ComfyUI offline — skipping poll")
            poll_count += 1
            continue

        # Same rule for ltx-engine: claiming a segment this worker cannot render burns it.
        # The segment would fail, and a failed segment is not free — it needs a human to
        # notice and retry. Better to leave it queued for a worker that can take it.
        if settings.engine == "ltx" and not await _ltx_healthy():
            if poll_count == 0 or poll_count % 60 == 0:
                logger.warning("ltx-engine offline — skipping poll")
            poll_count += 1
            continue

        # Don't claim work if ComfyUI is already processing (e.g. leftover from previous daemon run)
        if await comfyui.check_queue_busy():
            if poll_count == 0 or poll_count % 60 == 0:
                logger.info("ComfyUI queue busy — waiting for current job to finish")
            poll_count += 1
            continue

        try:
            segment = await queue.claim_next(worker_id, friendly_name_ref[0], kind="gpu")
        except Exception as e:
            logger.error("Poll failed: %s: %s", type(e).__name__, e)
            continue

        poll_count += 1
        if segment is None:
            if poll_count == 1 or poll_count % 60 == 0:
                logger.info("Waiting for work... (polled %d times)", poll_count)
            continue

        poll_count = 0

        executing_event.set()
        try:
            await queue.update_status(worker_id, "online-busy")
        except Exception as e:
            logger.error("Failed to update status to busy: %s", e)

        try:
            await execute_segment(segment, comfyui, queue)
        except Exception as e:
            logger.exception("Unexpected error executing segment %s", segment.id)
            try:
                from daemon.schemas import SegmentResult
                await queue.update_segment(
                    segment.id,
                    SegmentResult(status="failed", error_message=f"{type(e).__name__}: {e}"[:2000]),
                )
            except Exception as report_err:
                logger.error("Failed to report segment failure: %s", report_err)
        finally:
            # Log VRAM usage after each segment to spot memory leaks
            try:
                info = await comfyui.get_system_info()
                if info:
                    devices = info.get("devices", [])
                    if devices:
                        d = devices[0]
                        vram_used = d.get("vram_total", 0) - d.get("vram_free", 0)
                        vram_total = d.get("vram_total", 0)
                        if vram_total > 0:
                            logger.info("VRAM: %.0f / %.0f MiB (%.0f%%)",
                                vram_used / 1048576, vram_total / 1048576,
                                vram_used / vram_total * 100)
            except Exception:
                pass
            executing_event.clear()
            # If draining, don't go back to idle — shut down
            if drain_event.is_set():
                logger.info("Drain active — segment finished, shutting down")
                shutdown_event.set()
            else:
                # Only go idle if sd-scripts isn't training (heartbeat loop handles ongoing busy)
                sd_status = get_sd_scripts_status()
                if not sd_status.get("sd_scripts_training"):
                    try:
                        await queue.update_status(worker_id, "online-idle")
                    except Exception as e:
                        logger.error("Failed to update status to idle: %s: %s", type(e).__name__, e)


async def hologram_poll_loop(queue, comfyui, worker_id, friendly_name_ref, shutdown_event):
    """Poll for CPU-only AR hologram work and run it CONCURRENTLY with GPU generation.

    Holograms never touch ComfyUI/the GPU, so this loop skips the ComfyUI-busy gate and runs
    alongside job_poll_loop. _execute_ar_hologram offloads its heavy numpy/RVM/ffmpeg work to a
    thread, keeping the event loop free for the GPU loop + heartbeat.
    """
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=settings.poll_interval)
        except asyncio.TimeoutError:
            pass
        if shutdown_event.is_set():
            break
        try:
            segment = await queue.claim_next(worker_id, friendly_name_ref[0], kind="hologram")
        except Exception as e:
            logger.error("Hologram poll failed: %s: %s", type(e).__name__, e)
            continue
        if segment is None:
            continue
        try:
            await execute_segment(segment, comfyui, queue)
        except Exception as e:
            logger.exception("Unexpected error executing hologram segment %s", segment.id)
            try:
                from daemon.schemas import SegmentResult
                await queue.update_segment(
                    segment.id,
                    SegmentResult(status="failed", error_message=f"{type(e).__name__}: {e}"[:2000]),
                )
            except Exception as report_err:
                logger.error("Failed to report hologram failure: %s", report_err)


async def _terminate_runpod_pod() -> bool:
    """Terminate this pod. Returns True if RunPod accepted the termination.

    Terminate, not stop. podStop leaves the pod EXITED and still billing for its disk; the
    operator then has to notice and clean it up by hand. A drained worker should cost nothing.

    Missing credentials are logged loudly rather than ignored: this runs at the end of a
    drain, and if the pod does not actually stop the container respawns, the daemon
    re-registers and the worker goes straight back to claiming work. Returning silently
    made a drain look like it had simply been ignored, with nothing in the log to say why.
    """
    import httpx as _httpx

    pod_id = settings.runpod_pod_id
    api_key = settings.runpod_api_key
    if not pod_id or not api_key:
        logger.warning(
            "Drained, but cannot terminate the pod: RUNPOD_POD_ID=%s, RUNPOD_API_KEY=%s. "
            "The container will respawn and this worker will start claiming work again. "
            "Set both as pod environment variables to make drain actually stop the pod.",
            "set" if pod_id else "MISSING",
            "set" if api_key else "MISSING",
        )
        return False

    logger.info("Terminating RunPod pod %s ...", pod_id)

    # GraphQL podTerminate, not REST DELETE /v1/pods/{id}.
    #
    # A pod cannot delete ITSELF through the REST API — the identical key, on the identical
    # endpoint, returns 403 from inside the container and 204 from outside it. This was found in
    # production: a drain finished its segment, tried to terminate, got 403, and parked.
    #
    # The GraphQL mutation has no such restriction, which is why the earlier podStop worked from
    # inside. podTerminate gives the terminate semantics we want; podStop only stopped the pod
    # and left it billing for disk.
    query = f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'
    try:
        async with _httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"https://api.runpod.io/graphql?api_key={api_key}",
                json={"query": query},
            )
            resp.raise_for_status()
            body = resp.json()
            errors = body.get("errors") or []
            # POD_NOT_FOUND means it is already gone, which is the state we wanted.
            if errors and all(
                (e.get("extensions") or {}).get("code") == "POD_NOT_FOUND" for e in errors
            ):
                logger.info("RunPod pod %s already gone", pod_id)
                return True
            if errors:
                logger.error("RunPod refused to terminate %s: %s", pod_id, errors)
                return False
            logger.info("RunPod pod %s terminated", pod_id)
            return True
    except Exception as e:
        # A revoked API key lands here as a 401. That is not hypothetical: the key baked into
        # the pod template was revoked, every drain 401'd, and because the failure was silent
        # the worker simply carried on taking jobs. See wanly-console#286.
        logger.error("Failed to terminate RunPod pod: %s", e)
    return False


def kill_stale_daemons():
    """Kill any existing daemon.main processes from previous runs."""
    my_pid = os.getpid()
    killed = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace")
            if "daemon.main" in cmdline and "python" in cmdline.lower():
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
        except (OSError, PermissionError):
            continue
    if killed:
        logger.info("Killed %d stale daemon process(es): %s", len(killed), killed)


async def run():
    kill_stale_daemons()
    comfyui = ComfyUIClient()
    queue = QueueClient()
    shutdown_event = asyncio.Event()
    executing_event = asyncio.Event()
    drain_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    hostname = socket.gethostname()
    ip_address = get_ip_address()
    comfyui_running = await comfyui.check_health()

    logger.info("=== Wanly GPU Daemon ===")
    logger.info("Python %s | Worker: %s | ComfyUI: %s | API: %s",
        sys.version.split()[0],
        settings.friendly_name,
        "running" if comfyui_running else "NOT RUNNING",
        settings.queue_url,
    )
    logger.info("Models: clip=%s vae=%s unet_high=%s unet_low=%s",
        settings.clip_model, settings.vae_model,
        settings.unet_high_model, settings.unet_low_model,
    )
    logger.info("LightX2V strengths: high=%.1f low=%.1f",
        settings.lightx2v_strength_high, settings.lightx2v_strength_low,
    )

    sd_status = get_sd_scripts_status()
    logger.info("sd-scripts: installed=%s, training=%s%s",
        sd_status["sd_scripts_installed"],
        sd_status["sd_scripts_training"],
        f" ({sd_status['sd_scripts_training_info']['output_name']})" if sd_status.get("sd_scripts_training_info") else "",
    )

    a1111_stat = get_a1111_status()
    logger.info("A1111: installed=%s, running=%s",
        a1111_stat["a1111_installed"],
        a1111_stat["a1111_running"],
    )

    # Clear any orphaned ComfyUI queue items from previous daemon runs
    if comfyui_running:
        if await comfyui.clear_queue():
            logger.info("Cleared ComfyUI queue")
        else:
            logger.warning("Failed to clear ComfyUI queue")

    # Check and install required ComfyUI custom nodes
    nodes_ok = await check_and_install_nodes(comfyui)
    if not nodes_ok:
        logger.error("Required custom nodes are missing or could not be installed. Exiting.")
        await comfyui.close()
        await queue.close()
        return

    # Pre-flight: sync required resources (model weights for custom nodes)
    resources_ok = await sync_resources(queue)
    if not resources_ok:
        logger.error("Resource sync failed. Exiting.")
        await comfyui.close()
        await queue.close()
        return

    # Pre-flight: clean up partial downloads
    cleaned = cleanup_partial_downloads(settings.comfyui_path)
    if cleaned:
        logger.info("Cleaned %d partial download(s)", cleaned)

    # Which engine this worker drives, stated before anything else uses it. Per AGENTS.md the
    # expensive failure here is silence: a worker on the wrong engine still produces plausible
    # output that is not comparable to anything.
    logger.info("ENGINE=%s", settings.engine)

    # Pre-flight: validate all required models.
    #
    # MODEL_CHECKS describes the WAN 2.2 model set, so it is meaningless for an LTX worker and
    # would refuse a perfectly good one. ltx-engine validates its own models at startup, and
    # the LTX health gate below will not let this worker claim until it is up — so an LTX
    # worker is checked, just by the service that knows what to check for.
    # Replacing MODEL_CHECKS with the LTX set is wanly-gpu-docker#41.
    if settings.engine == "ltx":
        logger.info("Skipping WAN model validation — ltx-engine validates its own models")
    else:
        models_ok = await validate_models(comfyui)
        if not models_ok:
            logger.error("Model validation failed. Exiting.")
            await comfyui.close()
            await queue.close()
            return

    # Log GPU/VRAM info
    system_info = await comfyui.get_system_info()
    _log_system_info(system_info)

    worker_id, registered_name = await register_with_retry(
        queue,
        friendly_name=settings.friendly_name,
        hostname=hostname,
        ip_address=ip_address,
        comfyui_running=comfyui_running,
        shutdown_event=shutdown_event,
    )

    if worker_id is None:
        logger.info("Shutdown requested before registration completed")
        await queue.close()
        await comfyui.close()
        return

    # Mutable ref so heartbeat can update the name and poll loop sees it
    friendly_name_ref = [registered_name]

    try:
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(queue, comfyui, worker_id, friendly_name_ref, shutdown_event, drain_event)
        )
        job_task = asyncio.create_task(
            job_poll_loop(queue, comfyui, worker_id, friendly_name_ref, shutdown_event, executing_event, drain_event)
        )
        holo_task = asyncio.create_task(
            hologram_poll_loop(queue, comfyui, worker_id, friendly_name_ref, shutdown_event)
        )

        await asyncio.gather(heartbeat_task, job_task, holo_task)
    finally:
        # Graceful shutdown: wait for current segment if executing
        segment_abandoned = False
        if executing_event.is_set():
            # executing_event wraps the WHOLE of execute_segment, including the [7/7] upload —
            # so clearing it means the segment is finished and reported, not merely that the GPU
            # went quiet. That distinction matters: decode, RIFE and stitching all
            # identity scoring all run after the GPU idles, for 47s to over 2 minutes.
            logger.info(
                "Waiting for current segment to finish (up to %ds)...",
                settings.drain_wait_seconds,
            )
            try:
                await asyncio.wait_for(
                    asyncio.create_task(_wait_for_clear(executing_event)),
                    timeout=settings.drain_wait_seconds,
                )
            except asyncio.TimeoutError:
                segment_abandoned = True
                logger.error(
                    "Timed out after %ds waiting for the segment to finish. It is still "
                    "running. NOT terminating the pod — that would destroy work in progress.",
                    settings.drain_wait_seconds,
                )

        # Stop the pod BEFORE deregistering, not after.
        #
        # Deregistering deletes the worker row, and that row is where a pending drain lives.
        # wanly-api#133 added reregistered_drain_state() precisely so a re-register cannot
        # cancel a drain — but deleting the row first throws away the state that protection
        # depends on. So if the pod does not actually stop, the container respawns, the daemon
        # registers fresh as online-idle, and goes straight back to claiming work. The drain is
        # silently undone, which is indistinguishable from the drain never having been requested.
        terminated = False
        on_runpod = bool(settings.runpod_pod_id)
        if drain_event.is_set() and on_runpod and segment_abandoned:
            # Park rather than terminate. The segment is still running inside this container;
            # destroying the pod now loses it outright.
            logger.error(
                "DRAIN INCOMPLETE: segment still running past the drain timeout. Staying "
                "registered as 'draining' and NOT terminating. Raise DRAIN_WAIT_SECONDS if "
                "segments legitimately take this long."
            )
            await queue.close()
            await comfyui.close()
            while True:
                await asyncio.sleep(3600)

        if drain_event.is_set() and on_runpod:
            terminated = await _terminate_runpod_pod()
            if not terminated:
                # Park. Do not deregister, do not exit.
                #
                # Exiting would respawn the container into the same failed drain, in a loop.
                # Staying registered as `draining` means the console keeps showing a Draining
                # worker that is visibly not finishing — which is the honest picture — and the
                # job poll loop has already stopped, so it claims nothing while parked.
                logger.error(
                    "DRAIN INCOMPLETE: the pod is still running and could not be terminated. "
                    "Staying registered as 'draining' so this worker does NOT resume claiming "
                    "work. Terminate the pod from the RunPod console, and check RUNPOD_API_KEY "
                    "— a revoked key returns 401 here."
                )
                await queue.close()
                await comfyui.close()
                # Sleep rather than return: returning ends the container's main process and
                # RunPod restarts it.
                while True:
                    await asyncio.sleep(3600)

        logger.info("Shutting down, deregistering...")
        try:
            await queue.deregister(worker_id)
            logger.info("Deregistered successfully")
        except Exception as e:
            logger.error("Failed to deregister: %s", e)

        await queue.close()
        await comfyui.close()


async def _wait_for_clear(event: asyncio.Event):
    """Wait until the event is cleared."""
    while event.is_set():
        await asyncio.sleep(1)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
