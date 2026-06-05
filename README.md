# wanly-gpu-daemon

Worker daemon that claims segments from the wanly API and runs ComfyUI workflows for video generation.

## Overview

1. Registers with the wanly API on startup
2. Polls the API for pending segments
3. Downloads input images, executes ComfyUI workflows, and uploads output videos
4. Sends periodic heartbeats with GPU/stats to the API
5. Deregisters on shutdown (SIGINT/SIGTERM)

## Setup

### Prerequisites

- Python 3.11+
- ComfyUI running locally (or accessible via URL)

### Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `QUEUE_URL` | URL of the wanly API | `http://localhost:8001` |
| `QUEUE_API_KEY` | API key for wanly API auth | (required) |
| `FRIENDLY_NAME` | Display name for this worker | `gpu-worker-1` |
| `HEARTBEAT_INTERVAL` | Seconds between heartbeats | `30` |
| `COMFYUI_URL` | Local ComfyUI instance URL | `http://localhost:8188` |
| `COMFYUI_PATH` | Path to ComfyUI installation | (optional) |
| `POLL_INTERVAL` | Seconds between segment polls | `5` |

### Install and run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m daemon.main
```

## Key Files

| File | Purpose |
|------|---------|
| `daemon/main.py` | Entry point and polling loop |
| `daemon/executor.py` | Segment execution workflow |
| `daemon/comfyui_client.py` | ComfyUI WebSocket client |
| `daemon/workflow_builder.py` | Builds ComfyUI workflow JSON |
| `daemon/queue_client.py` | API client (segment claiming, worker registration, heartbeats) |
| `daemon/config.py` | Pydantic-settings configuration |

## Deployment

Runs as systemd service `wanly` on GPU servers:

```bash
sudo systemctl restart wanly
```

## Related Projects

- `wanly-api`: Backend API server (worker registry, segment queue, job management)
- `wanly-console`: Frontend React app
