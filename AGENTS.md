# Wanly GPU Daemon

Worker daemon that claims segments from the API and runs ComfyUI workflows for video generation.

## Purpose

- Polls API for pending segments
- Downloads input images from S3
- Executes ComfyUI workflow via WebSocket
- Uploads output video and last frame
- Reports progress and status

## Key Files

| File | Purpose |
|------|---------|
| `daemon/main.py` | Entry point, polling loop |
| `daemon/executor.py` | Segment execution workflow |
| `daemon/comfyui_client.py` | ComfyUI WebSocket client |
| `daemon/workflow_builder.py` | Builds ComfyUI workflow JSON |
| `daemon/queue_client.py` | API client for claiming/uploading |
| `daemon/motion_extractor.py` | Motion keyword extraction |

## Workflow Execution

1. Claim segment via `POST /segments/{id}/claim`
2. Download start image and reference frames
3. Build ComfyUI workflow
4. Execute via WebSocket
5. Extract last frame
6. Upload output + last frame
7. Report motion keywords

## Quality Enhancement Features

### Multi-Frame Identity Anchoring
- Downloads reference frames from previous segments
- Creates CLIP Vision nodes for each reference
- Passes to PainterLongVideo for stronger identity

### Motion Propagation
- Extracts motion keywords from prompt
- Augments next segment's prompt with continuity hints
- e.g., "continuing to walk forward steadily"

## ComfyUI Workflow

Uses custom nodes for Wan 2.2 video generation:
- `WanImageToVideo` or `PainterLongVideo`
- `WanVideoToVideo` (optional)
- `FaceSwap` (optional)
- `RIFE` for interpolation

## Configuration

Environment variables (see `daemon/settings.py`):
- `COMFYUI_URL`: ComfyUI server URL
- `API_BASE_URL`: Wanly API URL
- `API_KEY`: Authentication key
- `CLIP_VISION_MODEL`: Model for identity anchoring

## Deployment

Runs as systemd service `wanly` on GPU servers:
- `2070.zero` (RTX 4070)
- `3090.zero` (RTX 3090)

```bash
sudo systemctl restart wanly
```

## Related Projects

- `wanly-api`: Backend API server
- `wanly-console`: Frontend React app
