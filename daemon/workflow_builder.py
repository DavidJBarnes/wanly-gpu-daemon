"""Build ComfyUI API workflows from segment parameters.

Ports the Wan2.2 I2V workflow template from v1 (wan22-video-generator) with
dynamic node injection for LoRAs, face swap, RIFE interpolation, and video output.
"""

import copy
import logging
import math
from typing import Any

from daemon.config import settings
from daemon.schemas import SegmentClaim
from daemon.stage_log import log_event

logger = logging.getLogger(__name__)

# Generation is always at 15fps; RIFE interpolation brings it to target fps.
GENERATION_FPS = 15

# Node IDs for dynamically added user LoRA pairs (up to 3).
LORA_NODE_IDS = {
    "high": ["118", "120", "122"],
    "low": ["119", "121", "123"],
}

# Base Wan2.2 14B Image-to-Video workflow in ComfyUI API format.
# Dynamic nodes (RIFE, VHS_VideoCombine, user LoRAs) are added at runtime.
WAN_I2V_API_WORKFLOW: dict[str, Any] = {
    "84": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
    },
    "85": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "control_after_generate": "fixed",
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 2,
            "end_at_step": 4,
            "return_with_leftover_noise": "disable",
            "model": ["103", 0],
            "positive": ["98", 0],
            "negative": ["98", 1],
            "latent_image": ["86", 0],
        },
    },
    "86": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 138073435077572,
            "control_after_generate": "randomize",
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 2,
            "return_with_leftover_noise": "enable",
            "model": ["104", 0],
            "positive": ["98", 0],
            "negative": ["98", 1],
            "latent_image": ["98", 2],
        },
    },
    "87": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["85", 0],
            "vae": ["90", 0],
        },
    },
    "89": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0cJPEG\u538b\u7f29\u6b8b\u7559\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70",
            "clip": ["84", 0],
        },
    },
    "90": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors",
        },
    },
    "93": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["84", 0],
        },
    },
    "95": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "wan2.2_i2v_high_noise_14B_fp16.safetensors",
            "weight_dtype": "default",
        },
    },
    "96": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "wan2.2_i2v_low_noise_14B_fp16.safetensors",
            "weight_dtype": "default",
        },
    },
    "97": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "input.jpg",
            "upload": "image",
        },
    },
    "98": {
        "class_type": "WanImageToVideo",
        "inputs": {
            "width": 640,
            "height": 640,
            "length": 81,
            "batch_size": 1,
            "positive": ["93", 0],
            "negative": ["89", 0],
            "vae": ["90", 0],
            "start_image": ["97", 0],
        },
    },
    "101": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            "strength_model": 1.0,
            "model": ["95", 0],
        },
    },
    "102": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            "strength_model": 1.0,
            "model": ["96", 0],
        },
    },
    "103": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
            "shift": 5.0,
            "model": ["102", 0],
        },
    },
    "104": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
            "shift": 5.0,
            "model": ["101", 0],
        },
    },
}


def _calculate_generation_params(target_fps: int, duration_sec: float, speed: float = 1.0) -> dict[str, Any]:
    speed = max(speed, 0.25)
    # More speed = more WAN frames = more motion packed into the same duration.
    wan_frames = max(math.ceil(duration_sec * GENERATION_FPS * speed), 5)
    # RIFE smoothing based on target fps (2x for 30fps, 4x for 60fps).
    rife_multiplier = max(target_fps // GENERATION_FPS, 1)
    total_frames = wan_frames * rife_multiplier
    # Output fps adjusts so all frames fit into the requested duration.
    output_fps = round(total_frames / duration_sec)
    return {
        "wan_frames": wan_frames,
        "rife_multiplier": rife_multiplier,
        "output_fps": output_fps,
    }


def _add_user_loras(workflow: dict, loras: list[dict]) -> None:
    """Add user LoRA nodes and rewire the lightx2v chain."""
    loras = [item for item in loras if item.get("high_file") or item.get("low_file")]
    if not loras:
        return

    last_high_node = "95"  # UNET high
    last_low_node = "96"  # UNET low

    for i, lora in enumerate(loras[:3]):
        high_file = lora.get("high_file")
        high_weight = float(lora.get("high_weight", 1.0))
        low_file = lora.get("low_file")
        low_weight = float(lora.get("low_weight", 1.0))

        high_node_id = LORA_NODE_IDS["high"][i]
        low_node_id = LORA_NODE_IDS["low"][i]

        if high_file:
            workflow[high_node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": high_file,
                    "strength_model": high_weight,
                    "model": [last_high_node, 0],
                },
                "_meta": {"title": f"User LoRA {i + 1} High"},
            }
            last_high_node = high_node_id
            logger.info("Added LoRA %d high: %s (weight=%.2f)", i + 1, high_file, high_weight)

        if low_file:
            workflow[low_node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": low_file,
                    "strength_model": low_weight,
                    "model": [last_low_node, 0],
                },
                "_meta": {"title": f"User LoRA {i + 1} Low"},
            }
            last_low_node = low_node_id
            logger.info("Added LoRA %d low: %s (weight=%.2f)", i + 1, low_file, low_weight)

    # Rewire lightx2v to chain after last user LoRA
    workflow["101"]["inputs"]["model"] = [last_high_node, 0]
    workflow["102"]["inputs"]["model"] = [last_low_node, 0]


def build_workflow(
    segment: SegmentClaim,
    start_image_filename: str | None = None,
    initial_reference_image_filename: str | None = None,
    reference_frame_filenames: list[str] | None = None,
    previous_motion_magnitude: float | None = None,
) -> dict:
    """Build a complete ComfyUI workflow from segment parameters.

    Args:
        segment: The claimed segment with all generation parameters.
        start_image_filename: The ComfyUI-local filename of the start image
            (already uploaded). If None, the LoadImage node (97) is removed
            for text-to-video generation.
        initial_reference_image_filename: Accepted and ignored. It fed a
            PainterLongVideo swap that was inert on Wan 2.2 i2v and has been
            removed; the parameter stays so the executor's call site and any
            queued segments carrying an identity_reference_image keep working.
        previous_motion_magnitude: Accepted and ignored, for the same reason.
            It fed motion_amplitude, which PainterLongVideo only applied on a
            branch this graph never took.

    Note on what was removed (see wanly-gpu-daemon#124): the swap replaced node
    98 with PainterLongVideo whenever a segment had both an initial reference
    and a start image. On Wan 2.2 i2v every one of its distinguishing inputs was
    dead -- previous_video was wired to a single-frame LoadImage so the motion
    reference resolved to None, reference_latents needs a ref_conv weight this
    checkpoint does not have, clip_vision_output needs an img_emb it does not
    have, and motion_amplitude is only read on the previous-video-only branch.
    With start and end connected the node does exactly what stock
    WanFirstLastFrameToVideo does. It was WanImageToVideo with a wasted VAE
    encode and a wasted CLIP Vision load.
    """
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)
    
    workflow = copy.deepcopy(WAN_I2V_API_WORKFLOW)

    # Inject model filenames from config
    workflow["84"]["inputs"]["clip_name"] = settings.clip_model
    workflow["90"]["inputs"]["vae_name"] = settings.vae_model
    workflow["95"]["inputs"]["unet_name"] = settings.unet_high_model
    workflow["96"]["inputs"]["unet_name"] = settings.unet_low_model
    workflow["101"]["inputs"]["lora_name"] = settings.lightx2v_lora_high
    workflow["101"]["inputs"]["strength_model"] = segment.lightx2v_strength_high if segment.lightx2v_strength_high is not None else settings.lightx2v_strength_high
    workflow["102"]["inputs"]["lora_name"] = settings.lightx2v_lora_low
    workflow["102"]["inputs"]["strength_model"] = segment.lightx2v_strength_low if segment.lightx2v_strength_low is not None else settings.lightx2v_strength_low

    # CFG values for KSampler nodes
    workflow["86"]["inputs"]["cfg"] = segment.cfg_high if segment.cfg_high is not None else settings.cfg_high
    workflow["85"]["inputs"]["cfg"] = segment.cfg_low if segment.cfg_low is not None else settings.cfg_low

    # Sampler step schedule (per-job override). Both KSamplers share the total length; the
    # boundary splits it: high (86) runs [0, high_noise_steps], low (85) runs [high_noise_steps, total].
    steps_total = segment.steps_total if segment.steps_total is not None else settings.steps_total
    high_noise_steps = segment.high_noise_steps if segment.high_noise_steps is not None else settings.high_noise_steps
    workflow["86"]["inputs"]["steps"] = steps_total
    workflow["86"]["inputs"]["start_at_step"] = 0
    workflow["86"]["inputs"]["end_at_step"] = high_noise_steps
    workflow["85"]["inputs"]["steps"] = steps_total
    workflow["85"]["inputs"]["start_at_step"] = high_noise_steps
    workflow["85"]["inputs"]["end_at_step"] = steps_total

    # Flow-matching schedule shift (per-job). One value on BOTH experts (they share the noise
    # trajectory; mismatched shifts break the high->low handoff sigma). Higher = more high-noise
    # steps = more motion; raise when de-distilling to cure the "frozen" look.
    flow_shift = segment.flow_shift if segment.flow_shift is not None else settings.flow_shift
    workflow["104"]["inputs"]["shift"] = flow_shift  # high expert ModelSamplingSD3
    workflow["103"]["inputs"]["shift"] = flow_shift  # low expert ModelSamplingSD3

    # Optional sampler/scheduler from the preset (empty -> keep the template's euler/simple).
    if segment.sampler_name:
        workflow["86"]["inputs"]["sampler_name"] = segment.sampler_name
        workflow["85"]["inputs"]["sampler_name"] = segment.sampler_name
    if segment.scheduler:
        workflow["86"]["inputs"]["scheduler"] = segment.scheduler
        workflow["85"]["inputs"]["scheduler"] = segment.scheduler

    # De-distill bypass: when a lightx2v strength is 0, drop that Lightning LoRA from the graph
    # (repoint its consumer past it) so the expert runs raw and CFG > 1 does real work.
    if workflow["101"]["inputs"]["strength_model"] == 0:
        workflow["104"]["inputs"]["model"] = workflow["101"]["inputs"]["model"]  # high chain skips node 101
    if workflow["102"]["inputs"]["strength_model"] == 0:
        workflow["103"]["inputs"]["model"] = workflow["102"]["inputs"]["model"]  # low chain skips node 102

    # Negative prompt
    if segment.negative_prompt is not None:
        workflow["89"]["inputs"]["text"] = segment.negative_prompt

    # Positive prompt
    workflow["93"]["inputs"]["text"] = segment.prompt

    # Seed
    workflow["86"]["inputs"]["noise_seed"] = segment.seed

    # Video dimensions and frame count
    workflow["98"]["inputs"]["width"] = segment.width
    workflow["98"]["inputs"]["height"] = segment.height
    workflow["98"]["inputs"]["length"] = gen["wan_frames"]

    # Start image
    if start_image_filename:
        workflow["97"]["inputs"]["image"] = start_image_filename
    else:
        # Text-to-video: remove LoadImage and disconnect from WanImageToVideo
        del workflow["97"]
        del workflow["98"]["inputs"]["start_image"]

    # User LoRAs
    if segment.loras:
        _add_user_loras(workflow, [item.model_dump() for item in segment.loras])

    # RIFE frame interpolation, straight from VAEDecode
    rife_multiplier = gen["rife_multiplier"]
    rife_input = ["87", 0]   # straight from VAEDecode; nothing sits between any more
    workflow["200"] = {
        "class_type": "RIFE VFI",
        "inputs": {
            "ckpt_name": "rife49.pth",
            "clear_cache_after_n_frames": 10,
            "multiplier": rife_multiplier,
            "fast_mode": True,
            "ensemble": True,
            "scale_factor": 1.0,
            "dtype": "float16",
            "torch_compile": False,
            "batch_size": 1,
            "frames": rife_input,
        },
        "_meta": {"title": f"RIFE {rife_multiplier}x Interpolation"},
    }

    # VHS_VideoCombine output — always reads from RIFE (last processing step)
    video_input = ["200", 0]
    workflow["186"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": gen["output_fps"],
            "loop_count": 0,
            "filename_prefix": "output",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 15,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
            "images": video_input,
        },
        "_meta": {"title": "Video Combine"},
    }

    logger.info(
        "Built workflow: %dx%d, %d frames @ %dfps, RIFE %dx, speed=%.1fx, seed=%d",
        segment.width,
        segment.height,
        gen["wan_frames"],
        GENERATION_FPS,
        rife_multiplier,
        segment.speed,
        segment.seed,
    )
    return workflow
