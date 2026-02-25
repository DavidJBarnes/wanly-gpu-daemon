"""Build ComfyUI API workflows from segment parameters.

Ports the Wan2.2 I2V workflow template from v1 (wan22-video-generator) with
dynamic node injection for LoRAs, face swap, RIFE interpolation, and video output.
"""

import copy
import logging
from typing import Any

from daemon.config import settings
from daemon.schemas import SegmentClaim

logger = logging.getLogger(__name__)

# Generation is always at 15fps; RIFE interpolation brings it to target fps.
GENERATION_FPS = 15

# Node IDs for dynamically added user LoRA pairs (up to 3).
LORA_NODE_IDS = {
    "high": ["118", "120", "122"],
    "low": ["119", "121", "123"],
}

# Base Wan2.2 14B Image-to-Video workflow in ComfyUI API format.
# Dynamic nodes (RIFE, VHS_VideoCombine, faceswap, user LoRAs) are added at runtime.
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


def _calculate_generation_params(target_fps: int, duration_sec: float) -> dict[str, Any]:
    rife_multiplier = target_fps // GENERATION_FPS
    wan_frames = int(duration_sec * GENERATION_FPS) + 1
    return {
        "wan_frames": wan_frames,
        "rife_multiplier": rife_multiplier,
        "output_fps": target_fps,
    }


def _add_user_loras(workflow: dict, loras: list[dict]) -> None:
    """Add user LoRA nodes and rewire the lightx2v chain."""
    loras = [l for l in loras if l.get("high_file") or l.get("low_file")]
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


def _add_faceswap(workflow: dict, segment: SegmentClaim) -> None:
    """Add face swap nodes (188 LoadImage + 183 FaceSwap)."""
    # Node 188: load face source image
    workflow["188"] = {
        "class_type": "LoadImage",
        "inputs": {"image": segment.faceswap_image},
        "_meta": {"title": "Face Swap Source"},
    }

    faces_order = segment.faceswap_faces_order or "left-right"
    faces_index = segment.faceswap_faces_index or "0"

    if segment.faceswap_method == "reactor":
        workflow["189"] = {
            "class_type": "ReActorOptions",
            "inputs": {
                "input_faces_order": faces_order,
                "input_faces_index": faces_index,
                "detect_gender_input": "no",
                "source_faces_order": "left-right",
                "source_faces_index": "0",
                "detect_gender_source": "no",
                "console_log_level": 1,
                "restore_swapped_only": True,
            },
            "_meta": {"title": "ReActor Options"},
        }
        workflow["183"] = {
            "class_type": "ReActorFaceSwapOpt",
            "inputs": {
                "enabled": True,
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "codeformer-v0.1.0.pth",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.8,
                "input_image": ["87", 0],
                "source_image": ["188", 0],
                "options": ["189", 0],
            },
            "_meta": {"title": "ReActor Face Swap"},
        }
        logger.info("Added ReActor face swap nodes")
    else:
        # FaceFusion (default)
        facefusion_inputs: dict[str, Any] = {
            "source_images": ["188", 0],
            "target_image": ["87", 0],
            "api_token": "-1",
            "face_swapper_model": "inswapper_128",
            "face_detector_model": "retinaface",
            "pixel_boost": "512x512",
            "face_occluder_model": "xseg_1",
            "face_parser_model": "bisenet_resnet_34",
            "face_mask_blur": 0.3,
            "face_selector_mode": "reference",
            "face_position": int(faces_index),
            "sort_order": faces_order,
            "score_threshold": 0.5,
            "use_box_mask": True,
            "use_occlusion_mask": True,
            "use_area_mask": True,
            "use_region_mask": False,
            "face_mask_areas": "upper-face,lower-face,mouth",
            "face_mask_regions": "skin,nose,mouth,upper-lip,lower-lip",
            "face_mask_padding": "0,0,0,0",
            "reference_image": ["188", 0],
            "reference_face_distance": 0.8,
        }
        workflow["183"] = {
            "class_type": "AdvancedSwapFaceImage",
            "inputs": facefusion_inputs,
            "_meta": {"title": "FaceFusion Face Swap"},
        }
        logger.info("Added FaceFusion face swap nodes")


def build_workflow(
    segment: SegmentClaim,
    start_image_filename: str | None = None,
    initial_reference_image_filename: str | None = None,
) -> dict:
    """Build a complete ComfyUI workflow from segment parameters.

    Args:
        segment: The claimed segment with all generation parameters.
        start_image_filename: The ComfyUI-local filename of the start image
            (already uploaded). If None, the LoadImage node (97) is removed
            for text-to-video generation.
        initial_reference_image_filename: The ComfyUI-local filename of the
            job's original input image for PainterLongVideo identity anchoring.
            When provided (segment > 0), node 98 is swapped from WanImageToVideo
            to PainterLongVideo with dual-reference inputs.
    """
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds)
    workflow = copy.deepcopy(WAN_I2V_API_WORKFLOW)

    # Inject model filenames from config
    workflow["84"]["inputs"]["clip_name"] = settings.clip_model
    workflow["90"]["inputs"]["vae_name"] = settings.vae_model
    workflow["95"]["inputs"]["unet_name"] = settings.unet_high_model
    workflow["96"]["inputs"]["unet_name"] = settings.unet_low_model
    workflow["101"]["inputs"]["lora_name"] = settings.lightx2v_lora_high
    workflow["102"]["inputs"]["lora_name"] = settings.lightx2v_lora_low

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

    # PainterLongVideo swap for identity anchoring on segment > 0
    if initial_reference_image_filename and start_image_filename:
        # Node 300: load original input image (identity anchor)
        workflow["300"] = {
            "class_type": "LoadImage",
            "inputs": {"image": initial_reference_image_filename},
            "_meta": {"title": "Initial Reference Image"},
        }
        # Node 301: CLIP Vision model loader
        workflow["301"] = {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": settings.clip_vision_model},
            "_meta": {"title": "CLIP Vision Loader"},
        }
        # Node 302: Encode reference image with CLIP Vision
        workflow["302"] = {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["301", 0],
                "image": ["300", 0],
                "crop": "center",
            },
            "_meta": {"title": "CLIP Vision Encode Reference"},
        }
        # Replace WanImageToVideo with PainterLongVideo
        workflow["98"] = {
            "class_type": "PainterLongVideo",
            "inputs": {
                "positive": ["93", 0],
                "negative": ["89", 0],
                "vae": ["90", 0],
                "width": segment.width,
                "height": segment.height,
                "length": gen["wan_frames"],
                "batch_size": 1,
                "previous_video": ["97", 0],
                "motion_frames": 5,
                "motion_amplitude": 1.3,
                "initial_reference_image": ["300", 0],
                "clip_vision_output": ["302", 0],
                "start_image": ["97", 0],
            },
            "_meta": {"title": "PainterLongVideo Identity Anchor"},
        }
        logger.info(
            "Swapped to PainterLongVideo (segment %d, ref=%s, clip_vision=%s)",
            segment.index,
            initial_reference_image_filename,
            settings.clip_vision_model,
        )

    # User LoRAs
    if segment.loras:
        _add_user_loras(workflow, [l.model_dump() for l in segment.loras])

    # Face swap
    faceswap = segment.faceswap_enabled and segment.faceswap_image
    if faceswap:
        _add_faceswap(workflow, segment)

    # RIFE frame interpolation
    rife_multiplier = gen["rife_multiplier"]
    rife_input = ["183", 0] if faceswap else ["87", 0]
    workflow["200"] = {
        "class_type": "RIFE VFI",
        "inputs": {
            "ckpt_name": "rife49.pth",
            "clear_cache_after_n_frames": 10,
            "multiplier": rife_multiplier,
            "fast_mode": True,
            "ensemble": True,
            "scale_factor": 1,
            "frames": rife_input,
        },
        "_meta": {"title": f"RIFE {rife_multiplier}x Interpolation"},
    }

    # VHS_VideoCombine output
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
            "images": ["200", 0],
        },
        "_meta": {"title": "Video Combine"},
    }

    logger.info(
        "Built workflow: %dx%d, %d frames @ %dfps, RIFE %dx, seed=%d, faceswap=%s",
        segment.width,
        segment.height,
        gen["wan_frames"],
        GENERATION_FPS,
        rife_multiplier,
        segment.seed,
        faceswap,
    )
    return workflow
