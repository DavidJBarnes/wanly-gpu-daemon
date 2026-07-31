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
from daemon.motion_analyzer import estimate_motion_from_flow
from daemon.stage_log import log_event

logger = logging.getLogger(__name__)

# Generation is always at 15fps; RIFE interpolation brings it to target fps.
GENERATION_FPS = 15

# Node IDs for dynamically added user LoRA pairs (up to 3).
LORA_NODE_IDS = {
    "high": ["118", "120", "122"],
    "low": ["119", "121", "123"],
}

# Schedulers WanVideoSampler (WanVideoWrapper) accepts. These are NOT the KSampler
# names ("simple", "karras", ...) the traditional path uses — passing one of those
# makes ComfyUI reject the prompt with a 400. Shared by the VACE and Lynx builders.
WANVIDEO_SCHEDULERS = frozenset({
    "unipc", "unipc/beta", "dpm++", "dpm++/beta", "dpm++_sde", "dpm++_sde/beta",
    "euler", "euler/beta", "deis", "lcm", "lcm/beta", "res_multistep",
    "flowmatch_causvid", "flowmatch_distill", "flowmatch_pusa", "multitalk",
})

# Wan native resolution buckets Lynx is validated at. Anything else is rejected rather
# than snapped, because an off-bucket latent grid degrades identity silently.
# Wan native buckets, both orientations. Portrait matters for the i2v path: a start frame
# is usually shot portrait, and forcing it into a landscape bucket centre-crops the subject.
# Nothing in Lynx is orientation-specific — the constraint is Wan's bucket grid.
LYNX_RESOLUTIONS = frozenset({(832, 480), (480, 832), (1280, 720), (720, 1280)})

# Lynx adapter arms. The ip layers and the resampler are a matched pair — the
# resampler's proj_out dimension must match the ip layers it feeds. A mismatched pair
# loads without raising and yields garbage identity, so it is validated up front.
LYNX_ARMS = ("lite", "full")

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


def _calculate_motion_amplitude(
    segment_index: int,
    previous_motion_magnitude: float | None,
    motion_amplitude_setting: float,
) -> float:
    """Calculate motion_amplitude for a segment based on previous motion data.
    
    Args:
        segment_index: Index of the segment being built (0 = first segment)
        previous_motion_magnitude: Measured motion from previous segment (px/frame)
        motion_amplitude_setting: Default/configured motion_amplitude value
        
    Returns:
        motion_amplitude to use for this segment
    """
    # Segment 0 uses the default/configured value
    if segment_index == 0:
        return motion_amplitude_setting
    
    # If no previous motion data, use default
    if previous_motion_magnitude is None:
        return motion_amplitude_setting
    
    # Motion matching enabled: estimate amplitude to match previous motion
    # Use previous_motion_magnitude as target to achieve consistency
    estimated = estimate_motion_from_flow(
        previous_motion_magnitude=previous_motion_magnitude,
        previous_motion_amplitude=motion_amplitude_setting,
        target_motion_magnitude=previous_motion_magnitude,
    )
    
    # Clamp to valid range
    from daemon.config import settings
    return max(settings.motion_amplitude_min, min(settings.motion_amplitude_max, estimated))


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


def _add_faceswap(workflow: dict, segment: SegmentClaim, input_node: str = "87") -> None:
    """Add face swap nodes (188 LoadImage + 183 FaceSwap).

    input_node controls where frames come from: "87" (VAEDecode) for generation
    workflows, or "400" (VHS_LoadVideo) for faceswap-only reprocessing.
    """
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
                "input_image": [input_node, 0],
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
            "target_image": [input_node, 0],
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


def build_faceswap_workflow(segment: SegmentClaim, video_filename: str) -> dict:
    """Build a faceswap-only workflow: load video at 15fps → faceswap → RIFE → encode.

    Matches the normal generation pipeline: faceswap on native-rate frames,
    then RIFE interpolates back to target fps.
    """
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)
    workflow: dict[str, Any] = {}

    workflow["400"] = {
        "class_type": "VHS_LoadVideo",
        "inputs": {
            "video": video_filename,
            "force_rate": float(GENERATION_FPS),
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        },
        "_meta": {"title": "Load Existing Video @ 15fps"},
    }

    _add_faceswap(workflow, segment, input_node="400")

    rife_multiplier = gen["rife_multiplier"]
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
            "frames": ["183", 0],
        },
        "_meta": {"title": f"RIFE {rife_multiplier}x Interpolation"},
    }

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

    logger.info("Built faceswap-only workflow (%d nodes, rife %dx) for video: %s",
                len(workflow), rife_multiplier, video_filename)
    return workflow


def build_seed_faceswap_workflow(segment: SegmentClaim, seed_filename: str) -> dict:
    """Single-image faceswap of a continuation SEED frame (the extracted last frame),
    re-anchoring identity to the canonical face before it seeds the next i2v segment.

    Reuses the FaceFusion/ReActor node stack (node 183) on one still image instead of a
    video. `segment.faceswap_image` must already point at the canonical face (the caller
    sets it). If FaceFusion finds no face it passes the frame through unchanged — the
    caller diffs the result to detect that and falls back to the raw seed.
    """
    workflow: dict[str, Any] = {}
    workflow["400"] = {
        "class_type": "LoadImage",
        "inputs": {"image": seed_filename},
        "_meta": {"title": "Seed frame (last frame)"},
    }
    _add_faceswap(workflow, segment, input_node="400")
    workflow["186"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "seed_swap", "images": ["183", 0]},
        "_meta": {"title": "Re-anchored seed"},
    }
    logger.info("Built seed-faceswap workflow (%d nodes) for seed: %s", len(workflow), seed_filename)
    return workflow


def vace_num_frames(segment: SegmentClaim) -> int:
    """WAN generation frame count for a VACE segment, rounded to a valid 4n+1."""
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)
    n = max(gen["wan_frames"], 5)
    return ((n - 1) // 4) * 4 + 1


def _build_vace_lora_chains(workflow: dict, segment: SegmentClaim) -> tuple[list | None, list | None]:
    """Build per-expert WanVideoLoraSelect chains: Lightning distill (optional) + user
    character/motion LoRAs, chained via prev_lora. Returns (high_ref, low_ref) node refs."""
    counter = [700]

    def add(lora_name: str, strength: float, prev: list | None) -> list:
        nid = str(counter[0])
        counter[0] += 1
        inputs = {"lora": lora_name, "strength": float(strength), "merge_loras": True}
        if prev is not None:
            inputs["prev_lora"] = prev
        workflow[nid] = {"class_type": "WanVideoLoraSelect", "inputs": inputs}
        return [nid, 0]

    high_ref = low_ref = None
    if settings.vace_lightning:
        high_ref = add(settings.vace_lightning_high, 1.0, None)
        low_ref = add(settings.vace_lightning_low, 1.0, None)
    for lora in (segment.loras or []):
        if lora.high_file:
            high_ref = add(lora.high_file, lora.high_weight, high_ref)
        if lora.low_file:
            low_ref = add(lora.low_file, lora.low_weight, low_ref)
    return high_ref, low_ref


def build_vace_workflow(
    segment: SegmentClaim,
    control_filename: str,
    mask_filename: str,
    reference_filename: str,
    num_frames: int,
) -> dict:
    """Fun-VACE (Wan2.2 T2V-A14B) continuation workflow via WanVideoWrapper.

    control_filename = num_frames-long video (kept tail + grey pad); mask_filename =
    matching mask (black=keep tail, white=generate); reference = identity anchor image.
    Dual-expert (high/low) T2V base each fed its Fun-VACE module via extra_model, with
    Lightning + user LoRAs. Output is RIFE-interpolated to the segment's target fps so
    it stitches with traditional segments. Validated on the 3090 bench.
    """
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)
    neg = segment.negative_prompt or ""
    vhs = {"force_rate": 0, "custom_width": 0, "custom_height": 0,
           "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1}
    wf: dict[str, Any] = {}

    # VACE module selectors -> each loader's extra_model
    wf["501"] = {"class_type": "WanVideoVACEModelSelect", "inputs": {"vace_model": settings.vace_module_high}}
    wf["502"] = {"class_type": "WanVideoVACEModelSelect", "inputs": {"vace_model": settings.vace_module_low}}
    wf["503"] = {"class_type": "WanVideoBlockSwap", "inputs": {
        "blocks_to_swap": settings.vace_blocks_to_swap, "offload_img_emb": False,
        "offload_txt_emb": False, "vace_blocks_to_swap": 0}}

    high_lora, low_lora = _build_vace_lora_chains(wf, segment)

    high_in = {"model": settings.vace_t2v_high_model, "base_precision": "fp16",
               "quantization": "fp8_e4m3fn_scaled", "load_device": "offload_device",
               "attention_mode": "sdpa", "extra_model": ["501", 0], "block_swap_args": ["503", 0]}
    if high_lora:
        high_in["lora"] = high_lora
    wf["510"] = {"class_type": "WanVideoModelLoader", "inputs": high_in}

    low_in = {"model": settings.vace_t2v_low_model, "base_precision": "fp16",
              "quantization": "fp8_e4m3fn_scaled", "load_device": "offload_device",
              "attention_mode": "sdpa", "extra_model": ["502", 0], "block_swap_args": ["503", 0]}
    if low_lora:
        low_in["lora"] = low_lora
    wf["511"] = {"class_type": "WanVideoModelLoader", "inputs": low_in}

    wf["520"] = {"class_type": "WanVideoVAELoader", "inputs": {"model_name": settings.vae_model, "precision": "bf16"}}
    wf["521"] = {"class_type": "WanVideoTextEncodeCached", "inputs": {
        "model_name": settings.vace_t5_model, "precision": "bf16",
        "positive_prompt": segment.prompt, "negative_prompt": neg,
        "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}}

    wf["530"] = {"class_type": "VHS_LoadVideo", "inputs": {"video": control_filename, **vhs}}
    wf["531"] = {"class_type": "VHS_LoadVideo", "inputs": {"video": mask_filename, **vhs}}
    wf["532"] = {"class_type": "ImageToMask", "inputs": {"image": ["531", 0], "channel": "red"}}
    wf["533"] = {"class_type": "LoadImage", "inputs": {"image": reference_filename}}

    wf["540"] = {"class_type": "WanVideoVACEEncode", "inputs": {
        "vae": ["520", 0], "width": segment.width, "height": segment.height,
        "num_frames": num_frames, "strength": 1.0, "vace_start_percent": 0.0,
        "vace_end_percent": 1.0, "input_frames": ["530", 0], "ref_images": ["533", 0],
        "input_masks": ["532", 0]}}

    steps, cfg, boundary, shift = settings.vace_steps, settings.vace_cfg, settings.vace_boundary, settings.vace_shift
    # WanVideoSampler has no sampler_name and its own scheduler set — NOT the KSampler names
    # (e.g. "simple") the preset carries for the traditional path. Only honor the preset's
    # scheduler if WanVideoSampler accepts it; otherwise keep unipc (else ComfyUI 400s).
    vace_scheduler = segment.scheduler if segment.scheduler in WANVIDEO_SCHEDULERS else "unipc"
    wf["550"] = {"class_type": "WanVideoSampler", "inputs": {
        "model": ["510", 0], "image_embeds": ["540", 0], "text_embeds": ["521", 0],
        "steps": steps, "cfg": cfg, "shift": shift, "seed": segment.seed, "force_offload": True,
        "scheduler": vace_scheduler, "riflex_freq_index": 0, "start_step": 0, "end_step": boundary}}
    wf["551"] = {"class_type": "WanVideoSampler", "inputs": {
        "model": ["511", 0], "image_embeds": ["540", 0], "text_embeds": ["521", 0],
        "samples": ["550", 0], "steps": steps, "cfg": cfg, "shift": shift, "seed": segment.seed,
        "force_offload": True, "scheduler": vace_scheduler, "riflex_freq_index": 0,
        "start_step": boundary, "end_step": -1}}

    wf["560"] = {"class_type": "WanVideoDecode", "inputs": {
        "vae": ["520", 0], "samples": ["551", 0], "enable_vae_tiling": False,
        "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}}

    rife_mult = gen["rife_multiplier"]
    final_images = ["560", 0]
    if rife_mult > 1:
        wf["570"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10, "multiplier": rife_mult,
            "fast_mode": True, "ensemble": True, "scale_factor": 1.0, "dtype": "float16",
            "torch_compile": False, "batch_size": 1, "frames": ["560", 0]}}
        final_images = ["570", 0]
    wf["580"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate": gen["output_fps"], "loop_count": 0, "filename_prefix": "output",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 15, "save_metadata": True,
        "trim_to_audio": False, "pingpong": False, "save_output": True, "images": final_images}}

    logger.info("Built VACE continuation workflow (%d nodes, %d frames, rife %dx)",
                len(wf), num_frames, rife_mult)
    return wf


class LynxValidationError(ValueError):
    """A Lynx job asked for something the graph cannot honour.

    Raised instead of silently clamping: an off-bucket resolution or an off-grid
    frame count degrades identity in ways that are hard to attribute after the fact,
    so the job fails loudly with the offending value named.
    """


def _pick(override: Any, default: Any) -> Any:
    """Per-job-override -> settings-default precedence.

    ``None`` means "not set for this job"; every other value (including 0 and "")
    is an intentional override and wins.
    """
    return default if override is None else override


def _lynx_arm(filename: str) -> str | None:
    """Classify a Lynx adapter file as the 'lite' or 'full' arm by filename marker."""
    lowered = filename.lower()
    found = [arm for arm in LYNX_ARMS if arm in lowered]
    return found[0] if len(found) == 1 else None


def lynx_num_frames(segment: SegmentClaim) -> int:
    """WAN generation frame count for a Lynx segment, rounded to a valid 4n+1."""
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)
    n = max(int(gen["wan_frames"]), 5)
    return ((n - 1) // 4) * 4 + 1


LYNX_MAX_PIXELS = 1280 * 720   # 720p budget; beyond this block swap thrashes on 24GB


def _validate_lynx_resolution(width: int, height: int) -> None:
    """Any /16 resolution within the pixel budget is allowed — NOT a fixed bucket list.

    A fixed list was actively harmful: ``WanVideoImageToVideoEncode`` resizes the start
    image with ``common_upscale(..., W, H, "lanczos", "disabled")`` — crop mode *disabled*,
    i.e. a plain resize with no aspect preservation. Forcing a 768x1024 (0.750) start frame
    into 480x832 (0.577) therefore stretched the face ~1.30x vertically, which both looks
    wrong and depresses ArcFace similarity (the embedding is geometry-sensitive). Callers
    should derive the generation resolution from the start frame's aspect via
    ``aspect_matched_resolution`` instead of snapping to a bucket.
    """
    for name, value in (("width", width), ("height", height)):
        if value % 16 != 0:
            raise LynxValidationError(
                f"Lynx {name} must be divisible by 16 (VAE 8x downsample x 2x patch); "
                f"got {value}."
            )
    if width * height > LYNX_MAX_PIXELS:
        raise LynxValidationError(
            f"Lynx resolution {width}x{height} exceeds the {LYNX_MAX_PIXELS}px budget."
        )


def aspect_matched_resolution(
    src_width: int, src_height: int, pixel_budget: int = 480 * 832
) -> tuple[int, int]:
    """Largest /16 resolution matching the source aspect within ``pixel_budget``.

    Prevents the stretch described in ``_validate_lynx_resolution``: a 768x1024 source
    resolves to 624x832 (aspect 0.750, exact) rather than 480x832 (0.577).
    """
    if src_width <= 0 or src_height <= 0:
        raise LynxValidationError(f"invalid source dimensions {src_width}x{src_height}")
    aspect = src_width / src_height
    # height from the budget at this aspect: budget = (h*aspect) * h
    height = int(((pixel_budget / aspect) ** 0.5) // 16 * 16)
    width = int((height * aspect) // 16 * 16)
    while width * height > pixel_budget and height > 16:
        height -= 16
        width = int((height * aspect) // 16 * 16)
    return max(width, 16), max(height, 16)


def _validate_lynx(
    segment: SegmentClaim,
    subject_filename: str | None,
    num_frames: int,
    ip_layers: str,
    resampler: str,
) -> None:
    """Reject unsupported Lynx parameters with named values. No clamping."""
    if not subject_filename:
        raise LynxValidationError(
            "Lynx requires a subject_image: identity is conditioned on an ArcFace "
            "embedding of that image, so there is nothing to condition on without it."
        )
    _validate_lynx_resolution(segment.width, segment.height)
    if num_frames < 5 or (num_frames - 1) % 4 != 0:
        raise LynxValidationError(
            f"Lynx frame count must be on the 4n+1 grid and >= 5 (e.g. 81); got {num_frames}."
        )
    ip_arm, res_arm = _lynx_arm(ip_layers), _lynx_arm(resampler)
    if ip_arm is None or res_arm is None or ip_arm != res_arm:
        raise LynxValidationError(
            "Lynx ip layers and resampler must be the same arm (both 'lite' or both "
            f"'full'); got ip_layers={ip_layers!r} (arm={ip_arm}), "
            f"resampler={resampler!r} (arm={res_arm})."
        )


def build_lynx_workflow(
    segment: SegmentClaim,
    subject_filename: str,
    num_frames: int,
) -> dict[str, Any]:
    """Lynx identity-preserving workflow (Wan2.1 T2V-14B + Lynx adapters) via WanVideoWrapper.

    Lynx conditions a *text-to-video* base on a subject image through two adapters, both
    re-applied at every denoising step (unlike a character LoRA, which bakes identity into
    the weights):

      ip_scale  — strength of the ID-adapter: ArcFace embedding of a 112x112 aligned face
                  crop, mapped to identity tokens by a Perceiver resampler. Controls *who*
                  the face is; pushing it too high tends to freeze expression.
      ref_scale — strength of the Ref-adapter: dense VAE features of a larger (256px) face
                  crop, cross-attended in the DiT blocks. Controls fine appearance detail
                  (skin, hair, lighting); too high drags the reference's pose/lighting in.

    The subject image is NOT a first frame — this is not i2v, and the subject never appears
    as frame 0. Character LoRAs remain supported and stack on top of the adapters.

    ``num_frames`` must be on the 4n+1 grid and the resolution a Wan native bucket; both are
    validated rather than clamped. Output is RIFE-interpolated to the segment's target fps so
    Lynx segments stitch alongside traditional ones.
    """
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)

    ip_layers = _pick(segment.lynx_ip_layers, settings.lynx_ip_layers)
    resampler = _pick(segment.lynx_resampler, settings.lynx_resampler)
    _validate_lynx(segment, subject_filename, num_frames, ip_layers, resampler)

    ip_scale = _pick(segment.lynx_ip_scale, settings.lynx_ip_scale)
    ref_scale = _pick(segment.lynx_ref_scale, settings.lynx_ref_scale)
    lynx_cfg_scale = _pick(segment.lynx_cfg_scale, settings.lynx_lynx_cfg_scale)
    start_percent = _pick(segment.lynx_start_percent, settings.lynx_start_percent)
    end_percent = _pick(segment.lynx_end_percent, settings.lynx_end_percent)
    ref_blocks = _pick(segment.lynx_ref_blocks_to_use, settings.lynx_ref_blocks_to_use)
    steps = _pick(segment.lynx_steps, settings.lynx_steps)
    cfg = _pick(segment.lynx_cfg, settings.lynx_cfg)
    shift = _pick(segment.lynx_shift, settings.lynx_shift)
    distill_strength = _pick(segment.lynx_distill_strength, settings.lynx_distill_strength)
    scheduler_pref = _pick(segment.lynx_scheduler, settings.lynx_scheduler)
    scheduler = scheduler_pref if scheduler_pref in WANVIDEO_SCHEDULERS else settings.lynx_scheduler

    wf: dict[str, Any] = {}

    # --- Adapters + block swap -------------------------------------------------
    # Both adapters reach the model through one chained extra_model list; the loader
    # merges them into the base state dict at load time.
    wf["600"] = {"class_type": "WanVideoBlockSwap", "inputs": {
        "blocks_to_swap": settings.lynx_blocks_to_swap, "offload_img_emb": False,
        "offload_txt_emb": False, "vace_blocks_to_swap": 0}}
    wf["601"] = {"class_type": "WanVideoExtraModelSelect", "inputs": {"extra_model": ip_layers}}
    wf["602"] = {"class_type": "WanVideoExtraModelSelect", "inputs": {
        "extra_model": settings.lynx_ref_layers, "prev_model": ["601", 0]}}

    # --- LoRA chain: distill first, then optional character LoRAs on top --------
    # Wan2.1 T2V is a single-expert model, so unlike the 2.2 dual-expert path there is
    # one chain; a LoRA's high_file is used, falling back to low_file.
    lora_ref: list[Any] | None = None
    lora_counter = 700
    if distill_strength:
        wf[str(lora_counter)] = {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": settings.lynx_distill_lora, "strength": float(distill_strength),
            "merge_loras": True}}
        lora_ref = [str(lora_counter), 0]
        lora_counter += 1
    for lora in (segment.loras or []):
        lora_file = lora.high_file or lora.low_file
        if not lora_file:
            continue
        weight = lora.high_weight if lora.high_file else lora.low_weight
        inputs: dict[str, Any] = {
            "lora": lora_file, "strength": float(weight), "merge_loras": True}
        if lora_ref is not None:
            inputs["prev_lora"] = lora_ref
        wf[str(lora_counter)] = {"class_type": "WanVideoLoraSelect", "inputs": inputs}
        lora_ref = [str(lora_counter), 0]
        lora_counter += 1

    loader_inputs: dict[str, Any] = {
        "model": settings.lynx_t2v_model, "base_precision": settings.lynx_base_precision,
        "quantization": settings.lynx_quantization, "load_device": "offload_device",
        "attention_mode": "sdpa", "extra_model": ["602", 0], "block_swap_args": ["600", 0]}
    if lora_ref is not None:
        loader_inputs["lora"] = lora_ref
    wf["610"] = {"class_type": "WanVideoModelLoader", "inputs": loader_inputs}

    # --- VAE + text encoders ---------------------------------------------------
    wf["620"] = {"class_type": "WanVideoVAELoader", "inputs": {
        "model_name": settings.vae_model, "precision": "bf16"}}
    # Cached encode keeps umt5-xxl off the GPU between jobs — the residency that has
    # OOM'd this box before. Same handling as the VACE path.
    wf["621"] = {"class_type": "WanVideoTextEncodeCached", "inputs": {
        "model_name": settings.lynx_t5_model, "precision": "bf16",
        "positive_prompt": segment.prompt, "negative_prompt": segment.negative_prompt or "",
        "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}}
    # Second, separate embed for the reference-extraction pass. ByteDance hardcode this
    # prompt; WanVideoAddLynxEmbeds raises if ref_image is supplied without it.
    wf["622"] = {"class_type": "WanVideoTextEncodeCached", "inputs": {
        "model_name": settings.lynx_t5_model, "precision": "bf16",
        "positive_prompt": settings.lynx_ref_prompt, "negative_prompt": "",
        "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}}

    # --- Subject conditioning --------------------------------------------------
    # One crop node yields both faces: a 112x112 ArcFace-aligned crop for the ID adapter
    # and a wider 256px crop for the ref adapter. Raises "No face detected" upstream,
    # which the executor surfaces as a job validation failure.
    wf["630"] = {"class_type": "LoadImage", "inputs": {"image": subject_filename}}
    wf["631"] = {"class_type": "LynxInsightFaceCrop", "inputs": {"image": ["630", 0]}}
    wf["632"] = {"class_type": "LoadLynxResampler", "inputs": {
        "model_name": resampler, "precision": settings.lynx_resampler_precision}}
    wf["633"] = {"class_type": "LynxEncodeFaceIP", "inputs": {
        "resampler": ["632", 0], "ip_image": ["631", 0]}}

    # --- Empty latents + Lynx embed injection ----------------------------------
    wf["640"] = {"class_type": "WanVideoEmptyEmbeds", "inputs": {
        "width": segment.width, "height": segment.height, "num_frames": num_frames}}
    embed_inputs: dict[str, Any] = {
        "embeds": ["640", 0], "ip_scale": float(ip_scale), "ref_scale": float(ref_scale),
        "lynx_cfg_scale": float(lynx_cfg_scale), "start_percent": float(start_percent),
        "end_percent": float(end_percent), "vae": ["620", 0],
        "lynx_ip_embeds": ["633", 0], "ref_image": ["631", 1], "ref_text_embed": ["622", 0]}
    # Omit when empty: the node treats "" as "all blocks", and sending the key at all
    # would need it wired as a link (it is declared forceInput).
    if ref_blocks:
        embed_inputs["ref_blocks_to_use"] = ref_blocks
    wf["641"] = {"class_type": "WanVideoAddLynxEmbeds", "inputs": embed_inputs}

    # --- Sample / decode / output ----------------------------------------------
    wf["650"] = {"class_type": "WanVideoSampler", "inputs": {
        "model": ["610", 0], "image_embeds": ["641", 0], "text_embeds": ["621", 0],
        "steps": steps, "cfg": cfg, "shift": shift, "seed": segment.seed,
        "force_offload": True, "scheduler": scheduler, "riflex_freq_index": 0,
        # MUST be set explicitly. WanVideoSampler declares the widget default as "comfy"
        # (nodes_sampler.py: rope_function widget) but its Python signature defaults to
        # "default" — and only the "comfy"/"comfy_chunked" branches ever populate the RoPE
        # freqs table (wanvideo/modules/model.py). Omitting the input therefore takes the
        # "default" path, leaves freqs None, and the sampler dies inside rope_apply_3d with
        # "'NoneType' object has no attribute 'split'". The ComfyUI GUI always sends the
        # widget default, so Kijai's reference workflow never trips this; an API-built graph
        # must send it itself.
        "rope_function": "comfy",
        "start_step": 0, "end_step": -1}}
    wf["660"] = {"class_type": "WanVideoDecode", "inputs": {
        "vae": ["620", 0], "samples": ["650", 0], "enable_vae_tiling": False,
        "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}}

    rife_mult = gen["rife_multiplier"]
    final_images = ["660", 0]
    if rife_mult > 1:
        wf["670"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10, "multiplier": rife_mult,
            "fast_mode": True, "ensemble": True, "scale_factor": 1.0, "dtype": "float16",
            "torch_compile": False, "batch_size": 1, "frames": ["660", 0]}}
        final_images = ["670", 0]
    wf["680"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate": gen["output_fps"], "loop_count": 0, "filename_prefix": "output",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 15, "save_metadata": True,
        "trim_to_audio": False, "pingpong": False, "save_output": True, "images": final_images}}

    log_event(
        logger, "lynx.graph_built", str(segment.id),
        nodes=len(wf), num_frames=num_frames,
        resolution=f"{segment.width}x{segment.height}",
        arm=_lynx_arm(ip_layers), ip_scale=ip_scale, ref_scale=ref_scale,
        steps=steps, cfg=cfg, shift=shift, scheduler=scheduler,
        distill_strength=distill_strength, character_loras=len(segment.loras or []),
        rife_multiplier=rife_mult,
    )
    return wf


def build_lynx_i2v_workflow(
    segment: SegmentClaim,
    subject_filename: str,
    start_image_filename: str,
    num_frames: int,
    ref_from_start_frame: bool = False,
) -> dict[str, Any]:
    """Lynx identity adapters on the Wan2.2 i2v dual-expert base.

    This is the i2v identity-lock configuration. Unlike ``build_lynx_workflow`` (Wan2.1
    T2V, where the subject image is the ONLY conditioning and the scene is invented from
    the prompt), here a real ``start_image`` drives the video exactly as the production
    i2v path does, and the Lynx adapters ride on top — re-asserting identity at every
    denoising step of every segment rather than baking it into weights the way a
    character LoRA does.

    Two things differ structurally from the T2V graph:

    * ``WanVideoImageToVideoEncode`` replaces ``WanVideoEmptyEmbeds`` as the source of
      image_embeds. ``clip_embeds`` is deliberately left unset — Wan2.2 dropped the CLIP
      image encoder, and the deprecated ``WanVideoImageClipEncode`` node must not be used.
    * Wan2.2 is dual-expert, so there are TWO model loaders and TWO sampler passes split
      at ``boundary`` (mirroring ``build_vace_workflow``). The adapter chain is attached to
      BOTH loaders: each expert is a separate WanModel, and the sampler raises "Lynx IP
      embeds provided, but the no lynx ip adapter layers found in the model" for whichever
      expert is missing them.

    ``ref_from_start_frame`` splits the two adapters' sources, which is usually what you want
    for identity *stabilisation* (as opposed to replacement). The adapters encode different
    things:

    * **ip** is an ArcFace embedding — it encodes *who*, and is largely pose- and
      lighting-invariant. It wants the best face pixels available, so it is fed from the
      canonical subject photo. Anchoring it on the start frame instead perpetuates whatever
      per-frame variation the upstream swap introduced (measured at 0.54-0.86 between start
      frames of the same person), giving within-clip stability but no cross-clip consistency.
    * **ref** is dense VAE features — it encodes *appearance* (skin, hair, lighting). Feeding
      it the start frame's own face crop matches the shot, so it does not fight the start
      frame the way a differently-lit canonical photo does.

    Both crops still come from ``LynxInsightFaceCrop``, which does the detection and alignment
    internally; splitting the sources just means two instances of that node.

    UNPROVEN. No public example of this combination exists; see the config notes and
    BUILD_wanly-lynx-engine.md for the cross-generation transfer risk.
    """
    ip_layers = _pick(segment.lynx_ip_layers, settings.lynx_ip_layers)
    resampler = _pick(segment.lynx_resampler, settings.lynx_resampler)
    _validate_lynx(segment, start_image_filename, num_frames, ip_layers, resampler)
    if not subject_filename:
        raise LynxValidationError("Lynx i2v needs a subject image for the identity adapters")

    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)
    steps = int(_pick(segment.lynx_steps, settings.lynx_i2v_steps))
    cfg = float(_pick(segment.lynx_cfg, settings.lynx_i2v_cfg))
    shift = float(_pick(segment.lynx_shift, settings.lynx_i2v_shift))
    boundary = min(int(settings.lynx_i2v_boundary), steps)
    scheduler = _pick(segment.lynx_scheduler, settings.lynx_scheduler)
    if scheduler not in WANVIDEO_SCHEDULERS:
        raise LynxValidationError(
            f"scheduler {scheduler!r} is not a WanVideoSampler scheduler "
            f"(valid: {sorted(WANVIDEO_SCHEDULERS)})"
        )

    wf: dict[str, Any] = {}

    # --- Adapters + block swap -------------------------------------------------
    wf["600"] = {"class_type": "WanVideoBlockSwap", "inputs": {
        "blocks_to_swap": settings.lynx_i2v_blocks_to_swap, "offload_img_emb": False,
        "offload_txt_emb": False, "vace_blocks_to_swap": 0}}
    wf["601"] = {"class_type": "WanVideoExtraModelSelect", "inputs": {"extra_model": ip_layers}}
    wf["602"] = {"class_type": "WanVideoExtraModelSelect", "inputs": {
        "extra_model": settings.lynx_ref_layers, "prev_model": ["601", 0]}}

    # --- Per-expert LoRA chains (2.2 is dual-expert, so high and low differ) ----
    def _lora_chain(expert: str, start_id: int) -> tuple[list[Any] | None, int]:
        ref: list[Any] | None = None
        node_id = start_id
        distill_strength = (
            settings.lynx_i2v_distill_high_strength if expert == "high"
            else settings.lynx_i2v_distill_low_strength
        )
        distill_file = (
            settings.lightx2v_lora_high if expert == "high" else settings.lightx2v_lora_low
        )
        if distill_strength:
            wf[str(node_id)] = {"class_type": "WanVideoLoraSelect", "inputs": {
                "lora": distill_file, "strength": float(distill_strength),
                "merge_loras": True}}
            ref = [str(node_id), 0]
            node_id += 1
        for lora in (segment.loras or []):
            lora_file = lora.high_file if expert == "high" else lora.low_file
            if not lora_file:
                continue
            weight = lora.high_weight if expert == "high" else lora.low_weight
            inputs: dict[str, Any] = {
                "lora": lora_file, "strength": float(weight), "merge_loras": True}
            if ref is not None:
                inputs["prev_lora"] = ref
            wf[str(node_id)] = {"class_type": "WanVideoLoraSelect", "inputs": inputs}
            ref = [str(node_id), 0]
            node_id += 1
        return ref, node_id

    high_lora, next_id = _lora_chain("high", 700)
    low_lora, _ = _lora_chain("low", next_id)

    def _loader(model_name: str, lora_ref: list[Any] | None) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "model": model_name,
            "base_precision": settings.lynx_i2v_base_precision,
            "quantization": settings.lynx_i2v_quantization,
            "load_device": "offload_device", "attention_mode": "sdpa",
            # Both experts get the adapter chain — see the class docstring.
            "extra_model": ["602", 0], "block_swap_args": ["600", 0]}
        if lora_ref is not None:
            inputs["lora"] = lora_ref
        return {"class_type": "WanVideoModelLoader", "inputs": inputs}

    wf["610"] = _loader(settings.lynx_i2v_high_model, high_lora)
    wf["611"] = _loader(settings.lynx_i2v_low_model, low_lora)

    # --- VAE + text encoders ---------------------------------------------------
    wf["620"] = {"class_type": "WanVideoVAELoader", "inputs": {
        "model_name": settings.vae_model, "precision": "bf16"}}
    wf["621"] = {"class_type": "WanVideoTextEncodeCached", "inputs": {
        "model_name": settings.lynx_t5_model, "precision": "bf16",
        "positive_prompt": segment.prompt, "negative_prompt": segment.negative_prompt or "",
        "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}}
    # The ref-extraction pass needs its own text embed; hardcoded in ByteDance's
    # implementation and required by the node whenever ref_image is supplied.
    wf["622"] = {"class_type": "WanVideoTextEncodeCached", "inputs": {
        "model_name": settings.lynx_t5_model, "precision": "bf16",
        "positive_prompt": settings.lynx_ref_prompt, "negative_prompt": "",
        "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}}

    # --- Identity chain (unchanged from the proven T2V graph) ------------------
    wf["630"] = {"class_type": "LoadImage", "inputs": {"image": subject_filename}}
    wf["631"] = {"class_type": "LynxInsightFaceCrop", "inputs": {"image": ["630", 0]}}
    wf["632"] = {"class_type": "LoadLynxResampler", "inputs": {
        "model_name": resampler, "precision": settings.lynx_resampler_precision}}
    wf["633"] = {"class_type": "LynxEncodeFaceIP", "inputs": {
        "resampler": ["632", 0], "ip_image": ["631", 0]}}

    # --- i2v conditioning: the START FRAME drives the video --------------------
    wf["635"] = {"class_type": "LoadImage", "inputs": {"image": start_image_filename}}
    wf["640"] = {"class_type": "WanVideoImageToVideoEncode", "inputs": {
        "width": segment.width, "height": segment.height, "num_frames": num_frames,
        "noise_aug_strength": settings.lynx_i2v_noise_aug_strength,
        "start_latent_strength": settings.lynx_i2v_start_latent_strength,
        "end_latent_strength": settings.lynx_i2v_end_latent_strength,
        "force_offload": True, "vae": ["620", 0], "start_image": ["635", 0]}}

    wf["641"] = {"class_type": "WanVideoAddLynxEmbeds", "inputs": {
        "embeds": ["640", 0],
        "ip_scale": float(_pick(segment.lynx_ip_scale, settings.lynx_ip_scale)),
        "ref_scale": float(_pick(segment.lynx_ref_scale, settings.lynx_ref_scale)),
        "lynx_cfg_scale": float(_pick(segment.lynx_cfg_scale, settings.lynx_lynx_cfg_scale)),
        "start_percent": float(_pick(segment.lynx_start_percent, settings.lynx_start_percent)),
        "end_percent": float(_pick(segment.lynx_end_percent, settings.lynx_end_percent)),
        "vae": ["620", 0], "lynx_ip_embeds": ["633", 0], "ref_image": ["631", 1],
        "ref_text_embed": ["622", 0]}}
    if ref_from_start_frame:
        # Second crop instance on the start frame; ref (appearance) tracks the shot while ip
        # (identity) stays anchored on the canonical subject photo.
        wf["636"] = {"class_type": "LynxInsightFaceCrop", "inputs": {"image": ["635", 0]}}
        wf["641"]["inputs"]["ref_image"] = ["636", 1]
    ref_blocks = _pick(segment.lynx_ref_blocks_to_use, settings.lynx_ref_blocks_to_use)
    if ref_blocks:
        wf["641"]["inputs"]["ref_blocks_to_use"] = ref_blocks

    # --- Dual-expert sampling, split at boundary -------------------------------
    # rope_function must be explicit: the widget default is "comfy" but the Python
    # signature default is "default", and only the comfy branches build the RoPE freqs
    # table — omitting it dies in rope_apply_3d. See TestSamplerWidgetDefaults.
    common = {
        "image_embeds": ["641", 0], "text_embeds": ["621", 0], "steps": steps, "cfg": cfg,
        "shift": shift, "seed": segment.seed, "force_offload": True, "scheduler": scheduler,
        "riflex_freq_index": 0, "rope_function": "comfy"}
    wf["650"] = {"class_type": "WanVideoSampler", "inputs": {
        "model": ["610", 0], **common, "start_step": 0, "end_step": boundary}}
    wf["651"] = {"class_type": "WanVideoSampler", "inputs": {
        "model": ["611", 0], **common, "samples": ["650", 0],
        "start_step": boundary, "end_step": -1}}

    wf["660"] = {"class_type": "WanVideoDecode", "inputs": {
        "vae": ["620", 0], "samples": ["651", 0], "enable_vae_tiling": False,
        "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}}

    rife_mult = gen["rife_multiplier"]
    final_images: list[Any] = ["660", 0]
    if rife_mult > 1:
        wf["670"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10,
            "multiplier": rife_mult, "fast_mode": True, "ensemble": True,
            "scale_factor": 1.0, "dtype": "float16", "torch_compile": False,
            "batch_size": 1, "frames": ["660", 0]}}
        final_images = ["670", 0]
    wf["680"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate": gen["output_fps"], "loop_count": 0, "filename_prefix": "output",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 15, "save_metadata": True,
        "trim_to_audio": False, "pingpong": False, "save_output": True,
        "images": final_images}}

    log_event(
        logger, "lynx_i2v.graph_built", str(segment.id),
        nodes=len(wf), num_frames=num_frames,
        resolution=f"{segment.width}x{segment.height}",
        arm=_lynx_arm(ip_layers), boundary=boundary, steps=steps,
    )
    return wf


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
        initial_reference_image_filename: The ComfyUI-local filename of the
            job's original input image for PainterLongVideo identity anchoring.
            When provided (segment > 0), node 98 is swapped from WanImageToVideo
            to PainterLongVideo with dual-reference inputs.
            identity for characters. PainterLongVideo only accepts a single
            clip_vision_output, so multi-frame reference frames are not used.
        previous_motion_magnitude: Measured motion magnitude from previous 
            segment (px/frame). Used to auto-adjust motion_amplitude for 
            consistent motion across segments.
    """
    gen = _calculate_generation_params(segment.fps, segment.duration_seconds, segment.speed)
    
    # Calculate motion_amplitude based on previous segment's motion
    motion_amplitude = _calculate_motion_amplitude(
        segment_index=segment.index,
        previous_motion_magnitude=previous_motion_magnitude,
        motion_amplitude_setting=settings.painter_motion_amplitude,
    )
    logger.info("Segment %d motion_amplitude: %.2f (previous_magnitude=%s)",
                segment.index, motion_amplitude, previous_motion_magnitude)
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
                "motion_frames": settings.painter_motion_frames,
                "motion_amplitude": motion_amplitude,
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
        _add_user_loras(workflow, [item.model_dump() for item in segment.loras])

    # Face swap (before RIFE so it processes only native WAN frames, not
    # interpolated ones — cuts faceswap frame count by 50-75%)
    faceswap = segment.faceswap_enabled and segment.faceswap_image
    if faceswap:
        _add_faceswap(workflow, segment)

    # RIFE frame interpolation — reads from faceswap output if enabled,
    # otherwise directly from VAEDecode
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
        "Built workflow: %dx%d, %d frames @ %dfps, RIFE %dx, speed=%.1fx, seed=%d, faceswap=%s",
        segment.width,
        segment.height,
        gen["wan_frames"],
        GENERATION_FPS,
        rife_multiplier,
        segment.speed,
        segment.seed,
        faceswap,
    )
    return workflow
