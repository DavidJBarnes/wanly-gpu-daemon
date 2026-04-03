"""Extract motion keywords from generated video for context propagation."""

import asyncio
import json
import subprocess
from pathlib import Path


MOTION_KEYWORDS = {
    "walking": ["walk", "step", "stride", "pace"],
    "running": ["run", "jog", "sprint", "dash"],
    "standing": ["stand", "still", "idle", "static", "stationary"],
    "sitting": ["sit", "seated", "chair"],
    "turning": ["turn", "rotate", "spin", "twist"],
    "dancing": ["dance", "move", "rhythm"],
    "talking": ["talk", "speak", "gesture", "mouth"],
    "falling": ["fall", "tumble", "drop"],
    "flying": ["fly", "soar", "float", "hover"],
    "driving": ["drive", "car", "vehicle", "steer"],
    "hand_wave": ["wave", "hand", "gesture", "arm"],
    "jumping": ["jump", "leap", "hop", "skip"],
}


def _parse_motion_from_prompt(prompt: str) -> list[str]:
    """Extract motion keywords from prompt text."""
    prompt_lower = prompt.lower()
    detected = []
    for motion_type, keywords in MOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                if motion_type not in detected:
                    detected.append(motion_type)
                break
    return detected


async def extract_motion_keywords(prompt: str, video_data: bytes | None = None) -> list[str]:
    """Extract motion keywords from prompt and optionally video analysis.
    
    Currently uses prompt-based extraction. Video analysis can be added later
    using optical flow or motion detection algorithms.
    
    Args:
        prompt: The generation prompt used
        video_data: Optional video bytes for analysis
        
    Returns:
        List of detected motion keywords
    """
    keywords = _parse_motion_from_prompt(prompt)
    
    if video_data and keywords:
        pass
    
    return keywords


def _augment_prompt_with_motion(original_prompt: str, previous_keywords: list[str]) -> str:
    """Augment prompt with motion continuity from previous segment.
    
    Args:
        original_prompt: The original generation prompt
        previous_keywords: Motion keywords from previous segment
        
    Returns:
        Augmented prompt with motion continuity hints
    """
    if not previous_keywords:
        return original_prompt
    
    motion_augmentations = {
        "walking": "continuing to walk forward steadily",
        "running": "continuing to run at a steady pace",
        "standing": "remaining still and stable",
        "sitting": "staying seated in the same position",
        "turning": "continuing the turning motion smoothly",
        "dancing": "continuing to dance with fluid movement",
        "talking": "continuing to speak with natural gestures",
        "falling": "continuing the descent motion",
        "flying": "continuing to float or glide through the air",
        "driving": "continuing to drive forward steadily",
        "hand_wave": "continuing with the waving gesture",
        "jumping": "continuing to jump with momentum",
    }
    
    augmentations = [motion_augmentations.get(kw, kw) for kw in previous_keywords[:2]]
    
    if augmentations:
        return f"{original_prompt}, {', '.join(augmentations)}"
    
    return original_prompt
