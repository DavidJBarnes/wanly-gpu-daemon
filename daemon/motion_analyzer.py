"""Optical flow motion analysis for segment videos.

Measures the average motion magnitude in a video using OpenCV Farneback optical flow.
This is used to match motion speed across segments.
"""

import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def measure_motion_magnitude(video_data: bytes, fps: int = 15) -> Optional[float]:
    """Measure average optical flow magnitude in a video.

    Uses Farneback algorithm to compute dense optical flow between consecutive frames,
    then returns the average pixel displacement across all frame pairs.

    Args:
        video_data: Raw video bytes (mp4)
        fps: Frame rate to extract frames at (default 15 = native WAN generation rate)

    Returns:
        Average motion magnitude in pixels per frame, or None if measurement fails
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        tmp_video.write(video_data)
        tmp_video_path = tmp_video.name

    try:
        cap = cv2.VideoCapture(tmp_video_path)
        if not cap.isOpened():
            logger.warning("Could not open video for motion analysis")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 2:
            logger.warning("Video has fewer than 2 frames, cannot measure motion")
            return None

        magnitudes: list[float] = []

        ret, prev_frame = cap.read()
        if not ret:
            logger.warning("Could not read first frame")
            return None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = float(np.mean(magnitude))
            magnitudes.append(avg_magnitude)

            prev_gray = curr_gray

        cap.release()

        if magnitudes:
            avg_motion = float(np.mean(magnitudes))
            logger.info(
                "Motion analysis: %.2f px/frame (avg over %d frame pairs)",
                avg_motion,
                len(magnitudes),
            )
            return avg_motion

        logger.warning("No motion data extracted from video")
        return None

    except Exception as e:
        logger.warning("Motion analysis failed: %s", e)
        return None
    finally:
        try:
            os.unlink(tmp_video_path)
        except OSError:
            pass


def estimate_motion_from_flow(
    previous_motion_magnitude: Optional[float],
    previous_motion_amplitude: float,
    target_motion_magnitude: float,
) -> float:
    """Estimate motion_amplitude needed to achieve target motion.

    This is a simple linear estimation. The relationship between motion_amplitude
    and output motion magnitude is not perfectly linear, but this provides a starting
    point that can be refined empirically.

    Args:
        previous_motion_magnitude: Measured motion from previous segment (px/frame)
        previous_motion_amplitude: motion_amplitude used for previous segment
        target_motion_magnitude: Target motion magnitude to match

    Returns:
        Estimated motion_amplitude to achieve target motion
    """
    if previous_motion_magnitude is None or previous_motion_magnitude <= 0:
        return 1.0

    baseline_motion = previous_motion_magnitude / previous_motion_amplitude

    if baseline_motion <= 0:
        return 1.0

    estimated = target_motion_magnitude / baseline_motion

    return max(0.5, min(2.0, estimated))
