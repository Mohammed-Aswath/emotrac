import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from emotrace_utils.config import VIDEO_CONFIG, FRAMES_DIR
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


def extract_frames(
    video_path: str, 
    clip_id: str, 
    fps_sample: int = VIDEO_CONFIG["fps_sample"],
    max_frames: int = VIDEO_CONFIG["max_frames"]
) -> List[Tuple[int, str]]:
    """
    Extract frames from video at specified sampling rate.
    
    Returns:
        List of tuples (frame_number, frame_path)
    """
    logger.info(f"Extracting frames from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps_original == 0:
        fps_original = 30.0
    
    sample_interval = max(1, int(fps_original / fps_sample))
    
    output_dir = FRAMES_DIR / clip_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    processed_frames = 0
    
    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % sample_interval == 0:
            frame_path = output_dir / f"frame_{processed_frames:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append((processed_frames, str(frame_path)))
            processed_frames += 1
            logger.debug(f"Saved frame {processed_frames} to {frame_path}")
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frame_paths)} frames from video to {output_dir}")
    
    return frame_paths
