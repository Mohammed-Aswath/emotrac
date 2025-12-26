import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
FRAMES_CROPPED_DIR = DATA_DIR / "frames_cropped"
AU_RESULTS_DIR = DATA_DIR / "au_results"
MICRO_EVENTS_DIR = DATA_DIR / "micro_events"

VIDEO_CONFIG = {
    "fps_sample": 20,
    "max_frames": 30,
    "supported_formats": [".mp4", ".avi", ".mov"],
}

FACE_DETECTION_CONFIG = {
    "model_name": "yolov5s",
    "conf_threshold": 0.45,
    "face_size": 224,
    "use_mediapipe_fallback": True,
}

AU_EXTRACTION_CONFIG = {
    "n_jobs": 1,
}

MICRO_EXPRESSION_CONFIG = {
    "au_change_threshold": 5.0,
    "min_duration_frames": 2,
    "max_duration_frames": 15,
}

SCORING_CONFIG = {
    "negative_aus": [1, 2, 4, 5, 7, 15, 17, 23, 24, 25, 26],
    "positive_aus": [6, 12],
    "au_weight": 0.25,
    "emotion_weight": 0.60,
    "micro_expression_weight": 0.15,
}

RISK_BANDS = {
    "low": (0, 33),
    "medium": (34, 66),
    "high": (67, 100),
}

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
