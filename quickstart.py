#!/usr/bin/env python
"""
Quick start script for EmoTrace pipeline.
Tests all components without Streamlit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from emotrace_utils.logger import get_logger
from emotrace_utils.config import (
    VIDEO_CONFIG, FACE_DETECTION_CONFIG,
    AU_EXTRACTION_CONFIG, MICRO_EXPRESSION_CONFIG
)

logger = get_logger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import cv2
        logger.info("✓ OpenCV imported")
    except ImportError as e:
        logger.error(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        logger.info("✓ PyTorch imported")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas
        logger.info("✓ Pandas imported")
    except ImportError as e:
        logger.error(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        logger.info("✓ NumPy imported")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        logger.info("✓ Matplotlib imported")
    except ImportError as e:
        logger.error(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import streamlit
        logger.info("✓ Streamlit imported")
    except ImportError as e:
        logger.error(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        from video.extract_frames import extract_frames
        logger.info("✓ Video module imported")
    except ImportError as e:
        logger.error(f"✗ Video module import failed: {e}")
        return False
    
    try:
        from face.yolo_face_detector import YOLOFaceDetector
        logger.info("✓ Face detection module imported")
    except ImportError as e:
        logger.error(f"✗ Face detection module import failed: {e}")
        return False
    
    try:
        from features.au_extractor import AUExtractor
        logger.info("✓ AU extraction module imported")
    except ImportError as e:
        logger.error(f"✗ AU extraction module import failed: {e}")
        return False
    
    try:
        from features.micro_expression import MicroExpressionDetector
        logger.info("✓ Micro-expression module imported")
    except ImportError as e:
        logger.error(f"✗ Micro-expression module import failed: {e}")
        return False
    
    try:
        from scoring.feature_engineering import FeatureEngineer
        logger.info("✓ Feature engineering module imported")
    except ImportError as e:
        logger.error(f"✗ Feature engineering module import failed: {e}")
        return False
    
    try:
        from scoring.depression_screener import DepressionScreener
        logger.info("✓ Depression screener module imported")
    except ImportError as e:
        logger.error(f"✗ Depression screener module import failed: {e}")
        return False
    
    try:
        from scoring.recommendation import RecommendationEngine
        logger.info("✓ Recommendation module imported")
    except ImportError as e:
        logger.error(f"✗ Recommendation module import failed: {e}")
        return False
    
    try:
        from visualization.plots import (
            plot_au_trajectory,
            plot_emotion_distribution,
            plot_micro_expressions
        )
        logger.info("✓ Visualization module imported")
    except ImportError as e:
        logger.error(f"✗ Visualization module import failed: {e}")
        return False
    
    return True


def print_config():
    """Print configuration settings."""
    logger.info("\nConfiguration Settings:")
    logger.info(f"  VIDEO_CONFIG: {VIDEO_CONFIG}")
    logger.info(f"  FACE_DETECTION_CONFIG: {FACE_DETECTION_CONFIG}")
    logger.info(f"  AU_EXTRACTION_CONFIG: {AU_EXTRACTION_CONFIG}")
    logger.info(f"  MICRO_EXPRESSION_CONFIG: {MICRO_EXPRESSION_CONFIG}")


def print_usage():
    """Print usage instructions."""
    logger.info("\n" + "="*60)
    logger.info("EmoTrace Quick Start")
    logger.info("="*60)
    logger.info("\nUsage:")
    logger.info("  1. Install dependencies: pip install -r requirements.txt")
    logger.info("  2. Run Streamlit app: streamlit run app.py")
    logger.info("  3. Upload a .mp4 video and click 'Run Analysis'")
    logger.info("\nOr use the pipeline programmatically:")
    logger.info("  from run_pipeline import run_analysis_pipeline")
    logger.info("  result = run_analysis_pipeline('video.mp4')")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    logger.info("EmoTrace System Check")
    logger.info("-" * 60)
    
    if test_imports():
        logger.info("\n✓ All modules imported successfully!")
        print_config()
        print_usage()
        logger.info("System ready. Run: streamlit run app.py")
    else:
        logger.error("\n✗ Some modules failed to import.")
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)
