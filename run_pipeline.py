import os
import uuid
import cv2
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from emotrace_utils.logger import get_logger
from emotrace_utils.config import (
    RAW_VIDEOS_DIR, 
    FRAMES_DIR, 
    AU_RESULTS_DIR, 
    MICRO_EVENTS_DIR
)
from video.extract_frames import extract_frames
from face.yolo_face_detector import YOLOFaceDetector
from features.au_extractor import AUExtractor
from features.micro_expression import MicroExpressionDetector
from scoring.feature_engineering import FeatureEngineer
from scoring.depression_screener import DepressionScreener
from scoring.recommendation import RecommendationEngine
from visualization.plots import plot_au_trajectory, plot_emotion_distribution, plot_micro_expressions

logger = get_logger(__name__)


def create_visualizations(au_df, events_df, features):
    """Create all visualizations and return as dict."""
    plots = {}
    
    try:
        if not au_df.empty:
            fig, ax = plot_au_trajectory(au_df)
            plots['au_plot'] = fig
    except Exception as e:
        logger.warning(f"Failed to create AU plot: {str(e)}")
    
    try:
        if not au_df.empty:
            fig, ax = plot_emotion_distribution(au_df)
            plots['emotion_plot'] = fig
    except Exception as e:
        logger.warning(f"Failed to create emotion plot: {str(e)}")
    
    try:
        if not events_df.empty:
            total_frames = len(au_df) if not au_df.empty else 30
            fig, ax = plot_micro_expressions(events_df, total_frames)
            plots['micro_plot'] = fig
    except Exception as e:
        logger.warning(f"Failed to create micro-expression plot: {str(e)}")
    
    return plots


def run_analysis_pipeline(
    video_path: str,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Run complete analysis pipeline with progress tracking.
    
    Args:
        video_path: Path to input video
        progress_callback: Function to call with progress updates (step, message, percent)
    
    Returns:
        Dictionary with analysis results
    """
    clip_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting analysis pipeline for clip: {clip_id}")
    
    try:
        # Step 1: Extract frames
        _progress(progress_callback, 1, "Extracting frames from video...", 0)
        logger.info("Step 1: Extracting frames from video")
        
        frame_paths = extract_frames(video_path, clip_id)
        num_frames = len(frame_paths)
        
        logger.info(f"Extracted {num_frames} frames")
        _progress(progress_callback, 1, f"Extracted {num_frames} frames", 100)
        
        # Step 2: Detect faces
        _progress(progress_callback, 2, f"Detecting faces in {num_frames} frames...", 0)
        logger.info(f"Step 2: Detecting faces in {num_frames} frames")
        
        face_detector = YOLOFaceDetector()
        detected_face_paths = []
        
        for i, (frame_num, frame_path) in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is not None:
                face_path = face_detector.process_frame(frame, clip_id, i)
                detected_face_paths.append((frame_num, face_path) if face_path else (frame_num, None))
            else:
                detected_face_paths.append((frame_num, None))
            
            progress_pct = int((i + 1) / num_frames * 100)
            _progress(progress_callback, 2, f"Detecting faces: {i+1}/{num_frames} frames", progress_pct)
        
        faces_detected = sum(1 for _, p in detected_face_paths if p is not None)
        logger.info(f"Detected faces in {faces_detected} frames")
        _progress(progress_callback, 2, f"Detected faces in {faces_detected} frames", 100)
        
        # Step 3: Extract Action Units
        _progress(progress_callback, 3, "Extracting facial action units...", 0)
        logger.info("Step 3: Extracting Action Units and emotions")
        
        au_extractor = AUExtractor()
        au_df = au_extractor.extract_batch(detected_face_paths, clip_id)
        
        logger.info(f"Extracted AUs from {len(au_df)} frames")
        _progress(progress_callback, 3, f"Extracted AUs from {len(au_df)} frames", 100)
        
        # Step 4: Detect micro-expressions
        _progress(progress_callback, 4, "Detecting micro-expressions...", 0)
        logger.info("Step 4: Detecting micro-expressions")
        
        micro_detector = MicroExpressionDetector()
        events = micro_detector.detect_rapid_changes(au_df)
        events_df = micro_detector.save_events(events, clip_id)
        
        logger.info(f"Detected {len(events_df)} micro-expression events")
        _progress(progress_callback, 4, f"Detected {len(events_df)} micro-expression events", 100)
        
        # Step 5: Feature engineering
        _progress(progress_callback, 5, "Computing features...", 0)
        logger.info("Step 5: Feature engineering")
        
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_all_features(au_df, events_df)
        
        logger.info(f"Computed {len(features)} features")
        _progress(progress_callback, 5, f"Computed {len(features)} features", 100)
        
        # Step 6: Depression risk scoring
        _progress(progress_callback, 6, "Calculating depression risk score...", 0)
        logger.info("Step 6: Depression risk scoring")
        
        screener = DepressionScreener()
        risk_score = screener.compute_risk_score(features)
        risk_band = screener.get_risk_band(risk_score)
        components = {
            'au_component': screener.compute_au_risk_component(features),
            'emotion_component': screener.compute_emotion_risk_component(features),
            'micro_component': screener.compute_micro_expression_risk_component(features)
        }
        
        logger.info(f"Risk score: {risk_score:.1f}/100 ({risk_band})")
        _progress(progress_callback, 6, f"Risk score: {risk_score:.1f}/100", 100)
        
        # Step 7: Generate recommendations
        _progress(progress_callback, 7, "Generating recommendations...", 0)
        logger.info("Step 7: Generating recommendations")
        
        rec_engine = RecommendationEngine()
        recommendation_dict = rec_engine.generate_recommendation(risk_score, risk_band, features)
        recommendation = recommendation_dict.get('recommendation', 'No recommendation available')
        
        logger.info("Recommendation generated")
        _progress(progress_callback, 7, "Recommendation generated", 100)
        
        # Step 8: Create visualizations
        _progress(progress_callback, 8, "Creating visualizations...", 0)
        logger.info("Step 8: Creating visualizations")
        
        plots = create_visualizations(au_df, events_df, features)
        
        logger.info("Visualizations created")
        _progress(progress_callback, 8, "Visualizations created", 100)
        
        # Save results
        _progress(progress_callback, 9, "Saving results...", 0)
        logger.info("Step 9: Saving results")
        
        AU_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        au_csv_path = AU_RESULTS_DIR / f"{clip_id}_aus.csv"
        au_df.to_csv(au_csv_path, index=False)
        
        MICRO_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        events_csv_path = MICRO_EVENTS_DIR / f"{clip_id}_events.csv"
        events_df.to_csv(events_csv_path, index=False)
        
        logger.info("Results saved")
        _progress(progress_callback, 9, "Results saved", 100)
        
        result = {
            'status': 'success',
            'clip_id': clip_id,
            'num_frames': num_frames,
            'faces_detected': faces_detected,
            'risk_score': risk_score,
            'risk_band': risk_band,
            'components': components,
            'features': features,
            'au_df': au_df,
            'events_df': events_df,
            'recommendation': recommendation,
            'plots': plots,
            'au_csv': str(au_csv_path),
            'events_csv': str(events_csv_path)
        }
        
        logger.info(f"Pipeline completed successfully. Risk Score: {risk_score:.1f}")
        _progress(progress_callback, 10, "✅ Analysis completed successfully!", 100)
        
        return result
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Pipeline failed: {error_msg}")
        _progress(progress_callback, 0, f"❌ Analysis failed: {error_msg}", 0)
        
        return {
            'status': 'error',
            'error': error_msg,
            'clip_id': clip_id
        }


def _progress(callback, step: int, message: str, percent: int):
    """Helper to send progress updates to Streamlit."""
    if callback is not None:
        try:
            callback(step, message, percent)
        except Exception as e:
            logger.warning(f"Progress callback failed: {str(e)}")
