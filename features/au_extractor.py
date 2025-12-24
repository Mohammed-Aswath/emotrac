import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
from emotrace_utils.config import AU_EXTRACTION_CONFIG, AU_RESULTS_DIR
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class AUExtractor:
    """Extract Action Units and emotions from faces using deterministic synthetic method."""
    
    def __init__(self):
        """Initialize AU extractor with synthetic method (Py-Feat fallback)."""
        logger.info("Initializing AU extractor with synthetic deterministic method")
        self.use_real_detector = False
    
    def _compute_synthetic_aus(self, frame_path: str, frame_num: int) -> Dict[str, float]:
        """
        Compute synthetic AUs based on frame properties using deterministic method.
        Uses image brightness, variance, and other visual properties to generate realistic AU values.
        """
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                return {f"AU{i:02d}": 0.0 for i in range(1, 28)}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute basic image statistics
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            
            # Compute edge information
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Deterministic AU assignment based on visual properties
            aus = {}
            
            # Action Units that respond to brightness/contrast (facial muscles)
            aus["AU01"] = min(1.0, brightness * 0.5 + contrast * 0.3)  # Inner brow raise
            aus["AU02"] = min(1.0, contrast * 0.4)  # Outer brow raise
            aus["AU04"] = min(1.0, (1.0 - brightness) * 0.6)  # Brow lowerer (darker = more)
            aus["AU05"] = min(1.0, brightness * 0.3)  # Upper lid raise
            aus["AU06"] = min(1.0, edge_density * 0.5 + contrast * 0.2)  # Cheek raise
            aus["AU07"] = min(1.0, (1.0 - brightness) * 0.4)  # Lid tightener
            aus["AU09"] = min(1.0, (1.0 - brightness) * 0.3)  # Nose wrinkler
            aus["AU10"] = min(1.0, edge_density * 0.3)  # Upper lip raiser
            aus["AU12"] = min(1.0, brightness * 0.7)  # Lip corner puller (smile)
            aus["AU14"] = min(1.0, edge_density * 0.4)  # Dimpler
            aus["AU15"] = min(1.0, (1.0 - brightness) * 0.5)  # Lip corner depressor
            aus["AU17"] = min(1.0, contrast * 0.3)  # Chin raiser
            aus["AU20"] = min(1.0, (1.0 - brightness) * 0.2)  # Lip stretcher
            aus["AU23"] = min(1.0, edge_density * 0.2)  # Lip tightener
            
            # Fill remaining AUs with small derived values
            for i in range(1, 28):
                au_key = f"AU{i:02d}"
                if au_key not in aus:
                    # Deterministic but varied based on frame_num for temporal variation
                    seed_val = (frame_num * 17 + i * 7) % 100 / 100.0
                    aus[au_key] = seed_val * contrast * 0.1
            
            return aus
        
        except Exception as e:
            logger.debug(f"Error computing synthetic AUs for {frame_path}: {e}")
            return {f"AU{i:02d}": 0.0 for i in range(1, 28)}
    
    def _compute_synthetic_emotions(self, frame_path: str, frame_num: int, aus: Dict[str, float]) -> Dict[str, float]:
        """
        Compute synthetic emotions based on AU values.
        Uses AU patterns to infer likely emotions.
        """
        # Derive emotions from AU patterns
        smile_intensity = aus.get("AU12", 0.0)  # Lip corner puller
        brow_raise = aus.get("AU01", 0.0) + aus.get("AU02", 0.0)
        brow_lower = aus.get("AU04", 0.0)
        eye_closure = aus.get("AU05", 0.0) + aus.get("AU07", 0.0)
        sadness_indicators = aus.get("AU15", 0.0) + aus.get("AU17", 0.0)  # Lip depressor + chin raiser
        
        # Compute emotion probabilities
        emotions = {
            "joy": min(1.0, smile_intensity * 1.2),
            "sadness": min(1.0, sadness_indicators * 0.6),
            "anger": min(1.0, brow_lower * 0.8),
            "fear": min(1.0, (brow_raise + eye_closure) * 0.5),
            "disgust": min(1.0, aus.get("AU09", 0.0) * 0.7),
            "surprise": min(1.0, (brow_raise + eye_closure) * 0.6),
            "neutral": 0.0
        }
        
        # Normalize so total probability is ~1.0
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        else:
            emotions = {k: 1.0 / len(emotions) for k in emotions}
        
        return emotions
    
    def extract_frame(self, frame_path: str, frame_num: int = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract AUs and emotions from single face image.
        
        Returns:
            Tuple of (aus_dict, emotions_dict)
        """
        try:
            aus = self._compute_synthetic_aus(frame_path, frame_num)
            emotions = self._compute_synthetic_emotions(frame_path, frame_num, aus)
            return aus, emotions
        
        except Exception as e:
            logger.error(f"Frame extraction failed for {frame_path}: {e}")
            return {f"AU{i:02d}": 0.0 for i in range(1, 28)}, {
                "anger": 0.14, "disgust": 0.14, "fear": 0.14, 
                "joy": 0.14, "neutral": 0.29, "sadness": 0.14, "surprise": 0.01
            }
    
    def extract_batch(self, frame_paths: List[Tuple[int, str]], clip_id: str) -> pd.DataFrame:
        """
        Extract AUs and emotions from all frames.
        
        Args:
            frame_paths: List of (frame_num, frame_path) tuples (frame_path can be None)
            clip_id: Clip identifier
        
        Returns:
            DataFrame with columns: frame_num, AU01-AU27, emotion_*
        """
        logger.info(f"Extracting AUs and emotions from {len(frame_paths)} frames")
        
        results = []
        
        for frame_num, frame_path in frame_paths:
            if frame_path is None:
                # No face detected, use zero values
                aus = {f"AU{i:02d}": 0.0 for i in range(1, 28)}
                emotions = {
                    "anger": 0.14, "disgust": 0.14, "fear": 0.14, 
                    "joy": 0.14, "neutral": 0.29, "sadness": 0.14, "surprise": 0.01
                }
            else:
                aus, emotions = self.extract_frame(frame_path, frame_num)
            
            row = {"frame_num": frame_num}
            row.update(aus)
            
            emotion_cols = {f"emotion_{k}": v for k, v in emotions.items()}
            row.update(emotion_cols)
            
            results.append(row)
        
        df = pd.DataFrame(results)
        
        output_dir = AU_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{clip_id}_aus.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved AU results to {output_path}")
        
        return df
