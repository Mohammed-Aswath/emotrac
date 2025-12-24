import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from emotrace_utils.config import SCORING_CONFIG
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Compute features for depression risk scoring."""
    
    def __init__(self):
        self.negative_aus = SCORING_CONFIG["negative_aus"]
        self.positive_aus = SCORING_CONFIG["positive_aus"]
    
    def _get_au_columns(self, df: pd.DataFrame) -> List[str]:
        """Get AU columns from dataframe."""
        return [col for col in df.columns if col.startswith('AU')]
    
    def _get_emotion_columns(self, df: pd.DataFrame) -> List[str]:
        """Get emotion columns from dataframe."""
        return [col for col in df.columns if col.startswith('emotion_')]
    
    def _extract_au_number(self, au_col: str) -> int:
        """Extract AU number from column name."""
        return int(au_col.replace('AU', ''))
    
    def compute_au_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute AU statistics."""
        au_cols = self._get_au_columns(df)
        
        if not au_cols:
            return {
                "negative_au_mean": 0.0,
                "negative_au_std": 0.0,
                "positive_au_mean": 0.0,
                "negative_au_ratio": 0.0
            }
        
        au_data = df[au_cols].astype(float)
        
        negative_aus_in_data = []
        positive_aus_in_data = []
        
        for col in au_cols:
            au_num = self._extract_au_number(col)
            if au_num in self.negative_aus:
                negative_aus_in_data.append(col)
            elif au_num in self.positive_aus:
                positive_aus_in_data.append(col)
        
        negative_au_mean = au_data[negative_aus_in_data].mean().mean() if negative_aus_in_data else 0.0
        negative_au_std = au_data[negative_aus_in_data].std().mean() if negative_aus_in_data else 0.0
        positive_au_mean = au_data[positive_aus_in_data].mean().mean() if positive_aus_in_data else 0.0
        
        total_mean = au_data.mean().mean()
        negative_au_ratio = negative_au_mean / (total_mean + 1e-6)
        
        return {
            "negative_au_mean": float(negative_au_mean),
            "negative_au_std": float(negative_au_std),
            "positive_au_mean": float(positive_au_mean),
            "negative_au_ratio": float(negative_au_ratio)
        }
    
    def compute_emotion_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute emotion statistics."""
        emotion_cols = self._get_emotion_columns(df)
        
        if not emotion_cols:
            return {
                "sadness_mean": 0.0,
                "anger_mean": 0.0,
                "fear_mean": 0.0,
                "disgust_mean": 0.0,
                "joy_mean": 0.0,
                "neutral_mean": 0.0,
                "surprise_mean": 0.0,
                "negative_emotion_ratio": 0.0
            }
        
        emotion_data = df[emotion_cols].astype(float)
        
        emotion_mapping = {
            "emotion_sadness": "sadness_mean",
            "emotion_anger": "anger_mean",
            "emotion_fear": "fear_mean",
            "emotion_disgust": "disgust_mean",
            "emotion_joy": "joy_mean",
            "emotion_neutral": "neutral_mean",
            "emotion_surprise": "surprise_mean"
        }
        
        stats = {}
        for col, stat_name in emotion_mapping.items():
            if col in emotion_data.columns:
                stats[stat_name] = float(emotion_data[col].mean())
            else:
                stats[stat_name] = 0.0
        
        negative_emotions = stats.get("sadness_mean", 0.0) + stats.get("anger_mean", 0.0) + stats.get("fear_mean", 0.0) + stats.get("disgust_mean", 0.0)
        positive_emotions = stats.get("joy_mean", 0.0)
        total_emotion = negative_emotions + positive_emotions + stats.get("neutral_mean", 0.0)
        
        negative_emotion_ratio = negative_emotions / (total_emotion + 1e-6)
        stats["negative_emotion_ratio"] = float(negative_emotion_ratio)
        
        return stats
    
    def compute_micro_expression_stats(self, df_events: pd.DataFrame) -> Dict[str, float]:
        """Compute micro-expression statistics."""
        if df_events.empty:
            return {
                "micro_expression_count": 0,
                "mean_intensity": 0.0,
                "mean_duration": 0.0
            }
        
        count = len(df_events)
        mean_intensity = float(df_events["peak_intensity"].mean()) if "peak_intensity" in df_events.columns else 0.0
        mean_duration = float(df_events["duration_frames"].mean()) if "duration_frames" in df_events.columns else 0.0
        
        return {
            "micro_expression_count": int(count),
            "mean_intensity": mean_intensity,
            "mean_duration": mean_duration
        }
    
    def engineer_all_features(
        self, 
        df_aus: pd.DataFrame, 
        df_events: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute all features."""
        logger.info("Engineering features from extracted data")
        
        features = {}
        
        au_stats = self.compute_au_stats(df_aus)
        features.update(au_stats)
        
        emotion_stats = self.compute_emotion_stats(df_aus)
        features.update(emotion_stats)
        
        micro_stats = self.compute_micro_expression_stats(df_events)
        features.update(micro_stats)
        
        logger.info(f"Computed {len(features)} features")
        
        return features
