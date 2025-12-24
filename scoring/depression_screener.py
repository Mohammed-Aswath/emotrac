import numpy as np
from typing import Dict, Tuple
from emotrace_utils.config import SCORING_CONFIG, RISK_BANDS
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class DepressionScreener:
    """Compute depression risk score from features."""
    
    def __init__(self):
        self.au_weight = SCORING_CONFIG["au_weight"]
        self.emotion_weight = SCORING_CONFIG["emotion_weight"]
        self.micro_expression_weight = SCORING_CONFIG["micro_expression_weight"]
    
    def _normalize_0_100(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize value to 0-100 range."""
        if max_val <= min_val:
            return 50.0
        
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        
        return float(normalized * 100.0)
    
    def compute_au_risk_component(self, features: Dict[str, float]) -> float:
        """
        Compute AU-based risk component.
        Higher negative AU and lower positive AU indicate higher risk.
        """
        negative_au_mean = features.get("negative_au_mean", 0.0)
        positive_au_mean = features.get("positive_au_mean", 0.0)
        
        negative_risk = self._normalize_0_100(negative_au_mean, 0.0, 100.0)
        positive_benefit = self._normalize_0_100(positive_au_mean, 0.0, 100.0)
        
        au_risk = (negative_risk * 0.6 + (100.0 - positive_benefit) * 0.4)
        
        return float(au_risk)
    
    def compute_emotion_risk_component(self, features: Dict[str, float]) -> float:
        """
        Compute emotion-based risk component.
        Higher negative emotions indicate higher risk.
        """
        negative_emotion_ratio = features.get("negative_emotion_ratio", 0.0)
        joy_mean = features.get("joy_mean", 0.0)
        sadness_mean = features.get("sadness_mean", 0.0)
        
        emotion_risk = negative_emotion_ratio * 60.0 + sadness_mean * 30.0 - joy_mean * 10.0
        emotion_risk = max(0.0, min(100.0, emotion_risk))
        
        return float(emotion_risk)
    
    def compute_micro_expression_risk_component(self, features: Dict[str, float]) -> float:
        """
        Compute micro-expression-based risk component.
        Higher micro-expression activity (count and intensity) indicates higher risk.
        """
        micro_count = features.get("micro_expression_count", 0)
        mean_intensity = features.get("mean_intensity", 0.0)
        
        count_risk = min(100.0, micro_count * 10.0)
        intensity_risk = self._normalize_0_100(mean_intensity, 0.0, 100.0)
        
        micro_risk = (count_risk * 0.5 + intensity_risk * 0.5)
        
        return float(micro_risk)
    
    def compute_risk_score(self, features: Dict[str, float]) -> float:
        """
        Compute overall depression risk score (0-100).
        """
        logger.info("Computing depression risk score")
        
        au_component = self.compute_au_risk_component(features)
        emotion_component = self.compute_emotion_risk_component(features)
        micro_component = self.compute_micro_expression_risk_component(features)
        
        risk_score = (
            au_component * self.au_weight +
            emotion_component * self.emotion_weight +
            micro_component * self.micro_expression_weight
        )
        
        risk_score = max(0.0, min(100.0, risk_score))
        
        logger.info(f"Risk score: {risk_score:.2f} (AU: {au_component:.2f}, Emotion: {emotion_component:.2f}, Micro: {micro_component:.2f})")
        
        return float(risk_score)
    
    def get_risk_band(self, risk_score: float) -> str:
        """Classify risk score into band."""
        if risk_score <= RISK_BANDS["low"][1]:
            return "low"
        elif risk_score <= RISK_BANDS["medium"][1]:
            return "medium"
        else:
            return "high"
