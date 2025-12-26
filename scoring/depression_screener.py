import numpy as np
from typing import Dict, Tuple
from emotrace_utils.config import SCORING_CONFIG, RISK_BANDS
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class DepressionScreener:
    """Compute depression risk score from features."""
    
    def __init__(self):
        # Override config weights for MAXIMUM emotion sensitivity
        # Emotion is 92% (we detect this VERY well with DeepFace - 89%+ accuracy)
        # AU is 4% (synthetic fallback, less reliable)
        # Micro is 4% (requires consistent video quality)
        self.au_weight = 0.04
        self.emotion_weight = 0.92
        self.micro_expression_weight = 0.04
    
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
        Higher negative AU (sad/angry muscles) and lower positive AU (smile muscles) = higher risk.
        """
        negative_au_mean = features.get("negative_au_mean", 0.0)
        positive_au_mean = features.get("positive_au_mean", 0.0)
        negative_au_ratio = features.get("negative_au_ratio", 0.0)
        
        # Convert to 0-100 scale for negative AUs
        negative_risk = self._normalize_0_100(negative_au_mean, 0.0, 1.0)
        
        # Lack of positive AUs also indicates risk
        positive_deficit = 100.0 - self._normalize_0_100(positive_au_mean, 0.0, 1.0)
        
        # Ratio of negative to positive AU is strong indicator
        ratio_risk = self._normalize_0_100(negative_au_ratio, 0.0, 2.0)
        
        # Combine: negative activation is 40%, deficit of positive is 30%, ratio is 30%
        au_risk = (negative_risk * 0.40) + (positive_deficit * 0.30) + (ratio_risk * 0.30)
        
        logger.info(f"AU component: negative_risk={negative_risk:.1f}, positive_deficit={positive_deficit:.1f}, ratio_risk={ratio_risk:.1f} → Total={au_risk:.2f}")
        
        return float(au_risk)
    
    def compute_emotion_risk_component(self, features: Dict[str, float]) -> float:
        """
        Compute emotion-based risk component with EXPONENTIAL weighting.
        The DOMINANT (most prominent) emotion gets EXPONENTIALLY more weight.
        Lower secondary emotions are de-emphasized exponentially.
        """
        sadness_mean = features.get("sadness_mean", 0.0)
        anger_mean = features.get("anger_mean", 0.0)
        fear_mean = features.get("fear_mean", 0.0)
        disgust_mean = features.get("disgust_mean", 0.0)
        negative_emotion_ratio = features.get("negative_emotion_ratio", 0.0)
        joy_mean = features.get("joy_mean", 0.0)
        
        # Step 1: Find the DOMINANT emotion (the one with highest mean)
        emotion_values = {
            "sadness": sadness_mean,
            "anger": anger_mean,
            "fear": fear_mean,
            "disgust": disgust_mean,
            "joy": joy_mean
        }
        
        # Sort emotions by intensity (descending)
        sorted_emotions = sorted(emotion_values.items(), key=lambda x: x[1], reverse=True)
        dominant_emotion = sorted_emotions[0][0]
        dominant_intensity = sorted_emotions[0][1]
        
        # Step 2: EXPONENTIAL weighting for emotions
        # Top emotion: weight^1.0 (full strength)
        # 2nd emotion: weight^0.2 (extremely reduced)
        # 3rd emotion: weight^0.08 (nearly zero)
        # 4th emotion: weight^0.03 (almost nothing)
        # Lower: weight^0.01 (negligible)
        
        emotion_risk_components = []
        exponential_factors = [1.0, 0.2, 0.08, 0.03, 0.01]
        
        for idx, (emotion_name, intensity) in enumerate(sorted_emotions):
            if emotion_name == "joy":
                continue  # Joy is protective, handle separately
            
            # Get exponential factor for this position
            exp_factor = exponential_factors[min(idx, len(exponential_factors)-1)]
            
            # Apply exponential reduction: multiply intensity by exp_factor
            weighted_intensity = (intensity ** exp_factor) * 100.0
            
            # Risk contribution varies by emotion type
            if emotion_name == "sadness":
                risk_from_emotion = weighted_intensity * 0.40  # Sadness: 40% weight
            elif emotion_name == "anger":
                risk_from_emotion = weighted_intensity * 0.35  # Anger: 35% weight
            elif emotion_name == "fear":
                risk_from_emotion = weighted_intensity * 0.25  # Fear: 25% weight
            elif emotion_name == "disgust":
                risk_from_emotion = weighted_intensity * 0.15  # Disgust: 15% weight
            else:
                risk_from_emotion = weighted_intensity * 0.10
            
            emotion_risk_components.append((emotion_name, risk_from_emotion))
        
        # Sum all emotion components
        individual_risk = sum(risk for _, risk in emotion_risk_components)
        
        # Step 3: Base risk from negative emotion ratio
        ratio_based_risk = min(100.0, negative_emotion_ratio * 100.0)
        
        # Step 4: DOMINANT emotion exponential boost
        # If dominant emotion is strong, boost it further with exponential scaling
        dominant_bonus = 0.0
        if dominant_intensity > 0.20:
            # Exponential boost: (intensity^1.8) gives EXTREMELY strong weight to high values
            dominant_boost_factor = (dominant_intensity ** 1.8) * 100.0
            if dominant_emotion == "sadness":
                dominant_bonus = dominant_boost_factor * 1.00  # Sadness gets 100% boost (MAXIMUM!)
            elif dominant_emotion == "anger":
                dominant_bonus = dominant_boost_factor * 0.98  # Anger gets 98% boost
            else:
                dominant_bonus = dominant_boost_factor * 0.85
        
        # Step 5: Joy is protective (reduces risk)
        joy_protection = (joy_mean ** 1.0) * 100.0 * 0.10  # Linear for joy (not exponential)
        
        # Combine all components
        # Ratio: 5%, Individual emotions: 5%, Dominant exponential bonus: 90% (EXTREMELY HIGH!)
        emotion_risk = (ratio_based_risk * 0.05) + (individual_risk * 0.05) + (dominant_bonus * 0.90) - joy_protection
        
        emotion_risk = max(0.0, min(100.0, emotion_risk))
        
        component_breakdown = ", ".join([f"{name}={risk:.1f}" for name, risk in emotion_risk_components])
        logger.info(f"Emotion: dominant={dominant_emotion}({dominant_intensity*100:.1f}%), exponential_components=[{component_breakdown}], ratio={ratio_based_risk:.1f}, bonus={dominant_bonus:.1f}, joy_prot={joy_protection:.1f} → Total={emotion_risk:.2f}")
        
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
