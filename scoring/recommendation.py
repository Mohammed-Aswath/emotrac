from typing import Dict
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


DISCLAIMER = """
IMPORTANT DISCLAIMER:
This is a research prototype for non-diagnostic facial expression analysis.
This tool is NOT designed for medical diagnosis and cannot replace professional mental health assessment.
Results should not be used for any clinical decision-making.
For mental health concerns, always consult with qualified healthcare professionals.
"""


class RecommendationEngine:
    """Generate recommendations based on risk score and features."""
    
    def __init__(self):
        pass
    
    def generate_recommendation(self, risk_score: float, risk_band: str, features: Dict[str, float]) -> Dict[str, str]:
        """
        Generate human-readable recommendation.
        
        Returns:
            Dict with 'recommendation', 'disclaimer', and 'next_steps'
        """
        logger.info(f"Generating recommendation for risk band: {risk_band}")
        
        base_recommendation = self._get_base_recommendation(risk_band, risk_score)
        next_steps = self._get_next_steps(risk_band, features)
        
        result = {
            "recommendation": base_recommendation,
            "risk_band": risk_band,
            "risk_score": f"{risk_score:.1f}",
            "next_steps": next_steps,
            "disclaimer": DISCLAIMER
        }
        
        return result
    
    def _get_base_recommendation(self, risk_band: str, risk_score: float) -> str:
        """Get base recommendation based on risk band."""
        if risk_band == "low":
            return (
                f"Risk Score: {risk_score:.1f}/100 (LOW RISK)\n"
                "Facial expression analysis shows minimal indicators associated with depression risk.\n"
                "Continue maintaining healthy emotional and physical habits."
            )
        
        elif risk_band == "medium":
            return (
                f"Risk Score: {risk_score:.1f}/100 (MEDIUM RISK)\n"
                "Facial expression patterns show moderate indicators that warrant attention.\n"
                "Consider proactive mental health monitoring and self-care practices."
            )
        
        else:
            return (
                f"Risk Score: {risk_score:.1f}/100 (HIGH RISK)\n"
                "Facial expression analysis indicates elevated indicators of depression risk.\n"
                "Professional mental health evaluation is strongly recommended."
            )
    
    def _get_next_steps(self, risk_band: str, features: Dict[str, float]) -> str:
        """Get next steps based on risk band and features."""
        steps = []
        
        if risk_band == "high":
            steps.append("• Seek consultation with a mental health professional (psychiatrist, psychologist, or counselor)")
            steps.append("• Consider formal psychological assessment")
            steps.append("• Discuss screening results with a healthcare provider")
        
        elif risk_band == "medium":
            steps.append("• Schedule a check-up with your primary care physician")
            steps.append("• Consider speaking with a mental health professional for assessment")
            steps.append("• Monitor emotional state regularly")
        
        else:
            steps.append("• Maintain regular self-care and healthy lifestyle practices")
            steps.append("• Consider periodic emotional check-ins with trusted individuals")
            steps.append("• Seek help immediately if mood changes significantly")
        
        negative_emotion_ratio = features.get("negative_emotion_ratio", 0.0)
        if negative_emotion_ratio > 0.5:
            steps.append("• Pay attention to emotional experiences and mood patterns")
        
        sadness_mean = features.get("sadness_mean", 0.0)
        if sadness_mean > 40.0:
            steps.append("• Consider speaking with someone about feelings of sadness")
        
        return "\n".join(steps)
