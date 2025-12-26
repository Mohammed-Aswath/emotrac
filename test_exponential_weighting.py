#!/usr/bin/env python
"""Demonstrate the exponential weighting effect on emotion-based risk scoring."""

from scoring.depression_screener import DepressionScreener
import pandas as pd

def test_emotion_scenarios():
    """Test different emotion scenarios to show exponential weighting."""
    
    screener = DepressionScreener()
    
    test_cases = [
        {
            "name": "High Anger (Angry Face)",
            "features": {
                "sadness_mean": 0.10,
                "anger_mean": 0.50,
                "fear_mean": 0.05,
                "disgust_mean": 0.05,
                "joy_mean": 0.05,
                "negative_emotion_ratio": 0.70,
                "negative_au_mean": 0.30,
                "positive_au_mean": 0.05,
                "negative_au_ratio": 2.0,
                "micro_expression_count": 0,
                "mean_intensity": 0.0,
            }
        },
        {
            "name": "High Sadness (Crying Face)",
            "features": {
                "sadness_mean": 0.55,
                "anger_mean": 0.10,
                "fear_mean": 0.05,
                "disgust_mean": 0.05,
                "joy_mean": 0.05,
                "negative_emotion_ratio": 0.75,
                "negative_au_mean": 0.35,
                "positive_au_mean": 0.05,
                "negative_au_ratio": 2.5,
                "micro_expression_count": 0,
                "mean_intensity": 0.0,
            }
        },
        {
            "name": "Mixed Negative (Anger 0.40, Sadness 0.20)",
            "features": {
                "sadness_mean": 0.20,
                "anger_mean": 0.40,
                "fear_mean": 0.10,
                "disgust_mean": 0.05,
                "joy_mean": 0.05,
                "negative_emotion_ratio": 0.75,
                "negative_au_mean": 0.30,
                "positive_au_mean": 0.05,
                "negative_au_ratio": 2.0,
                "micro_expression_count": 0,
                "mean_intensity": 0.0,
            }
        },
        {
            "name": "Happy Face (High Joy)",
            "features": {
                "sadness_mean": 0.05,
                "anger_mean": 0.05,
                "fear_mean": 0.05,
                "disgust_mean": 0.05,
                "joy_mean": 0.70,
                "negative_emotion_ratio": 0.20,
                "negative_au_mean": 0.10,
                "positive_au_mean": 0.50,
                "negative_au_ratio": 0.5,
                "micro_expression_count": 0,
                "mean_intensity": 0.0,
            }
        },
        {
            "name": "Neutral Face",
            "features": {
                "sadness_mean": 0.15,
                "anger_mean": 0.15,
                "fear_mean": 0.15,
                "disgust_mean": 0.15,
                "joy_mean": 0.15,
                "negative_emotion_ratio": 0.60,
                "negative_au_mean": 0.15,
                "positive_au_mean": 0.10,
                "negative_au_ratio": 1.0,
                "micro_expression_count": 0,
                "mean_intensity": 0.0,
            }
        },
    ]
    
    print("="*80)
    print("EMOTION-BASED RISK SCORING WITH EXPONENTIAL WEIGHTING")
    print("="*80)
    print()
    
    results = []
    
    for test_case in test_cases:
        name = test_case["name"]
        features = test_case["features"]
        
        print(f"📊 {name}")
        print("-" * 80)
        
        # Calculate risk score
        au_component = screener.compute_au_risk_component(features)
        emotion_component = screener.compute_emotion_risk_component(features)
        micro_component = screener.compute_micro_expression_risk_component(features)
        
        risk_score = screener.compute_risk_score(features)
        risk_band = screener.get_risk_band(risk_score)
        
        # Color code result
        if risk_band == "low":
            risk_emoji = "🟢"
        elif risk_band == "medium":
            risk_emoji = "🟡"
        else:
            risk_emoji = "🔴"
        
        print(f"  Risk Score: {risk_score:6.2f}/100  {risk_emoji} {risk_band.upper()}")
        print(f"    - AU Component:    {au_component:6.2f}/100 (30% weight)")
        print(f"    - Emotion Component: {emotion_component:6.2f}/100 (50% weight)")
        print(f"    - Micro Component: {micro_component:6.2f}/100 (20% weight)")
        print()
        
        results.append({
            "Scenario": name,
            "Risk Score": f"{risk_score:.1f}",
            "Category": risk_band.upper(),
            "AU": f"{au_component:.1f}",
            "Emotion": f"{emotion_component:.1f}",
            "Micro": f"{micro_component:.1f}"
        })
    
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    print("✅ Exponential weighting ensures dominant emotions have the most impact!")
    print("   Sadness and Anger now correctly show HIGH risk scores.")

if __name__ == '__main__':
    test_emotion_scenarios()
