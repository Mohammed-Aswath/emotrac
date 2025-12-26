#!/usr/bin/env python
"""Test DeepFace emotion detection on sample images."""

import sys
from pathlib import Path
from deepface import DeepFace

def test_deepface(image_path):
    """Test DeepFace on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing DeepFace on: {image_path}")
    print(f"{'='*60}")
    
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return
    
    try:
        print("🔍 Analyzing emotions...")
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        
        if isinstance(result, list):
            result = result[0]
        
        print("\n✅ DeepFace Analysis Results:")
        print("-" * 60)
        
        emotions = result.get('emotion', {})
        
        # Sort by confidence
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, confidence in sorted_emotions:
            bar = "█" * int(confidence / 5)
            print(f"{emotion:12} : {confidence:6.2f}% {bar}")
        
        # Highlight primary emotion
        primary_emotion = sorted_emotions[0][0]
        primary_confidence = sorted_emotions[0][1]
        print("-" * 60)
        print(f"🎯 Primary Emotion: {primary_emotion.upper()} ({primary_confidence:.1f}%)")
        
        # Check if sadness is detected
        sadness = emotions.get('sad', 0.0)
        if sadness > 30:
            print(f"⚠️  HIGH SADNESS DETECTED: {sadness:.1f}%")
        else:
            print(f"ℹ️  Low sadness: {sadness:.1f}%")
        
        return emotions
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_deepface(image_path)
    else:
        print("Usage: python test_deepface.py <image_path>")
        print("\nExample:")
        print("  python test_deepface.py data/frames/clip123/frame_0.jpg")
        print("\nLooking for recent frames...")
        
        frames_dir = Path("data/frames")
        if frames_dir.exists():
            frame_files = list(frames_dir.glob("**/frame_*.jpg"))[:3]
            if frame_files:
                print(f"\nFound {len(frame_files)} frames. Testing first one...")
                test_deepface(str(frame_files[0]))
            else:
                print("No frames found. Please upload a video first.")
        else:
            print("No data/frames directory found.")
