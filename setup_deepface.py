#!/usr/bin/env python
"""
Fix for EmoTrace: Ensures DeepFace is properly configured before running analysis.

Run this script once before uploading videos.
"""

import shutil
from pathlib import Path
import cv2

def setup_deepface_cascades():
    """Copy cascade files to DeepFace expected location."""
    cv2_path = Path(cv2.__file__).parent
    source_dir = cv2_path / 'data' / 'haarcascades'
    target_dir = cv2_path / 'data'
    
    print("🔧 Setting up DeepFace cascade files...")
    print(f"   Source: {source_dir}")
    print(f"   Target: {target_dir}\n")
    
    cascades = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml',
    ]
    
    for cascade in cascades:
        source = source_dir / cascade
        target = target_dir / cascade
        
        if source.exists():
            if target.exists():
                print(f"   ✓ {cascade} (already exists)")
            else:
                shutil.copy(str(source), str(target))
                print(f"   ✓ Copied {cascade}")
        else:
            print(f"   ✗ Source not found: {cascade}")
    
    print("\n✅ DeepFace cascade files are ready!\n")

def test_deepface_import():
    """Test that DeepFace can be imported and used."""
    print("🧪 Testing DeepFace import...")
    try:
        from deepface import DeepFace
        print("   ✓ DeepFace imported successfully")
        return True
    except Exception as e:
        print(f"   ✗ Failed to import DeepFace: {e}")
        return False

def test_au_extractor():
    """Test that AU extractor uses DeepFace."""
    print("🧪 Testing AU extractor...")
    try:
        from features.au_extractor import AUExtractor
        extractor = AUExtractor()
        
        if extractor.use_real_detector:
            print("   ✓ AU extractor configured to use real DeepFace")
            return True
        else:
            print("   ✗ AU extractor using synthetic fallback (DeepFace failed)")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("EmoTrace DeepFace Setup & Verification")
    print("="*60 + "\n")
    
    setup_deepface_cascades()
    deepface_ok = test_deepface_import()
    au_ok = test_au_extractor()
    
    print("\n" + "="*60)
    if deepface_ok and au_ok:
        print("✅ All checks passed! Ready to analyze videos.")
        print("\nNow you can:")
        print("1. Open http://localhost:8502")
        print("2. Upload a video (sad/crying face recommended)")
        print("3. Click 'Run Analysis'")
        print("\nExpected result for sad video: HIGH RISK (70+/100)")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
    print("="*60)
