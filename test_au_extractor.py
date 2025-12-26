#!/usr/bin/env python
"""Test if AUExtractor uses DeepFace."""

from features.au_extractor import AUExtractor
from pathlib import Path

extractor = AUExtractor()
print(f'Using real detector: {extractor.use_real_detector}')
print(f'DeepFace available: {extractor.deepface is not None}')

# Test on a frame
frame_path = 'data/frames/0118ce2c/frame_0015.jpg'
if Path(frame_path).exists():
    print(f'\nTesting extract_frame on {frame_path}')
    aus, emotions = extractor.extract_frame(frame_path)
    print(f'Emotions returned: {emotions}')
    sad_val = emotions.get('sadness', 0.0)
    print(f'Sadness: {sad_val:.4f}')
    
    if sad_val < 0.3:
        print('\n⚠️  WARNING: Sadness is very low!')
        print('This indicates synthetic method is being used, not DeepFace')
else:
    print(f'Frame not found: {frame_path}')
