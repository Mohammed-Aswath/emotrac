#!/usr/bin/env python
"""Download OpenCV Haar Cascade classifier files."""

import os
import urllib.request
from pathlib import Path

def download_cascade_files():
    """Download and install Haar Cascade files."""
    
    # Get OpenCV installation path
    import cv2
    cv2_path = Path(cv2.__file__).parent
    cascade_dir = cv2_path / 'data' / 'haarcascades'
    
    # Create directory if it doesn't exist
    cascade_dir.mkdir(parents=True, exist_ok=True)
    
    # Cascade files to download
    cascades = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml',
    ]
    
    base_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/'
    
    for cascade_file in cascades:
        url = base_url + cascade_file
        output_path = cascade_dir / cascade_file
        
        if output_path.exists():
            print(f"✓ {cascade_file} already exists")
            continue
        
        try:
            print(f"⬇ Downloading {cascade_file}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Downloaded {cascade_file} to {output_path}")
        except Exception as e:
            print(f"✗ Failed to download {cascade_file}: {e}")
            return False
    
    return True

if __name__ == '__main__':
    print("Setting up OpenCV Haar Cascade files...\n")
    if download_cascade_files():
        print("\n✅ All cascade files downloaded successfully!")
    else:
        print("\n❌ Some cascade files failed to download")
