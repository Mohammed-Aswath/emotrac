#!/usr/bin/env python
"""
Fix script for CPU-only PyTorch installation.
Run this after updating requirements.txt
"""

import subprocess
import sys
import os

def fix_torch_installation():
    """Reinstall PyTorch for CPU-only."""
    
    print("=" * 60)
    print("EmoTrace: CPU-Only PyTorch Fix")
    print("=" * 60)
    print()
    
    print("Step 1: Uninstalling current PyTorch (CUDA version)...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    print("✓ Uninstalled")
    print()
    
    print("Step 2: Installing CPU-only PyTorch...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.0.1+cpu",
        "torchvision==0.15.2+cpu",
        "-f", "https://download.pytorch.org/whl/torch_stable.html"
    ])
    print("✓ Installed CPU-only PyTorch")
    print()
    
    print("Step 3: Installing remaining dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ All dependencies installed")
    print()
    
    print("=" * 60)
    print("✓ Fix Complete!")
    print("=" * 60)
    print()
    print("You can now run: streamlit run app.py")
    print()

if __name__ == "__main__":
    try:
        fix_torch_installation()
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Manual fix:")
        print("1. pip uninstall -y torch torchvision torchaudio")
        print("2. pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html")
        print("3. pip install -r requirements.txt")
        sys.exit(1)
