#!/usr/bin/env python
"""
Install missing dependencies for EmoTrace
Run this if you get 'No module named' errors
"""

import subprocess
import sys

def install_missing_packages():
    """Install all missing packages."""
    
    print("=" * 70)
    print("EmoTrace: Installing Missing Dependencies")
    print("=" * 70)
    print()
    
    packages = [
        ("ultralytics", ">=8.0.0", "YOLOv5 utilities"),
    ]
    
    for package, version, description in packages:
        print(f"Installing {package} {version} ({description})...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                f"{package}{version}"
            ])
            print(f"✓ {package} installed successfully")
            print()
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}")
            print(f"  Error: {e}")
            return False
    
    print("=" * 70)
    print("✓ All dependencies installed!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Verify: python quickstart.py")
    print("  2. Run app: streamlit run app.py")
    print("  3. Open browser to http://localhost:8501")
    print()
    
    return True

if __name__ == "__main__":
    success = install_missing_packages()
    sys.exit(0 if success else 1)
