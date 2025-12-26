#!/usr/bin/env python
"""Clean up old analysis results to ensure fresh analysis."""

import shutil
from pathlib import Path

def cleanup_old_results():
    """Remove old analysis data to force fresh analysis."""
    
    print("🧹 Cleaning up old analysis results...\n")
    
    dirs_to_check = [
        Path('data/frames'),
        Path('data/frames_cropped'),
        Path('data/au_results'),
        Path('data/micro_events'),
    ]
    
    total_deleted = 0
    
    for dir_path in dirs_to_check:
        if dir_path.exists():
            # Get all subdirectories/files
            items = list(dir_path.glob('*'))
            
            if items:
                for item in items:
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                            print(f"   ✓ Removed directory: {item.name}")
                        else:
                            item.unlink()
                            print(f"   ✓ Removed file: {item.name}")
                        total_deleted += 1
                    except Exception as e:
                        print(f"   ✗ Failed to remove {item.name}: {e}")
    
    print(f"\n✅ Cleaned up {total_deleted} items")
    print("\n📝 Next steps:")
    print("   1. Start Streamlit: streamlit run app.py")
    print("   2. Upload your crying/sad face video")
    print("   3. Click 'Run Analysis'")
    print("   4. Results should now show HIGH RISK for sad expressions")

if __name__ == '__main__':
    print("="*60)
    print("EmoTrace: Cleanup Old Analysis Results")
    print("="*60 + "\n")
    
    response = input("⚠️  This will delete ALL old analysis results. Continue? (yes/no): ").lower().strip()
    
    if response == 'yes':
        cleanup_old_results()
    else:
        print("Cancelled.")
