# EmoTrace - Quick Start Guide

## Installation (5 minutes)

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python quickstart.py
```

Should show: `✓ All modules imported successfully!`

## Running the App (1 minute)

```bash
streamlit run app.py
```

Opens at: http://localhost:8501

## Usage Steps

1. **Upload Video** → Click file uploader, select .mp4
2. **Run Analysis** → Click "🚀 Run Analysis" button
3. **View Results** → See risk score and visualizations
4. **Get Recommendations** → Read personalized guidance

## Example Workflow

```bash
# Terminal 1: Start the app
streamlit run app.py

# Browser: Navigate to http://localhost:8501
# - Upload your video.mp4
# - Click Run Analysis
# - View results in 2-5 minutes
```

## Output Files

Results saved in `data/` folder:
- `raw_videos/` - Uploaded videos
- `frames/` - Extracted frames
- `frames_cropped/` - Face crops
- `au_results/` - AU analysis CSV
- `micro_events/` - Detected events CSV

## Key Configuration (in `utils/config.py`)

```python
VIDEO_CONFIG = {
    "fps_sample": 20,        # Sample at 20 FPS
    "max_frames": 30,        # Process first 30 frames
}

FACE_DETECTION_CONFIG = {
    "conf_threshold": 0.45,  # Detection confidence
    "face_size": 224,        # Output face crop size
}
```

## Troubleshooting

### Import Errors?
```bash
pip install --upgrade -r requirements.txt
```

### Slow Processing?
- Video with less movement runs faster
- First run downloads models (~1GB)
- Subsequent runs use cached models

### No Faces Detected?
- Ensure good lighting
- Face clearly visible in video
- Try with different video

### Memory Issues?
- Reduce `max_frames` in config (e.g., to 15)
- Close other applications
- Use 64-bit Python

## Command Line Usage

```python
from run_pipeline import run_analysis_pipeline

result = run_analysis_pipeline("video.mp4")
print(f"Risk Score: {result['risk_score']:.1f}")
print(f"Risk Band: {result['risk_band']}")
```

## System Requirements

- **Python**: 3.10+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for models + video
- **GPU**: Optional but recommended (NVIDIA/CUDA)

## Next Steps

- Run with sample video
- Review AU and emotion plots
- Check recommendations section
- Read detailed documentation in README.md

---

**Version**: 1.0
**Status**: Ready to use
**Time to first result**: ~2-5 minutes per video
