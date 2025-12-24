# EmoTrace: Facial Expression Analysis for Depression Risk Screening

## Overview

EmoTrace is a research prototype that analyzes facial expressions in video to screen for depression risk indicators. It implements a complete pipeline from video ingestion through feature extraction, risk scoring, and personalized recommendations.

**IMPORTANT:** This is a non-diagnostic research tool and cannot replace professional mental health assessment.

## Key Features

- **Video Processing**: Extract frames from uploaded videos at configurable sampling rates
- **Face Detection**: YOLOv5-Face for robust multi-face detection with highest-confidence selection
- **Action Unit Extraction**: Py-Feat for extracting facial Action Units (AUs) and emotion probabilities
- **Micro-Expression Detection**: Detect rapid facial movements and emotional micro-expressions
- **Feature Engineering**: Compute statistical features from AU and emotion time-series
- **Risk Scoring**: Rule-based weighted scoring model (0-100 scale)
- **Visualization**: Interactive plots for AU trajectories, emotion distributions, and micro-expression timelines
- **Recommendations**: Human-readable personalized recommendations based on risk band

## System Architecture

```
Input Video
    ↓
Frame Extraction (OpenCV)
    ↓
Face Detection (YOLOv5)
    ↓
Face Cropping & Resizing
    ↓
AU & Emotion Extraction (Py-Feat)
    ↓
Micro-Expression Detection
    ↓
Feature Engineering
    ↓
Risk Scoring & Classification
    ↓
Visualization & Recommendations
```

## Project Structure

```
EmoTrace/
├── app.py                          # Streamlit frontend
├── run_pipeline.py                 # Core analysis pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/                           # Data storage directories
│   ├── raw_videos/                # Uploaded video files
│   ├── frames/                    # Extracted frames
│   ├── frames_cropped/            # Face crops
│   ├── au_results/                # AU extraction results
│   └── micro_events/              # Detected events
│
├── video/                          # Video processing
│   └── extract_frames.py          # Frame extraction
│
├── face/                           # Face detection
│   └── yolo_face_detector.py      # YOLOv5-Face wrapper
│
├── features/                       # Feature extraction
│   ├── au_extractor.py            # AU & emotion extraction
│   └── micro_expression.py        # Micro-expression detection
│
├── scoring/                        # Risk scoring
│   ├── feature_engineering.py     # Feature computation
│   ├── depression_screener.py     # Risk scoring model
│   └── recommendation.py          # Recommendation generation
│
├── visualization/                  # Data visualization
│   └── plots.py                   # Matplotlib plotting
│
└── utils/                          # Utilities
    ├── config.py                  # Configuration constants
    └── logger.py                  # Logging setup
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- 4GB+ RAM (8GB+ recommended)
- CUDA-capable GPU optional but recommended for faster processing

### Step 1: Clone or Set Up Project

```bash
cd EmoTrace
```

### Step 2: Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n emotrace python=3.10
conda activate emotrace
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- OpenCV (video processing)
- NumPy & Pandas (data handling)
- PyTorch & Torchvision (deep learning)
- YOLOv5 (face detection)
- Py-Feat (AU extraction)
- Streamlit (web interface)
- Matplotlib (visualization)

### Step 4: Download Model Weights

Models are automatically downloaded on first run. Ensure internet connection is available during first execution.

## Usage

### Running the Streamlit App

Start the interactive web interface:

```bash
streamlit run app.py
```

The app opens in your default browser at `http://localhost:8501`

#### Interface Steps:
1. Click "Upload a video file (.mp4)" to select your video
2. Click "🚀 Run Analysis" button
3. Wait for processing to complete (typically 1-5 minutes depending on video length)
4. Review results including:
   - Risk score and classification
   - Detailed AU trajectories
   - Emotion probability distribution
   - Detected micro-expression timeline
   - Personalized recommendations

### Running the Pipeline Programmatically

```python
from run_pipeline import run_analysis_pipeline

result = run_analysis_pipeline("path/to/video.mp4")

print(f"Risk Score: {result['risk_score']:.2f}")
print(f"Risk Band: {result['risk_band']}")
print(f"Frames Analyzed: {result['num_frames']}")
print(f"Faces Detected: {result['num_faces']}")
```

## Technical Details

### Video Processing
- Supports .mp4 files
- Configurable frame sampling (default 20 FPS)
- Limits processing to first 30 frames for efficiency
- Frames stored at original resolution

### Face Detection
- YOLOv5s-Face model
- Confidence threshold: 0.45
- Selects highest-confidence face per frame
- Crops and resizes to 224×224

### Action Unit Extraction
- Py-Feat Detector (RetinaFace + MobileFaceNet)
- Extracts 27 Action Units (AU01-AU27)
- Computes emotion probabilities (7 classes)
- Runs with n_jobs=1 for stability

### Micro-Expression Detection
- Detects rapid AU changes (ΔAU > 5.0)
- Requires 2-15 frames duration
- Identifies onset, apex, offset
- Maps to dominant emotion

### Risk Scoring
Combines three components:

1. **AU Component (40% weight)**
   - Negative AUs (1,2,4,5,7,15,17,23,24,25,26): indicate sadness, disgust, fear
   - Positive AUs (6,12): indicate joy, smile
   - Metric: negative_au_mean, positive_au_mean

2. **Emotion Component (35% weight)**
   - Negative emotions: sadness, anger, fear, disgust
   - Positive emotions: joy
   - Metric: negative_emotion_ratio

3. **Micro-Expression Component (25% weight)**
   - Count and intensity of rapid facial movements
   - Metric: micro_expression_count, mean_intensity

Final score normalized to 0-100:
- **Low Risk**: 0-33
- **Medium Risk**: 34-66
- **High Risk**: 67-100

### Feature Engineering
Computes 12 features from extracted data:
- `negative_au_mean`: Mean intensity of negative AUs
- `negative_au_std`: Std deviation of negative AUs
- `positive_au_mean`: Mean intensity of positive AUs
- `negative_au_ratio`: Negative AU ratio
- `sadness_mean`, `anger_mean`, `fear_mean`, `disgust_mean`, `joy_mean`, `neutral_mean`, `surprise_mean`: Emotion means
- `negative_emotion_ratio`: Ratio of negative emotions
- `micro_expression_count`: Number of detected micro-expressions
- `mean_intensity`: Mean intensity of micro-expressions
- `mean_duration`: Mean duration in frames

## Output Files

All results saved to `data/` directory:

```
data/
├── raw_videos/{clip_id}/          # Original uploaded video
├── frames/{clip_id}/              # Extracted frames (frame_0000.jpg, ...)
├── frames_cropped/{clip_id}/      # Face crops (face_0000.jpg, ...)
├── au_results/{clip_id}_aus.csv   # AU & emotion time-series
└── micro_events/{clip_id}_events.csv # Detected micro-expression events
```

## Important Disclaimers

### NOT a Medical Tool
- **This system is a research prototype**, not a medical diagnostic device
- Results should **NEVER be used for clinical decision-making**
- Facial expressions alone cannot diagnose depression
- Professional mental health assessment is essential

### Limitations
- Accuracy depends on video quality and lighting
- Multiple faces may produce unexpected results
- Limited to 30 frames for processing efficiency
- Cultural and individual variation in expressions not accounted for

### Privacy
- Videos are processed locally (not uploaded to external servers)
- All data files stored in `data/` directory
- Delete data manually after use if privacy is a concern

## Recommendations for Use

### For Researchers
- Use as baseline feature extraction for larger studies
- Combine with validated depression screening instruments
- Collect ground truth labels for validation
- Test on diverse populations

### For Individuals
- Use only as educational tool
- **Always consult with healthcare professionals** for concerns
- Do not rely on results for self-diagnosis
- Seek professional help if experiencing mood changes

## Troubleshooting

### Model Download Failures
If models fail to download:
1. Check internet connection
2. Clear Torch cache: `torch.hub.set_dir('/path/to/cache')`
3. Download manually and specify path in code

### GPU/Memory Issues
- Reduce `max_frames` in config
- Use CPU instead: set `device='cpu'` in detector
- Close other applications
- Use 64-bit Python

### Video Processing Errors
- Ensure video is valid .mp4 file
- Check file permissions
- Try converting video: `ffmpeg -i input.mov -c:v libx264 output.mp4`

### Streamlit Issues
- Clear cache: `streamlit cache clear`
- Update Streamlit: `pip install --upgrade streamlit`
- Check localhost:8501 in browser

## Performance

Typical processing times on standard hardware:
- **30-frame video**: 2-5 minutes
- **Face detection**: 10-20 seconds
- **AU extraction**: 60-120 seconds
- **Scoring & visualization**: 5-10 seconds

With GPU acceleration, 2-3x faster.

## References & Acknowledgments

- YOLOv5: https://github.com/ultralytics/yolov5
- Py-Feat: https://py-feat.org/
- Facial Action Units: Ekman & Friesen (1978)
- Depression screening: DSM-5 criteria

## Citation

If using this tool in research:

```
@software{emotrace2024,
  title={EmoTrace: Facial Expression Analysis for Depression Risk Screening},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo/emotrace}
}
```

## License

This project is provided as-is for research and educational purposes.

## Contact & Support

For questions or issues:
- Check troubleshooting section
- Review code comments and docstrings
- Test with sample video first

---

**Last Updated**: December 2024
**Version**: 1.0
**Status**: Research Prototype - Not for Clinical Use
