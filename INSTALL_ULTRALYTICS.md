# Missing ultralytics Package - Quick Fix

## Problem
The YOLOv5 model requires the `ultralytics` package which was missing.

Error:
```
ModuleNotFoundError: No module named 'ultralytics'
```

## Solution - Install Missing Package

Run this ONE command in your terminal:

```bash
pip install ultralytics>=8.0.0
```

**That's it!** Just 1 command.

## Step by Step

1. Open PowerShell (or Command Prompt)
2. Make sure you're in the emotrace environment:
   ```bash
   conda activate emotrace
   ```

3. Install ultralytics:
   ```bash
   pip install ultralytics>=8.0.0
   ```

4. Wait 1-2 minutes for it to finish
5. Run quickstart to verify:
   ```bash
   python quickstart.py
   ```

6. Then run the app:
   ```bash
   streamlit run app.py
   ```

## What This Does

`ultralytics` is the official package from Ultralytics that provides YOLOv5 models and utilities. Without it, PyTorch can't load the face detection model.

## Installation Time

About 1-2 minutes (depending on internet speed).

## Then You're Done

After installing, just:
```bash
streamlit run app.py
```

Upload a video and it should work!
