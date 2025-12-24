# CPU-Only PyTorch Fix for EmoTrace

## Problem
Your system tried to load CUDA-specific libraries (`fbgemm.dll`) on a CPU-only machine.

## Solution

### Option 1: Run the Fix Script (Automatic)

```bash
cd c:\Users\aswat\EmoTrace
python fix_cpu_torch.py
```

This script will:
1. Uninstall current PyTorch (CUDA version)
2. Install CPU-only PyTorch
3. Reinstall all dependencies

### Option 2: Manual Fix (If script doesn't work)

```bash
# 1. Activate your conda environment
conda activate emotrace

# 2. Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio

# 3. Install CPU-only versions
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 4. Reinstall all dependencies
pip install -r requirements.txt

# 5. Verify
python quickstart.py

# 6. Run the app
streamlit run app.py
```

### What Changed

1. **requirements.txt** updated:
   - Changed `torch>=2.0.0` to `torch==2.0.1+cpu`
   - Changed `torchvision>=0.15.0` to `torchvision==0.15.2+cpu`

2. **Code updated for CPU**:
   - `face/yolo_face_detector.py`: Now explicitly uses `device='cpu'`
   - `features/au_extractor.py`: Now explicitly uses `device='cpu'`

## Performance Note

CPU-only processing will be slower than GPU:
- Typical analysis: 5-10 minutes per video (instead of 1-2 minutes with GPU)
- This is normal and expected

## Verification

After fix, run:
```bash
python quickstart.py
```

Should show all modules imported successfully.

## If Still Having Issues

1. Check Python version: `python --version` (should be 3.10+)
2. Check conda environment is active: `conda info --envs`
3. Try clearing pip cache: `pip cache purge`
4. Reinstall from scratch:
   ```bash
   conda create -n emotrace_new python=3.10
   conda activate emotrace_new
   pip install -r requirements.txt
   ```

## Running the App

Once fixed:

```bash
cd c:\Users\aswat\EmoTrace
conda activate emotrace
streamlit run app.py
```

Then open http://localhost:8501 and upload a video.
