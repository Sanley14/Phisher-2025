# Quick Reference: GPU & External Storage

## TL;DR - Fast Start

### Check GPU Status
```powershell
python check_gpu.py
```

### Use External Storage (USB Drive)
```powershell
# One-time setup
mkdir E:\phisher_models

# Before training (set environment variable)
$env:EXTERNAL_MODEL_DIR = 'G:\phisher_models'

# Then run training
python -m src.model.train_baseline
```

### Use GPU (if available)
- TensorFlow auto-detects GPU when available
- No code changes needed—just run training
- If GPU not detected but you have NVIDIA card, install CUDA + cuDNN

---

## Commands Cheat Sheet

| Task | Command |
|------|---------|
| Check GPU status | `python check_gpu.py` |
| Set external storage (session) | `$env:EXTERNAL_MODEL_DIR = 'E:\path'` |
| Set external storage (permanent) | `[Environment]::SetEnvironmentVariable('EXTERNAL_MODEL_DIR', 'E:\path', 'User')` |
| Train with external storage | `python -m src.model.train_baseline` (after setting env var) |
| Interactive setup | `python setup_and_train.py` |

---

## Disk Space Requirements

| Scenario | Space Needed |
|----------|-------------|
| 1-epoch smoke test | ~200 MB (weights only) |
| Full model save | ~400-500 MB |
| With TensorBoard logs | ~100-200 MB additional |
| Safe margin (3 epochs) | ~1.5 GB |

**Current free space: 6 GB** ✓ (Use external storage to be safe)

---

## GPU Setup Status

### Current Status
```
GPU Detected: ❌ No
Reason: No NVIDIA GPU or CUDA/cuDNN not installed
Training: Will use CPU (slower, but works)
```

### To Enable GPU (if you have NVIDIA card)

1. **Verify GPU installed:**
   - Open Device Manager → Display adapters
   - Look for NVIDIA GPU entry

2. **Install CUDA Toolkit:**
   - https://developer.nvidia.com/cuda-downloads
   - Select Windows + architecture
   - Install to default location

3. **Install cuDNN:**
   - https://developer.nvidia.com/cudnn
   - Extract to CUDA folder (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X`)

4. **Reinstall TensorFlow:**
   ```powershell
   pip install --upgrade "tensorflow[and-cuda]"
   ```

5. **Verify:**
   ```powershell
   python check_gpu.py
   # Should show: Number of GPUs detected: 1
   ```

---

## Storage Options (Ranked by Recommendation)

| Option | Speed | Reliability | Ease | When to Use |
|--------|-------|-------------|------|------------|
| **USB 3.0+ drive** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | **RECOMMENDED** |
| **External SSD** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Best if you have one |
| **Secondary HDD** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | If internal |
| **Network share** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | If available |
| **OneDrive (synced)** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | If in sync folder |

**DO NOT USE:** Cloud-only storage, partially-synced folders, or any path you don't control.

---

## Troubleshooting

### Training fails with "No space left on device"
```powershell
# Use external storage (see TL;DR above)
# OR delete old logs/checkpoints:
Remove-Item -Recurse .\logs\run_* -Confirm
Remove-Item -Recurse .\models\checkpoints\* -Confirm
```

### GPU not detected but I have NVIDIA card
```powershell
# Verify drivers
nvidia-smi

# If command not found, install NVIDIA drivers:
# https://www.nvidia.com/Download/driverDetails.aspx

# Then install CUDA + cuDNN + reinstall TensorFlow
```

### External storage not recognized
```powershell
# Check path exists
Test-Path E:\phisher_models

# Create if missing
mkdir E:\phisher_models

# Test write permission
'test' | Out-File E:\phisher_models\test.txt
```

---

## Full Setup (Interactive)

Run the interactive helper:

```powershell
# Activate venv
& C:/Users/hp/Desktop/Phisher2025/.venv310/Scripts/Activate.ps1

# Run setup guide
python setup_and_train.py
```

This will:
1. ✓ Check GPU
2. ✓ Prompt for external storage path
3. ✓ Check disk space
4. ✓ Start training (if confirmed)

---

## After Training

### Verify Model Saved
```powershell
# If using external storage
ls E:\phisher_models\final_model\

# If using default
ls .\models\final_model\
```

### Run Streamlit UI
```powershell
python -m streamlit run src/agent_interface/chat_ui.py
# Open browser to: http://localhost:8501
```

### Copy Model to Project (Optional)
```powershell
# Copy from external storage to project (if you want)
Copy-Item E:\phisher_models\final_model\* .\models\final_model\ -Force
```

---

## Environment Variable Reference

### Session-Only (Lost when PowerShell closes)
```powershell
$env:EXTERNAL_MODEL_DIR = 'E:\phisher_models'
```

### Permanent (Survives restart & new sessions)
```powershell
# Set for current user
[Environment]::SetEnvironmentVariable('EXTERNAL_MODEL_DIR', 'E:\phisher_models', 'User')

# Verify
echo $env:EXTERNAL_MODEL_DIR

# Unset (if needed)
[Environment]::SetEnvironmentVariable('EXTERNAL_MODEL_DIR', $null, 'User')
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `check_gpu.py` | Detect GPU and show TensorFlow GPU config |
| `setup_and_train.py` | Interactive setup + training launcher |
| `GPU_AND_STORAGE_SETUP.md` | Detailed setup guide (this file) |
| `config/default.yaml` | Training hyperparameters (model size, epochs, etc.) |
| `src/model/train_baseline.py` | Main trainer (reads `EXTERNAL_MODEL_DIR` env var) |

---

## Contact & Support

If training still fails:

1. **Check disk space first:**
   ```powershell
   Get-PSDrive C:
   ```

2. **Verify external storage:**
   ```powershell
   Test-Path E:\phisher_models
   $env:EXTERNAL_MODEL_DIR
   ```

3. **Run GPU check:**
   ```powershell
   python check_gpu.py
   ```

4. **Check logs:**
   ```powershell
   # Training logs saved to (if external storage set):
   ls E:\phisher_models\logs\
   ```

---

**Last updated:** Nov 14, 2025
