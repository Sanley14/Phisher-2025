# GPU & External Storage Setup Guide

## GPU Configuration

### Check if GPU is Available

```powershell
# From project root, activate venv
& C:/Users/hp/Desktop/Phisher2025/.venv310/Scripts/Activate.ps1

# Check GPU status
python check_gpu.py
```

**Expected output if GPU detected:**
```
Number of GPUs detected: 1
✓ GPU(s) found:
  - /physical_device:GPU:0
✓ GPU memory growth enabled
✓ GPU is available and will be used for training
```

### Enable GPU (if not detected)

If no GPU is found, install NVIDIA support:

1. **Check if NVIDIA GPU is installed:**
   - Open Device Manager (devmgmt.msc)
   - Look under "Display adapters" for NVIDIA GPU

2. **Install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Match your GPU and Windows version
   - Follow installation wizard

3. **Install cuDNN:**
   - Download from: https://developer.nvidia.com/cudnn
   - Extract files to CUDA installation directory
   - Add to PATH if needed

4. **Reinstall TensorFlow with CUDA support:**
   ```powershell
   pip install --upgrade "tensorflow[and-cuda]"
   ```

5. **Verify GPU is now detected:**
   ```powershell
   python check_gpu.py
   ```

---

## External Storage Setup (Recommended)

### Why Use External Storage?

- **6 GB free space is tight** for model training (models ~126-200 MB each, logs can grow large)
- **Avoid "No space left on device"** errors during training
- **Keep system drive responsive** for other tasks

### Quick Setup with USB/External Drive

**Step 1: Prepare External Storage**

Plug in USB drive or external hard drive (e.g., appears as `E:\` or `F:\`)

**Step 2: Create Model Directory**

```powershell
# Create a folder on external drive
mkdir E:\phisher_models
```

**Step 3: Set Environment Variable (Current Session)**

```powershell
# Activate venv first
& C:/Users/hp/Desktop/Phisher2025/.venv310/Scripts/Activate.ps1

# Set external storage path
$env:EXTERNAL_MODEL_DIR = 'E:\phisher_models'

# Verify it's set
echo $env:EXTERNAL_MODEL_DIR
```

**Step 4: Run Training**

```powershell
# Training will now save all models, checkpoints, and logs to E:\phisher_models
python -m src.model.train_baseline
```

### Permanent Setup (All Future Sessions)

```powershell
# Set for current user (survives restart, all new PowerShell windows)
[Environment]::SetEnvironmentVariable('EXTERNAL_MODEL_DIR', 'E:\phisher_models', 'User')

# Close and reopen PowerShell to verify
echo $env:EXTERNAL_MODEL_DIR
```

---

## Alternative External Storage Options

### Network Drive

If you have a NAS or network share (e.g., `\\server\share`):

```powershell
# Map to a drive letter
net use Z: \\server\share /persistent:yes

# Set environment variable
$env:EXTERNAL_MODEL_DIR = 'Z:\phisher_models'
```

### Cloud Storage (OneDrive, Google Drive, etc.)

**Supported (fully synced locally):**
- OneDrive: `C:\Users\YourName\OneDrive\phisher_models`
- Google Drive (synced): `D:\GoogleDrive\phisher_models`
- Dropbox: `C:\Users\YourName\Dropbox\phisher_models`

**NOT recommended (still syncing):**
- Partial/cloud-only folders will cause permission errors during model saves
- Use fully-synced local copies only

### SSD or Secondary HDD

If you have an internal secondary drive:

```powershell
# D:\ drive example
$env:EXTERNAL_MODEL_DIR = 'D:\phisher_models'
```

---

## Verify External Storage Is Working

Run the setup helper:

```powershell
# Activate venv
& C:/Users/hp/Desktop/Phisher2025/.venv310/Scripts/Activate.ps1

# Run setup guide (interactive)
python setup_and_train.py
```

This will:
1. Check GPU status
2. Prompt for external storage path
3. Verify disk space
4. Start training (if you confirm)

---

## Training Command Reference

### Default (Uses C:/models if space available)
```powershell
python -m src.model.train_baseline
```

### With External Storage (Current Session)
```powershell
$env:EXTERNAL_MODEL_DIR = 'E:\phisher_models'
python -m src.model.train_baseline
```

### With External Storage (One-liner)
```powershell
& {$env:EXTERNAL_MODEL_DIR = 'E:\phisher_models'; python -m src.model.train_baseline}
```

---

## Troubleshooting

### GPU not detected but installed?

1. Verify NVIDIA drivers:
   ```powershell
   nvidia-smi
   ```
   Should show GPU info and CUDA version

2. Check TensorFlow GPU support:
   ```powershell
   python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"
   ```

3. Verify CUDA/cuDNN paths are in system PATH:
   ```powershell
   echo $env:PATH
   ```
   Should include NVIDIA CUDA bin and lib paths

4. Reinstall TensorFlow with explicit GPU flag:
   ```powershell
   pip uninstall tensorflow
   pip install "tensorflow[and-cuda]==2.20.0"
   ```

### "No space left on device" error?

1. Check available space:
   ```powershell
   Get-PSDrive -PSProvider FileSystem
   ```

2. Free up space on C:\ drive:
   ```powershell
   # Delete old logs (confirm first!)
   Remove-Item -Recurse .\logs\run_* -Confirm
   
   # Delete old checkpoints
   Remove-Item -Recurse .\models\checkpoints\* -Confirm
   ```

3. Or switch to external storage as described above.

### External storage not recognized?

1. Check if path exists and is accessible:
   ```powershell
   Test-Path E:\phisher_models
   ```

2. Create directory if missing:
   ```powershell
   mkdir E:\phisher_models
   ```

3. Check write permissions:
   ```powershell
   $testFile = 'E:\phisher_models\test.txt'
   'test' | Out-File $testFile
   Remove-Item $testFile
   ```
   If this fails, check drive permissions.

---

## After Training

Once training completes and model is saved:

```powershell
# Verify model files exist
ls E:\phisher_models\final_model\

# Run Streamlit UI (will load model from external storage)
python -m streamlit run src/agent_interface/chat_ui.py
```

The Streamlit UI will auto-detect the model or create a dummy model if needed.

---

## FAQ

**Q: Will training be slower with external storage?**
A: Slightly slower for model saves/loads (network/USB overhead), but GPU training itself is unaffected. Worth it to avoid disk-full crashes.

**Q: Can I move the model to C:\ after training?**
A: Yes. Copy `E:\phisher_models\final_model\model.*` to `C:\Users\hp\Desktop\Phisher2025\models\final_model\`.

**Q: What if my external drive disconnects during training?**
A: Training will crash with a file write error. Keep the drive connected during training; remove only after training completes.

**Q: Can I use multiple external drives?**
A: Yes, but one at a time. Set `EXTERNAL_MODEL_DIR` to the active drive.
