#!/usr/bin/env python
"""
Quick setup guide and launcher for Phisher2025 training with GPU support.
Shows GPU status, external storage options, and executes training.
"""

import os
import sys
import subprocess
from pathlib import Path

print("\n" + "=" * 80)
print(" " * 20 + "PHISHER2025 - TRAINING SETUP GUIDE")
print("=" * 80)

# ============================================================================
# STEP 1: GPU CHECK
# ============================================================================

print("\n[STEP 1] Checking GPU availability...")
print("-" * 80)

result = subprocess.run(
    [sys.executable, "check_gpu.py"],
    cwd=Path(__file__).parent,
    capture_output=False
)

# ============================================================================
# STEP 2: EXTERNAL STORAGE SETUP
# ============================================================================

print("\n[STEP 2] External storage configuration")
print("-" * 80)

print("\nYou have 6 GB of free space on C:/ drive.")
print("To avoid 'No space left on device' errors, use an external drive.")
print("\nOptions:")
print("  A) Use external USB drive or SD card (e.g., E:, F:)")
print("  B) Use network drive (e.g., mapped as Z:)")
print("  C) Use cloud storage with local sync (e.g., OneDrive, Google Drive)")
print("\nWARNING: Do NOT use cloud storage that is still syncing—it will cause issues!")
print("\nSetup external storage:")

external_dir = os.getenv("EXTERNAL_MODEL_DIR")
if external_dir:
    print(f"\n✓ EXTERNAL_MODEL_DIR already set to: {external_dir}")
    proceed = input("\nUse this path? (yes/no): ").strip().lower()
    if proceed != "yes":
        external_dir = None
else:
    print("\nTo use external storage, set the EXTERNAL_MODEL_DIR environment variable:")
    print("\n  On PowerShell:")
    print("    $env:EXTERNAL_MODEL_DIR = 'E:\\phisher_models'")
    print("    python -m src.model.train_baseline")
    print("\n  Or permanently (for current session):")
    print("    [Environment]::SetEnvironmentVariable('EXTERNAL_MODEL_DIR', 'E:\\phisher_models', 'User')")
    print("    # Then restart PowerShell")
    print("\nExamples:")
    print("  - USB drive:       $env:EXTERNAL_MODEL_DIR = 'E:\\phisher_models'")
    print("  - Network drive:   $env:EXTERNAL_MODEL_DIR = 'Z:\\phisher_models'")
    print("  - Synced folder:   $env:EXTERNAL_MODEL_DIR = 'C:\\Users\\YourName\\OneDrive\\phisher_models'")
    
    user_input = input("\nEnter external storage path (or press Enter to use C:/models): ").strip()
    if user_input:
        external_dir = user_input
        os.environ["EXTERNAL_MODEL_DIR"] = external_dir
        Path(external_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Set EXTERNAL_MODEL_DIR = {external_dir}")

# ============================================================================
# STEP 3: DISK SPACE CHECK
# ============================================================================

print("\n[STEP 3] Checking available disk space...")
print("-" * 80)

import shutil

if external_dir:
    path = Path(external_dir)
else:
    path = Path("models")

try:
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    
    print(f"Path: {path}")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Free:  {free_gb:.1f} GB")
    
    if free_gb < 1.0:
        print("\n⚠ WARNING: Less than 1 GB free!")
        print("  Training may fail if insufficient space for model saves.")
        print("  Please free up space or use a different external drive.")
    elif free_gb < 2.0:
        print("\n⚠ WARNING: Less than 2 GB free!")
        print("  Training should work but space is tight.")
    else:
        print("\n✓ Sufficient space available for training")
        
except Exception as e:
    print(f"⚠ Could not check disk space: {e}")

# ============================================================================
# STEP 4: TRAINING EXECUTION
# ============================================================================

print("\n[STEP 4] Ready to start training")
print("-" * 80)

print("\nTraining will:")
print("  1. Load 1000 synthetic multilingual emails (4 languages)")
print("  2. Build a Hybrid CNN-LSTM model with XLM-RoBERTa tokenizer")
print("  3. Train for 1 epoch (smoke test)")
print("  4. Save model weights and optionally full model")
print(f"  5. Save to: {external_dir if external_dir else 'models/final_model'}")

proceed = input("\nStart training now? (yes/no): ").strip().lower()
if proceed == "yes":
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    # Set env var if using external storage
    if external_dir:
        os.environ["EXTERNAL_MODEL_DIR"] = external_dir
    
    # Run trainer
    try:
        subprocess.run(
            [sys.executable, "-m", "src.model.train_baseline"],
            cwd=Path(__file__).parent
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Error running training: {e}")
else:
    print("\nTraining skipped. When ready, run:")
    if external_dir:
        print(f"\n  $env:EXTERNAL_MODEL_DIR = '{external_dir}'")
    print("  python -m src.model.train_baseline")

# ============================================================================
# FINAL INSTRUCTIONS
# ============================================================================

print("\n" + "=" * 80)
print("After training completes:")
print("-" * 80)
print("\n1. Verify model was saved:")
if external_dir:
    model_path = Path(external_dir) / "final_model"
else:
    model_path = Path("models/final_model")

print(f"   ls {model_path}")
print("\n2. Run Streamlit UI:")
print("   python -m streamlit run src/agent_interface/chat_ui.py")
print("\n3. Open browser to http://localhost:8501")
print("=" * 80 + "\n")
