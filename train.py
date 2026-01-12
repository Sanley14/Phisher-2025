"""
Training runner script - handles path setup and launches training.
"""
import os
import sys
from pathlib import Path

from src.model.train_baseline import PhishingModelTrainer
from src.utils.config import load_config

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Change to project directory
os.chdir(project_root)

if __name__ == "__main__":
    # Load and validate the configuration first
    config_path = Path("config/default.yaml")
    try:
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("✓ Configuration loaded and validated successfully.")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or validating configuration: {e}")
        sys.exit(1)
    # Pass the validated config object to the trainer
    trainer = PhishingModelTrainer(config=config)
    results = trainer.run_full_pipeline()
