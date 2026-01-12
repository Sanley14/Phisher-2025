#!/usr/bin/env python3
"""CLI wrapper to run the Phase-1 trainer using config YAML.

Usage examples:
  python tools/run_trainer.py --config config/default.yaml --epochs 1
  python tools/run_trainer.py --config config/default.yaml --dry-run

The script will load the YAML config, map it to a lightweight namespace
compatible with `PhishingModelTrainer`, and run the training pipeline.
"""
import argparse
import yaml
from types import SimpleNamespace
from pathlib import Path
import sys
import os

# Ensure project root is on sys.path so `from src...` imports work when
# running this script from tools/ or other subfolders.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dict_to_namespace(d):
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml", help="path to YAML config")
    p.add_argument("--epochs", type=int, help="override epochs")
    p.add_argument("--batch-size", type=int, help="override batch size")
    p.add_argument("--dry-run", action="store_true", help="only validate dataset, do not train")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(2)

    cfg = load_yaml(cfg_path)
    ns = dict_to_namespace(cfg)

    # Ensure required top-level fields exist with safe defaults
    # We'll create the shape the trainer expects
    project_name = getattr(ns, "project_name", "phisher")
    model = getattr(ns, "model", SimpleNamespace(type="hybrid", max_seq_len=128, vocab_size=20000))
    training = getattr(ns, "training", SimpleNamespace(batch_size=32, epochs=1, early_stopping_patience=5, lr_patience=3, learning_rate=1e-3))
    paths = getattr(ns, "paths", SimpleNamespace(models="models/final_model", logs="logs"))
    data = getattr(ns, "data", SimpleNamespace(path="data/processed/phishing_data.csv"))

    # Apply overrides
    if args.epochs is not None:
        training.epochs = args.epochs
    if args.batch_size is not None:
        training.batch_size = args.batch_size

    # Build a simple config object
    # Ensure model defaults exist (some YAMLs may omit these)
    if not hasattr(model, "vocab_size"):
        setattr(model, "vocab_size", 20000)
    if not hasattr(model, "max_seq_len"):
        setattr(model, "max_seq_len", 128)

    # Ensure training defaults exist
    if not hasattr(training, "early_stopping_patience"):
        setattr(training, "early_stopping_patience", 5)
    if not hasattr(training, "lr_patience"):
        setattr(training, "lr_patience", 3)
    if not hasattr(training, "learning_rate"):
        setattr(training, "learning_rate", 1e-3)
    if not hasattr(training, "batch_size"):
        setattr(training, "batch_size", 32)

    root = SimpleNamespace(
        project_name=project_name,
        model=model,
        training=training,
        paths=paths,
        data=data,
    )

    # Import here to keep fast failures earlier if trainer missing
    try:
        from src.model.train_baseline import PhishingModelTrainer
    except Exception as e:
        print("Could not import trainer:", e)
        raise

    trainer = PhishingModelTrainer(root)

    if args.dry_run:
        print("Running dry-run: validating dataset only")
        import pandas as pd
        df = pd.read_csv(root.data.path)
        ok, report = trainer.validate_dataset(df)
        print("Valid:", ok)
        print(report)
        return

    print("Starting training run — epochs=", int(training.epochs))
    res = trainer.run_full_pipeline()
    print("Run result:", res)


if __name__ == "__main__":
    main()
