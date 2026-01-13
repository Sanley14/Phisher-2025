"""
Training pipeline for the Phisher2025 model.

Phase 1 enhancements added here:
- Mixed precision (optional, enabled when a GPU is available)
- Learning rate scheduling (ReduceLROnPlateau)
- EarlyStopping callback
- Simple dataset validation utility

The trainer accepts a validated `AppConfig` object (see `src/utils/config_schema.py`).
"""
import os
from pathlib import Path
from typing import Tuple
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import callbacks

try:
    from src.utils.config_schema import AppConfig
except Exception as _err:  # guard against schema import errors during quick tests
    AppConfig = None
    print(f"Warning: could not import AppConfig: {_err}")
from src.data.load_data import PhishingDataLoader
from src.model.build_model import build_hybrid_cnn_lstm


class PhishingModelTrainer:
    """Trainer with Phase 1 production improvements.

    Usage:
        trainer = PhishingModelTrainer(config)
        trainer.run_full_pipeline()
    """

    def __init__(self, config: AppConfig):
        self.config = config
        print("PhishingModelTrainer initialized with configuration:")
        print(f"  Project: {self.config.project_name}")
        print(f"  Model Type: {self.config.model.type}")
        print(f"  Batch Size: {self.config.training.batch_size}")
        print(f"  Epochs: {self.config.training.epochs}")

        # Enable mixed precision when GPUs are available and configured
        self._maybe_enable_mixed_precision()

        # directories
        self.model_dir = Path(self.config.paths.models or "models/final_model")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_enable_mixed_precision(self):
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"GPUs detected: {len(gpus)} — enabling mixed precision policy")
                from tensorflow.keras.mixed_precision import experimental as mixed_precision

                policy = mixed_precision.Policy("mixed_float16")
                mixed_precision.set_policy(policy)
            else:
                print("No GPUs detected — mixed precision not enabled")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")

    # ---------------- Data validation ----------------
    def validate_dataset(self, df) -> Tuple[bool, dict]:
        #Basic validation of the dataset DataFrame.Checks performed:required columns exist - no missing labels - language distribution and class balance reported#
        required = ["email_text", "language", "label"]
        report = {}
        for c in required:
            if c not in df.columns:
                report["error"] = f"Missing required column: {c}"
                return False, report

        # Missing values
        missing = df[required].isnull().sum().to_dict()
        report["missing_values"] = missing

        # Class balance
        counts = df["label"].value_counts().to_dict()
        report["class_counts"] = counts

        # Language distribution
        langs = df["language"].value_counts().to_dict()
        report["language_counts"] = langs

        # Basic length stats for text
        lengths = df["email_text"].astype(str).apply(len)
        report["text_len_mean"] = float(lengths.mean())
        report["text_len_max"] = int(lengths.max())
        report["text_len_min"] = int(lengths.min())

        # Simple validity: non-empty and labels only 0/1
        if df["label"].isin([0, 1]).all() and len(df) > 10:
            return True, report
        report["error"] = "Invalid labels or too few rows"
        return False, report

    # ---------------- Callbacks ----------------
    def _make_callbacks(self):
        cb = []
        # Early stopping
        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.training.early_stopping_patience or 5,
            restore_best_weights=True,
            verbose=1,
        )
        cb.append(es)

        # Reduce LR on plateau
        rl = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.config.training.lr_patience or 3,
            min_lr=1e-6,
            verbose=1,
        )
        cb.append(rl)

        # Model checkpoint
        ckpt = callbacks.ModelCheckpoint(
            filepath=str(self.model_dir / "model_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        cb.append(ckpt)

        # TensorBoard (optional)
        try:
            tb_logdir = Path(self.config.paths.logs or "logs") / "tensorboard"
            tb_logdir.mkdir(parents=True, exist_ok=True)
            tb = callbacks.TensorBoard(log_dir=str(tb_logdir))
            cb.append(tb)
        except Exception:
            pass

        return cb

    # ---------------- Full pipeline ----------------
    def run_full_pipeline(self):
        """Run data load → validate → build → train → evaluate."""
        # Load data via PhishingDataLoader (expects a prepare_full_pipeline method)
        data_path = self.config.data.path or "data/processed/phishing_data.csv"
        loader = PhishingDataLoader(max_seq_len=self.config.model.max_seq_len,
                                    vocab_size=self.config.model.vocab_size)
        print(f"Loading data from: {data_path}")
        # PhishingDataLoader exposes `load_dataset` — use that
        df = loader.load_dataset(data_path)

        valid, report = self.validate_dataset(df)
        print("Dataset validation report:")
        print(json.dumps(report, indent=2))
        if not valid:
            raise RuntimeError(f"Dataset validation failed: {report.get('error')}")

        # Prepare tf.data datasets
        batch_size = int(self.config.training.batch_size or 32)
        train_ds, val_ds, test_ds = loader.prepare_full_pipeline(
            data_path,
            batch_size=batch_size,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
        )

        # Build model
        model = build_hybrid_cnn_lstm(config_path="config/default.yaml",
                                      vocab_size=loader.vocab_size if hasattr(loader, "vocab_size") else None)

        # Compile model (use optimizer from config or Adam)
        lr = float(self.config.training.learning_rate or 1e-3)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        # Callbacks
        cbs = self._make_callbacks()

        # Fit
        epochs = int(self.config.training.epochs or 10)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)

        # Save final model and metadata
        final_path = self.model_dir / f"model_final.keras"
        model.save(str(final_path))
        metadata = {
            "model_path": str(final_path),
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        }
        with open(self.model_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("Training complete — model saved to:", final_path)
        return {"status": "ok", "model_path": str(final_path)}

