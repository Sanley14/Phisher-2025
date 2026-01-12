# Phisher2025 - Production Readiness Checklist Before Cloud Training

**Goal**: Ensure the model is robust, well-documented, reproducible, and optimized before deploying to cloud infrastructure for tuning/scaling.

---

## ✅ CORE MODEL COMPLETENESS (CRITICAL)

### Model Architecture & Training
- [x] Model architecture defined (CNN-LSTM hybrid) in `src/model/build_model.py`
- [x] Config system working (`src/utils/config_loader.py`)
- [x] Trainer class created (`src/model/train_baseline.py`)
- [x] GPU detection & memory growth enabled
- [ ] **TODO**: Add mixed-precision training for GPU efficiency
  - Add `tf.keras.mixed_precision.Policy('mixed_float16')` in trainer
  - Reduces memory use by ~2x; trains faster on modern GPUs
  - Quick fix: 5 lines in `_configure_gpu()`

- [ ] **TODO**: Add learning rate scheduling
  - Improves convergence; prevents divergence in later epochs
  - Add `ReduceLROnPlateau` or `CosineDecayScheduler` callback
  - Quick fix: ~10 lines in trainer

- [ ] **TODO**: Add loss weighting for class imbalance
  - If dataset is imbalanced (more phishing than legit), use `class_weight` in `fit()`
  - Quick check: run `check_class_balance.py` helper
  - Quick fix: 1 line in `model.fit()`

- [ ] **TODO**: Hyperparameter sweep / tuning documentation
  - Add `HYPERPARAMETER_TUNING.md` with suggested ranges (learning rate, dropout, batch size)
  - Document which params are most impactful for your dataset/hardware
  - Reference: start with LR=1e-3, dropout=0.3-0.5, batch=32-64

---

## ✅ DATA PIPELINE (CRITICAL)

### Data Loading & Validation
- [x] Data loader exists (`src/data/load_data.py`)
- [x] Synthetic data generator works (`src/data/generate_synthetic_data.py`)
- [x] CSV helper to add samples (`tools/add_samples.py`)
- [ ] **TODO**: Data validation & stats report
  - Add script to check: class balance, sequence length distribution, missing values, language distribution
  - Output a `data/stats_report.json` before training
  - Quick fix: 30 lines in `tools/validate_dataset.py`

- [ ] **TODO**: Add data augmentation (on-the-fly)
  - Apply random noise, synonym swaps during training
  - Improves generalization and reduces overfitting
  - Add 1-2 augmentation layers in data pipeline or via tf.data

- [ ] **TODO**: Verify tokenizer is cached and reproducible
  - Ensure the same tokenizer is used at train and inference time
  - Save tokenizer config/weights with the model
  - Check: `PhishingDataLoader` saves tokenizer metadata

- [ ] **TODO**: Add train/val/test split reproducibility
  - Set random seed in `train_baseline.py` (line ~50)
  - Use `random_state=42` in all sklearn splits
  - Quick fix: 2 lines

---

## ✅ TRAINING & EVALUATION (CRITICAL)

### Training Pipeline
- [x] Trainer has checkpoint callbacks
- [ ] **TODO**: Add early stopping with patience
  - Prevents overfitting; saves best model automatically
  - Check if `EarlyStopping` is in trainer callbacks; if not, add with patience=5-10
  - Quick fix: 3 lines

- [ ] **TODO**: Log training metrics to file (JSON/CSV)
  - Save `history.json` with train/val loss & metrics per epoch
  - Needed for cloud training dashboards (e.g., W&B, TensorBoard)
  - Quick fix: ~10 lines at end of trainer

- [ ] **TODO**: Add per-language evaluation metrics
  - After training, evaluate on test set split by language
  - Report: accuracy, precision, recall, F1 per language + overall
  - Output: `logs/evaluation_report_<TIMESTAMP>.json`
  - Implementation: ~30 lines in `src/model/evaluate.py` or trainer

- [ ] **TODO**: Confusion matrix + ROC curve per language
  - Visualize model mistakes
  - Save to `logs/` as PNG for review
  - Optional but recommended: ~20 lines

### Model Artifacts
- [x] Model saves in `.keras` format (preferred over H5)
- [ ] **TODO**: Save model metadata with version
  - Create `model_metadata.json`:
    ```json
    {
      "version": "1.0.0",
      "created_at": "2025-11-19T...",
      "training_config": {...},
      "data_stats": {...},
      "metrics": {...},
      "git_commit": "abc123",
      "tensorflow_version": "2.20.0",
      "trained_on": "GPU/CPU"
    }
    ```
  - Quick fix: ~20 lines in trainer after model.save()

- [ ] **TODO**: Save best & final models
  - Keep track of which checkpoint is "best" (by val loss) vs "final" (last epoch)
  - Name: `model_best.keras`, `model_final.keras`
  - Quick fix: 2 lines

---

## ✅ REPRODUCIBILITY & DOCUMENTATION (HIGH)

### Code Quality
- [x] Code has docstrings (partial — check `train_baseline.py`)
- [ ] **TODO**: Add type hints to all public functions
  - Especially in `train_baseline.py`, `build_model.py`, `load_data.py`
  - Quick fix: ~20 lines

- [ ] **TODO**: Add inline comments for key model decisions
  - Why CNN + LSTM? Why mask_zero? Why dropout=0.5?
  - Add 1-2 line comments above each layer
  - Quick fix: ~10 lines in `build_model.py`

### Configuration & Reproducibility
- [x] Config file in YAML (`config/default.yaml`)
- [x] **DONE**: Add config schema/validation
  - Created `src/utils/config_schema.py` to validate required keys
  - Prevents silent failures from typos in config
  - Implemented with `pydantic` for robust type and value checking

- [ ] **TODO**: Save config with trained model
  - When saving model, also save the config used
  - Ensure reproducibility: `model_config.yaml` in model artifact folder
  - Quick fix: ~5 lines in trainer

- [ ] **TODO**: Document all hyperparameters
  - In `config/default.yaml`, add comments explaining each param
  - Include recommended ranges and why they matter
  - Quick fix: ~50 lines

### Version Control
- [ ] **TODO**: Tag current state as `v1.0-base`
  - Commit all code: `git add -A && git commit -m "v1.0: Base model before cloud tuning"`
  - Tag: `git tag -a v1.0-base -m "Production baseline model"`
  - Quick fix: 2 commands

- [ ] **TODO**: Document git workflow
  - Branch for cloud experiments: `git checkout -b cloud/tuning-v1`
  - Keep main branch stable for serving

---

## ✅ INFERENCE & SERVING (HIGH)

### Model Loading & Inference
- [x] Predictor class exists (`src/inference/predict.py`)
- [ ] **TODO**: Add inference benchmarking
  - Measure latency: time per prediction (single & batch)
  - Add `tools/benchmark_inference.py`: run 1000 predictions, report stats (mean, p95, p99 latency)
  - Needed to set SLA for cloud deployment
  - Quick fix: ~30 lines

- [ ] **TODO**: Add inference error handling
  - What if input is empty? Wrong language? Adversarial? OOV tokens?
  - Add try/except with meaningful error messages
  - Quick fix: ~10 lines in predictor

- [ ] **TODO**: Batch inference optimization
  - Ensure `predict_batch()` is efficient (vectorized, not loop)
  - Test: batch of 100 should be ~10x faster than 100 singles
  - Quick fix: check if using tf.data.Dataset or numpy batching

### API & Deployment Readiness
- [x] FastAPI app exists (`src/api/app.py`)
- [x] Streamlit UI exists (`src/agent_interface/chat_ui.py`)
- [ ] **TODO**: Add API request/response validation
  - Ensure input validation (e.g., max text length)
  - Return consistent error responses
  - Quick fix: ~10 lines in `/predict` endpoint

- [ ] **TODO**: Add API versioning
  - Use URL versioning: `/v1/predict` (future-proof for `/v2/predict`)
  - Quick fix: ~5 lines in `app.py`

- [ ] **TODO**: Add request logging & metrics
  - Log: timestamp, input length, prediction time, output
  - Save to file: `logs/api_requests.log`
  - Later: connect to Prometheus for monitoring
  - Quick fix: ~15 lines in middleware

---

## ✅ CLOUD READINESS (HIGH)

### Containerization
- [ ] **TODO**: Create `Dockerfile` for training
  ```dockerfile
  FROM tensorflow/tensorflow:latest-gpu-jupyter
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["python", "-m", "src.model.train_baseline"]
  ```
  - Quick fix: ~10 lines

- [ ] **TODO**: Create `Dockerfile` for inference (API)
  - Minimal image (no training deps)
  - Single entrypoint: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
  - Quick fix: ~15 lines

- [ ] **TODO**: Create `docker-compose.yml` for local testing
  - Spin up API + Streamlit + optional DB in one command
  - Quick fix: ~30 lines

### Environment & Dependencies
- [x] `requirements.txt` exists
- [ ] **TODO**: Pin all versions
  - Current: likely has some unpinned deps
  - Run: `pip freeze > requirements.txt` (or use Poetry/pipenv for lock files)
  - Important for cloud reproducibility

- [ ] **TODO**: Create `requirements-dev.txt` for local dev
  - Includes: pytest, black, flake8, jupyter, tensorboard, wandb
  - Quick fix: ~10 lines

- [ ] **TODO**: Document GPU requirements
  - Minimum GPU VRAM needed (estimate based on batch size)
  - Suggested cloud SKU (e.g., NVIDIA A100 40GB, V100 16GB, etc.)
  - Add to README or `CLOUD_DEPLOYMENT.md`

### Monitoring & Logging
- [ ] **TODO**: Add structured logging (JSON format)
  - Use `python-json-logger` or similar
  - Enables log aggregation in cloud (e.g., CloudWatch, Stackdriver)
  - Quick fix: ~20 lines

- [ ] **TODO**: Add TensorBoard integration
  - Save logs to TensorBoard format in trainer
  - Cloud can auto-display training curves
  - Quick fix: 1 line (`callbacks.TensorBoard()`)

- [ ] **TODO**: Add model artifact versioning
  - Save models with timestamp or version: `model_v1.0_2025-11-19.keras`
  - Enable rollback if needed
  - Quick fix: 1 line in trainer save path

---

## ✅ TESTING (MEDIUM)

### Unit Tests
- [ ] **TODO**: Add test for data loading
  - File: `tests/test_data_loading.py`
  - Test: Load CSV, check shape, verify no NaNs
  - Quick fix: ~20 lines

- [ ] **TODO**: Add test for model building
  - File: `tests/test_model_build.py`
  - Test: Build model, check layer names, verify shapes
  - Quick fix: ~15 lines

- [ ] **TODO**: Add test for inference
  - File: `tests/test_inference.py`
  - Test: Load rebuilt model, predict on dummy input, check output shape & range
  - Quick fix: ~20 lines

### Integration Tests
- [ ] **TODO**: Add E2E test
  - File: `tests/test_e2e.py`
  - Test: Load data → build model → train 1 epoch → evaluate → save → load → predict
  - Takes ~2-3 mins to run, catches major breakages
  - Quick fix: ~30 lines

### Running Tests
- [ ] **TODO**: Add `pytest.ini` or configure pytest in `pyproject.toml`
- [ ] **TODO**: Add GitHub Actions workflow to run tests on push
  - File: `.github/workflows/test.yml`
  - Quick fix: ~20 lines

---

## ✅ DOCUMENTATION (MEDIUM)

### Code Documentation
- [ ] **TODO**: Update `README.md`
  - Add sections: Architecture, Quickstart, Cloud Deployment, Hyperparameter Tuning
  - Quick fix: ~50 lines

- [ ] **TODO**: Create `CLOUD_DEPLOYMENT.md`
  - Step-by-step for deploying to AWS/GCP/Azure
  - Environment variables, data paths, GPU setup
  - Quick fix: ~100 lines

- [ ] **TODO**: Create `HYPERPARAMETER_TUNING.md`
  - Explain each hyperparameter
  - Recommended ranges
  - How to run grid/random search
  - Quick fix: ~50 lines

- [ ] **TODO**: Create `MODEL_ARCHITECTURE.md`
  - ASCII diagram of CNN-LSTM
  - Why each choice (mask_zero, dropout, LSTM units, etc.)
  - Quick fix: ~30 lines

---

## 🚀 PRIORITY ORDER (Complete in this order)

### Phase 1: CRITICAL (Do first — enables training)
1. Add mixed-precision training (5 min) — saves GPU memory & time
2. Add learning rate scheduling (10 min) — better convergence
3. Add class weighting check (5 min) — handles imbalance
4. Add early stopping if missing (5 min) — prevents overfitting
5. Add data validation script (30 min) — catches data issues early
6. Save model metadata (15 min) — reproducibility

**Time: ~1.5 hours | Impact: HIGH** ✅

### Phase 2: IMPORTANT (Do next — ensures reproducibility)
7. Add random seed & reproducibility (5 min)
8. Save config with model (5 min)
9. Per-language evaluation metrics (45 min)
10. Tag v1.0-base and commit (5 min)
11. Add type hints to core modules (30 min)
12. Add inline comments to build_model.py (15 min)

**Time: ~2 hours | Impact: HIGH** ✅

### Phase 3: CLOUD READY (Do before uploading)
13. Inference benchmarking script (30 min)
14. Docker setup (training + inference) (45 min)
15. Environment versioning (pin requirements.txt) (10 min)
16. Add structured logging (20 min)
17. Add TensorBoard callback (5 min)

**Time: ~1.5 hours | Impact: MEDIUM** ✅

### Phase 4: NICE TO HAVE (After cloud runs are working)
18. Unit tests (tests/ directory) (60 min)
19. E2E test (30 min)
20. Complete documentation (CLOUD_DEPLOYMENT.md, HYPERPARAMETER_TUNING.md) (90 min)

**Time: ~3 hours | Impact: MEDIUM**

---

## 📋 QUICK ACTION PLAN

**If you have 1 hour**: Do Phase 1 (mixed precision, LR scheduling, class weight, early stopping, data validation).

**If you have 3 hours**: Do Phase 1 + Phase 2 (add reproducibility, metrics, git tag, type hints).

**If you have 5+ hours**: Do Phase 1 + Phase 2 + Phase 3 (fully cloud-ready: Docker, logging, benchmarks).

---

## 🎯 SUCCESS CRITERIA FOR CLOUD DEPLOYMENT

Before uploading to cloud, ensure:
- ✅ Model trains end-to-end without errors
- ✅ Achieves >75% accuracy on test set (or define your SLA)
- ✅ Data is validated (class balance, no NaNs, correct format)
- ✅ Hyperparameters are documented
- ✅ Git is tagged with version (v1.0-base or similar)
- ✅ Requirements.txt is pinned
- ✅ Dockerfile builds and runs locally
- ✅ Config with model is saved (metadata)
- ✅ Per-language metrics are calculated
- ✅ Inference latency is benchmarked (<100ms single, <1s batch-100 is good)

---

## 💡 TIPS FOR CLOUD TRAINING

1. **Use spot instances** (GCP Preemptible, AWS Spot) for 70% cost savings during tuning.
2. **Stream data from cloud storage** (GCS/S3) rather than copying to instance.
3. **Log to cloud storage** (GCS/S3) automatically via `model_callbacks.py`.
4. **Monitor via TensorBoard** — easier than SSH + logs.
5. **Save best checkpoints to cloud** — continue training from latest if interrupted.
6. **Use distributed training** (multi-GPU or multi-node) for large datasets.
7. **Track experiments** with Weights & Biases or MLflow — enables reproducible tuning.

---

## 📦 DELIVERABLES FOR CLOUD

Before upload, prepare:
- [ ] `models/model_v1.0.keras` (trained baseline)
- [ ] `models/model_metadata.json` (config, metrics, timestamp)
- [ ] `config/default.yaml` (with comments)
- [ ] `requirements.txt` (pinned)
- [ ] `Dockerfile` (for training & inference)
- [ ] `docker-compose.yml` (local test)
- [ ] `CLOUD_DEPLOYMENT.md` (setup steps)
- [ ] `HYPERPARAMETER_TUNING.md` (ranges & tips)
- [ ] `.github/workflows/test.yml` (CI)
- [ ] `logs/evaluation_report_v1.0.json` (baseline metrics)

Total prep time: **~5 hours** for a complete, cloud-ready setup.

---

## 🔗 References

- [TensorFlow Mixed Precision](https://www.tensorflow.org/guide/mixed_precision)
- [Learning Rate Schedules](https://www.tensorflow.org/guide/keras/optimizers#learning_rate_schedules)
- [Class Weights in Keras](https://keras.io/api/models/model_training_apis/#fit-method)
- [TensorBoard Profiling](https://www.tensorflow.org/tensorboard/profiling_keras_model)
- [Docker for ML](https://docs.docker.com/develop/dev-best-practices/)
