# Phisher2025 - Prototype Roadmap to Saturday

**Target**: Full working prototype by Saturday with commented code
**Current Status**: Model building works ✅ | Data pipeline needed ⏳

---

## High-Level Architecture

```
User Email → API → Model Inference → Prediction (Phishing/Legitimate)
                      ↑
                   Trained CNN-LSTM Model
                      ↑
                  Training Pipeline
                      ↑
              Multilingual Dataset (4 languages)
```

---

## Phased Implementation Plan

### Phase 1: Data & Infrastructure (Friday 8 hours)
**Goal**: Get synthetic data and training pipeline ready

#### 1.1 Create Synthetic Dataset (2 hours)
- **File**: `src/data/generate_synthetic_data.py`
- **Output**: `data/processed/phishing_data.csv`
- **Content**: 
  - 250 samples per language × 4 languages = 1000 samples
  - Languages: English, Swahili, Chinese, Russian
  - Features: [email_text, language, label (0=legitimate, 1=phishing)]
  - Quick solution: Use templates with variations (URL patterns, urgency keywords, etc.)

#### 1.2 Data Loading Pipeline (1 hour)
- **File**: `src/data/load_data.py`
- **Key Functions**:
  ```python
  - load_dataset(path) → pandas DataFrame
  - tokenize_texts(texts, max_len=256) → token sequences
  - split_data(X, y, test_size=0.2) → train/val/test
  - create_batches(data, batch_size=32) → tf.data.Dataset
  ```

#### 1.3 Preprocessing Module (1 hour)
- **File**: `src/data/preprocess.py`
- **Functions**:
  ```python
  - clean_text(text) → clean string
  - get_tokenizer() → transformer tokenizer
  - pad_sequences(seqs, max_len) → padded arrays
  ```

#### 1.4 Simple Adversarial Augmentation (1 hour)
- **File**: `src/data/augment_adversarial.py`
- **Methods**: 
  - Synonym replacement (common phishing words)
  - Number-to-word substitution (e.g., "0" → "O")
  - URL obfuscation (e.g., "amazon.com" → "am@z0n.c0m")

#### 1.5 Configuration System (1 hour)
- **Update**: `config/default.yaml` with paths and hyperparameters
- **Already done**: `src/utils/config_loader.py` ✅

#### 1.6 Logging & Utilities (1 hour)
- **File**: `src/utils/logging.py` - Simple logging setup
- **File**: `src/utils/metrics.py` - Accuracy, Precision, Recall, F1

---

### Phase 2: Model Training (Friday 4 hours)
**Goal**: Train and save model checkpoint

#### 2.1 Complete Training Script (3 hours)
- **File**: `src/model/train_baseline.py`
- **Checklist**:
  - ✅ Load config
  - ✅ Load dataset
  - ✅ Build model (already have `build_model.py`)
  - ⏳ Set random seed
  - ⏳ Create train/val/test splits
  - ⏳ Setup callbacks (ModelCheckpoint, EarlyStopping)
  - ⏳ Train model (20-50 epochs)
  - ⏳ Save best model to `models/final_model/`
  - ⏳ Log training history

#### 2.2 Evaluation Module (1 hour)
- **File**: `src/model/evaluate.py`
- **Outputs**: Metrics per language + overall metrics
- **Save**: `logs/evaluation_results.json`

---

### Phase 3: Inference & API (Saturday 3 hours)
**Goal**: Make model accessible via REST API

#### 3.1 Prediction Module (1 hour)
- **File**: `src/inference/predict.py`
- **Functions**:
  ```python
  - load_model(path) → model
  - predict_email(text, model) → (score, label)
  - batch_predict(texts, model) → predictions
  ```

#### 3.2 FastAPI Backend (1.5 hours)
- **File**: `src/api/app.py` & `src/api/routes.py`
- **Endpoints**:
  - `POST /predict` - Predict single email
  - `GET /health` - API health check
  - `GET /model-info` - Model metadata

#### 3.3 Web UI (Streamlit) (0.5 hours)
- **File**: `src/agent_interface/chat_ui.py`
- **Features**:
  - Text input for email
  - Real-time prediction
  - Confidence score display
  - Language selection

---

### Phase 4: Testing & Documentation (Saturday 2 hours)
**Goal**: Ensure everything works end-to-end

#### 4.1 Unit Tests (1 hour)
- **File**: `tests/` directory
- **Coverage**:
  - Data loading
  - Preprocessing
  - Model prediction
  - API endpoints

#### 4.2 Documentation (1 hour)
- **Update**: README.md with:
  - Setup instructions
  - Usage guide (training, inference, API)
  - Architecture diagram
  - Results summary

---

## Quick Implementation Tips

### ✅ Things Already Done
- ✅ Project structure
- ✅ Configuration loader (`src/utils/config_loader.py`)
- ✅ Model architecture (`src/model/build_model.py`)
- ✅ Environment setup (venv with TensorFlow, Keras, FastAPI)

### ⏳ Priority Order (Most → Least)
1. **Synthetic dataset** (blocks everything)
2. **Data loading & preprocessing** (needed for training)
3. **Training script** (get model working)
4. **Evaluation** (validate model quality)
5. **API** (make it accessible)
6. **Web UI** (nice to have)
7. **Tests** (documentation)

### 💡 Code Style
- Add docstrings to all functions
- Use type hints (`def func(x: str) -> bool:`)
- Add inline comments explaining logic
- Use meaningful variable names
- Keep functions small and focused

---

## Timeline Summary

```
Friday Evening (8 hours):
├── 2 hrs: Synthetic dataset
├── 1 hr:  Data loading
├── 1 hr:  Preprocessing  
├── 1 hr:  Adversarial aug
├── 1 hr:  Config/Utils
├── 3 hrs: Training + Eval

Saturday Morning (6 hours):
├── 1 hr:  Prediction module
├── 1.5 hrs: FastAPI
├── 0.5 hrs: Streamlit UI
├── 1 hr:  Tests
└── 1 hr:  Documentation
Also Today 
Immediate (safe, low-effort — do these first)

 # Use mask_zero so subsequent Masking layers / Keras RNNs can detect padding.... Embedding

Commit & tag the prototype (snapshot + backup artifacts).
Add a few unit tests and an API health E2E test so CI can catch regressions.
Create a GitHub Actions workflow to run tests + lint on PRs.
Near-term (make it production-ready)

Docker image + publish to container registry.
Model artifact management (don’t store big .h5 in git; move to storage).
Add monitoring/logging and an API auth layer.
Mid-term (robust ops & model management)

Retraining pipeline (scheduled), dataset versioning (DVC or timestamped CSVs).
Performance tuning (GPU training, mixed precision, batching).
Staging deployment and load testing.
Long-term (hardening & scale)

Model registry, A/B experiments, explainability (SHAP), adversarial testing, observability dashboards, and official docs.

Complete the UI and Ensure that It Meets Development Standards
```

---

## Success Criteria

- ✅ Model trains without errors
- ✅ Achieves >80% accuracy on test set
- ✅ Handles multilingual inputs
- ✅ API responds to requests
- ✅ Can make predictions on new emails
- ✅ Well-commented code
- ✅ README with usage instructions

---

## Sample Commands (Once Implemented)

```bash
# Training
python -m src.model.train_baseline --config config/default.yaml

# Evaluation
python -m src.model.evaluate --model models/final_model/

# API
uvicorn src.api.app:app --reload --port 8000

# Streamlit UI
streamlit run src/agent_interface/chat_ui.py
```

---

## Notes

- Use simple implementations (no fancy tricks)
- Focus on getting it working first
- Comment everything for clarity
- Test each module as you build it
- Keep dataset small (1000 samples is enough for demo)
- Use TensorFlow/Keras 3.x APIs consistently

