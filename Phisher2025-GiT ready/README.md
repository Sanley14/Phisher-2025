# Phisher2025 — Multilingual Phishing Email Detection

A production-ready prototype for detecting phishing emails across multiple languages using a hybrid CNN-LSTM deep-learning model with a REST API and Streamlit web UI.

## Features

- **Multilingual Support**: Detects phishing across English, Swahili, Chinese, and Russian
- **Hybrid CNN-LSTM Model**: Combines convolutional and recurrent neural networks for robust email classification
- **REST API**: FastAPI endpoints for single and batch predictions with full OpenAPI documentation
- **Web UI**: Streamlit interface for interactive email analysis with social-engineering heuristics
- **External Dataset Support**: Generate synthetic multilingual data or load custom CSV datasets
- **Model Evaluation**: Batch evaluation on external datasets with per-language metrics
- **Model Artifact Management**: Rebuilt Keras `.keras` format model for reliability
- **Production Ready**: Health checks, batch predictions, configurable thresholds, and comprehensive logging

---

## Prerequisites

- **Python 3.10.x** (tested on 3.10.11)
- **pip** or **conda** for package management
- **~6 GB disk space** for model artifacts and dependencies (or use external storage via `EXTERNAL_MODEL_DIR`)
- **Windows PowerShell 5.1+** (or bash/zsh on Linux/Mac)

---

## Quick Start (Local Development)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Phisher2025.git
cd Phisher2025

# Create and activate a Python 3.10 virtual environment
python -m venv .venv310
.\.venv310\Scripts\Activate.ps1  # PowerShell on Windows
# or
source .venv310/bin/activate  # Bash on Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Model Artifact

The prototype uses a rebuilt Keras model at `models/final_model/model_rebuilt.keras`. Check it exists:

```powershell
Test-Path .\models\final_model\model_rebuilt.keras
# Should output: True
```

If it doesn't exist, you can rebuild it from the original H5:

```powershell
python .\tools\rebuild_model_from_weights.py --h5 .\models\final_model\model.h5 --out .\models\final_model\model_rebuilt.keras
```

### 3. Run Backend (FastAPI)

In one terminal:

```powershell
# Activate venv
.\.venv310\Scripts\Activate.ps1

# Set the model path (optional; defaults to model.h5)
$env:MODEL_PATH = "models\final_model\model_rebuilt.keras"

# Start backend (default: http://127.0.0.1:8000; if unavailable, use 8080)
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

If port 8000 is restricted, use port 8080 instead:
```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8080 --reload
```

**API endpoints:**
- `GET /health` — health check (http://127.0.0.1:8000/health or :8080/health)
- `POST /predict` — single email prediction
- `POST /predict-batch` — batch email predictions
- `GET /docs` — interactive API documentation (Swagger UI)
- `GET /redoc` — ReDoc documentation

### 4. Run Streamlit UI

In another terminal:

```powershell
# Activate venv
.\.venv310\Scripts\Activate.ps1

# Start UI on http://localhost:8501
streamlit run src/agent_interface/chat_ui.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

Then open http://localhost:8501 in your browser. The UI will:
- Automatically detect and use the rebuilt `.keras` model
- Show a badge indicating model status
- Offer buttons to switch between `.keras` and `.h5` models
- Allow pasting or uploading emails for analysis
- Provide social-engineering heuristics and model predictions
- Support external CSV datasets for batch evaluation

---

## Example Usage

### Via REST API (curl)

**Health check:**
```bash
curl http://127.0.0.1:8000/health
```

**Single prediction:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent: Verify your PayPal account immediately. Click here: bit.ly/verify"}'
```

**Response:**
```json
{
  "raw_score": 0.87,
  "label": "PHISHING",
  "confidence": 0.87,
  "is_phishing": true
}
```

### Via Streamlit UI

1. Open http://localhost:8501
2. Paste an email in the text area or upload a `.txt` file
3. View model predictions, heuristic analysis, and recommendations
4. Use sidebar to evaluate on external datasets or generate synthetic data

---

## Working with External Storage

If you have limited disk space, use an external drive for model checkpoints and datasets:

```powershell
# Set external storage (e.g., F: drive)
$env:EXTERNAL_MODEL_DIR = "F:\phisher_artifacts"

# Then run the backend/UI as usual
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

---

## Generate Synthetic Dataset

To create a multilingual synthetic dataset for testing:

```powershell
# Via CLI
python .\tools\create_external_dataset.py --output "F:\phisher_data.csv" --samples-per-lang 500

# Or via Streamlit UI
# Settings sidebar → "Create synthetic dataset on external path" button
```

**Output format:** CSV with columns: `email_text`, `language`, `label` (0=legitimate, 1=phishing)

---

## Model Architecture

- **Embedding**: Trainable embedding layer (20K vocabulary, 128 dimensions)
- **CNN Branch**: Conv1D (128 filters, kernel=5) + MaxPooling1D
- **LSTM Branch**: Bidirectional LSTM (64 units)
- **Output**: Dense + Dropout + Sigmoid for binary classification

---

## Project Structure

```
Phisher2025/
├── src/
│   ├── agent_interface/     # Streamlit UI
│   │   └── chat_ui.py
│   ├── api/                  # FastAPI backend
│   │   ├── app.py
│   │   └── routes.py
│   ├── data/                 # Data loading & generation
│   │   ├── load_data.py
│   │   ├── generate_synthetic_data.py
│   │   └── preprocess.py
│   ├── model/                # Model architecture & training
│   │   ├── build_model.py
│   │   ├── train_baseline.py
│   │   └── evaluate.py
│   ├── inference/            # Prediction utilities
│   │   └── predict.py
│   └── utils/                # Config & logging
│       ├── config_loader.py
│       └── logging.py
├── config/                   # Configuration files (YAML)
│   ├── default.yaml
│   └── multilingual.yml
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed datasets
│   └── interim/              # Intermediate data
├── models/
│   ├── final_model/
│   │   ├── model.h5          # Original model (HDF5 format)
│   │   ├── model_rebuilt.keras  # Rebuilt Keras format (preferred)
│   │   └── model.h5.bak      # Backup
│   └── checkpoints/          # Training checkpoints
├── tools/                    # Utilities & helpers
│   ├── create_external_dataset.py
│   ├── rebuild_model_from_weights.py
│   └── repair_model.py
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── ci_cd/
├── tests/                    # Unit & integration tests
├── notebooks/                # Jupyter notebooks (EDA, experiments)
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `models/final_model/model.h5` | Path to model artifact (set to `.keras` for rebuilt model) |
| `EXTERNAL_MODEL_DIR` | (unset) | External storage path for checkpoints/logs when disk space is limited |
| `STREAMLIT_SERVER_PORT` | `8501` | Port for Streamlit UI |

---

## Troubleshooting

### "Model could not be loaded"

**Symptom:** Streamlit UI shows ⚠️ (model not loaded).

**Solution:**
1. Verify model file exists: `Test-Path .\models\final_model\model_rebuilt.keras`
2. In the UI sidebar, click "Use rebuilt model" to switch from `.h5` to `.keras`
3. Refresh the page (F5)
4. If still failing, check backend logs for error messages

### "No space left on device"

**Symptom:** Training or evaluation fails with `[Errno 28]`.

**Solution:**
1. Free up disk space on C:, or
2. Use `EXTERNAL_MODEL_DIR` to redirect writes to a larger drive:
   ```powershell
   $env:EXTERNAL_MODEL_DIR = "F:\phisher_artifacts"
   ```

### Backend fails to start ("Address already in use")

**Symptom:** Port 8000 is already in use.

**Solution:**
```powershell
# Kill existing Python process on port 8000
Get-Process python | Stop-Process -Force

# Or use a different port
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 9000
```

### Streamlit not found

**Symptom:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```powershell
pip install streamlit
```

---

## Advanced: Rebuild Model from Weights

If the `.keras` model is missing or corrupted, rebuild it from the original H5:

```powershell
python .\tools\rebuild_model_from_weights.py \
  --h5 .\models\final_model\model.h5 \
  --out .\models\final_model\model_rebuilt.keras
```

This script:
1. Infers the vocabulary size from the H5 file
2. Rebuilds the canonical CNN-LSTM architecture
3. Loads weights from the H5 (by name, then by index)
4. Saves a new `.keras` model file

---

## Next Steps (Roadmap)

- [ ] Add GitHub Actions CI/CD (tests, lint, build)
- [ ] Dockerize the API for easy deployment
- [ ] Add model versioning and artifact registry
- [ ] Implement retraining pipeline
- [ ] Add monitoring and observability (Prometheus, Grafana)
- [ ] Integrate explainability (SHAP, attention visualizations)
- [ ] Adversarial robustness testing
- [ ] Multi-model ensemble support

---

## Contributing

Contributions are welcome! Please open an issue or submit a PR with:
- Clear description of changes
- Unit tests for new features
- Updated documentation

---

## License

[Your License Here — e.g., MIT, Apache 2.0]

---

## Support & Questions

For issues, questions, or feedback:
- Open a GitHub issue
- Check existing issues for similar problems
- Review the `docs/` folder for detailed guides

---
# Terminal 1: Backend
.\.venv310\Scripts\Activate.ps1
$env:MODEL_PATH = "models\final_model\model_rebuilt.keras"
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8080 --reload

# Terminal 2: UI
.\.venv310\Scripts\Activate.ps1
streamlit run src/agent_interface/chat_ui.py --server.port 8501


**Built with ❤️ using TensorFlow, FastAPI, and Streamlit**
