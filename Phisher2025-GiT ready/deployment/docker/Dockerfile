# Minimal Dockerfile for Phisher2025
# Builds a small Python image containing the app and (optionally) the model

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system deps (if needed for tokenizers or TF; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Ensure models directory exists (image may include model if you placed it in repo)
RUN mkdir -p /app/models/final_model

# Default external model dir inside container
ENV EXTERNAL_MODEL_DIR=/app/models/final_model

EXPOSE 8080

# Run uvicorn by default. Override CMD or pass env var to run different server.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
