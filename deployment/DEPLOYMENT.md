Deployment guide for Phisher2025

This document explains how to build a Docker image locally, include the trained model into the image, run the container, and perform a smoke-test.

1) Confirm model artifacts
- The trainer saves artifacts to `models/final_model/`:
  - `model_best.keras` (checkpoint)
  - `model_final.keras` (final model)
  - `model_metadata.json`

2) Build image locally (including model)
If you want the model baked into the image, ensure `models/final_model/model_best.keras` is present in the repository root before building.

PowerShell example:
```powershell
# Build image (tags: phisher:local)
docker build -t phisher:local -f deployment/docker/Dockerfile .

# Run container, mapping model path and exposing port 8080
docker run --rm -p 8080:8080 -e EXTERNAL_MODEL_DIR="/app/models/final_model" phisher:local
```

3) Run container that downloads model at start (alternative)
If you prefer not to bake the model into the image, upload the model to object storage (S3/GCS/Azure Blob), then modify your deploy/startup to download the model at container start. Example using AWS S3 CLI:
```powershell
# Build image without model baked in
docker build -t phisher:local -f deployment/docker/Dockerfile .

# Run container and pass environment variables needed to download model on start (app must support it)
docker run --rm -p 8080:8080 -e MODEL_S3_URI="s3://my-bucket/models/phisher/model_best.keras" -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... phisher:local
```

4) Smoke test the running service
Once the container is up, check the health and a prediction endpoint.

PowerShell (curl):
```powershell
# health
curl http://127.0.0.1:8080/health

# sample predict (adjust JSON schema to api)
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d '{"text":"You won a prize, click here","language":"en"}'
```

5) CI/CD pipeline
- See `deployment/ci_cd/pipeline.yml` for a template pipeline: run tests, build image, push to registry, deploy.
- Fill in registry secrets and provider-specific deploy steps.

6) Observability & readiness
- The API exposes `/health` for basic health checks. Add Prometheus metrics and structured logs in production.

7) Security
- Do not store secrets in the repo; use your provider's secret store.
- Scan Python dependencies and keep minimal privileges for IAM roles accessing model storage.

If you want, I can:
- bake the model into the image and build/test it locally now,
- or modify the app to download model from S3 at startup (I can implement an `EXTERNAL_MODEL_DIR` handler that understands `s3://` URIs).
