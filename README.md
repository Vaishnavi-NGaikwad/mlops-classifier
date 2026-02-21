# 🐾 Pet Classifier — MLOps Pipeline

A full end-to-end MLOps project that trains a Cat vs Dog image classifier and deploys it through a fully automated CI/CD pipeline with monitoring.

---

## 📌 Project Overview

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Model Development & Experiment Tracking | ✅ |
| M2 | Model Packaging & Containerization | ✅ |
| M3 | CI Pipeline — Build, Test & Image Creation | ✅ |
| M4 | CD Pipeline & Deployment | ✅ |
| M5 | Monitoring, Logs & Final Submission | ✅ |

---

## 🏗️ Architecture

```
GitHub Push
    │
    ▼
GitHub Actions CI
    ├── Run Unit Tests (pytest)
    ├── Build Docker Image
    └── Push to Docker Hub
            │
            ▼
    GitHub Actions CD
        ├── Pull Image from Docker Hub
        ├── Deploy via Docker Compose
        └── Run Smoke Tests
                │
                ▼
        Flask Inference Service
            ├── /health
            ├── /predict
            ├── /metrics
            └── /prediction-log
```

---

## 🧠 Model

- **Architecture:** SimpleCNN (4 Conv layers + 3 FC layers)
- **Task:** Binary classification — Cat vs Dog
- **Input:** RGB image resized to 224×224
- **Output:** Sigmoid score (0 = cat, 1 = dog)
- **Test Accuracy:** 70–80%
- **Framework:** PyTorch
- **Model file:** `simple_cnn_baseline_exp1_20260217_053749_best.pt` (299MB)
- **Stored in:** GitHub Releases v1.0

---

## 📁 Project Structure

```
mlops-classifier/
├── .dvc/                          # DVC configuration
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
├── static/css/                    # Frontend styles
├── templates/                     # Flask HTML templates
├── tests/
│   └── test_pipeline.py           # Unit tests (pytest)
├── app.py                         # Flask inference service
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Deployment manifest
├── requirements.txt               # Pinned dependencies
├── smoke_test.sh                  # Post-deploy smoke tests
└── test_model.py                  # Model inference script
```

---

## 🚀 Quick Start

### Run locally with Docker Compose

```bash
docker compose up -d
```

### Or build manually

```bash
docker build -t pet-classifier .
docker run -p 5000:5000 pet-classifier
```

---

## 🌐 API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Prediction
```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:5000/predict
```
```json
{
  "success": true,
  "prediction": "dog",
  "confidence": 92.3,
  "probabilities": {
    "cat": 7.7,
    "dog": 92.3
  },
  "latency_ms": 134.5
}
```

### Metrics (M5)
```bash
curl http://localhost:5000/metrics
```
```json
{
  "total_requests": 42,
  "successful_predictions": 40,
  "failed_requests": 2,
  "average_latency_ms": 145.3
}
```

### Prediction Log (M5)
```bash
curl http://localhost:5000/prediction-log
```
```json
{
  "total_predictions": 40,
  "dog_predictions": 22,
  "cat_predictions": 18,
  "average_confidence": 87.4,
  "recent_predictions": [...]
}
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/test_pipeline.py -v
```

Tests cover:
- Image preprocessing output shape and normalization
- SimpleCNN forward pass output range
- Batch inference
- Model eval mode

---

## ⚙️ CI/CD Pipeline

Defined in `.github/workflows/ci.yml`:

**On every push to `main`:**

**Job 1 — CI (test-and-build):**
1. Checkout repository
2. Install dependencies
3. Run unit tests via pytest
4. Build Docker image
5. Push to Docker Hub

**Job 2 — CD (deploy-and-smoke-test):**
1. Pull latest image from Docker Hub
2. Deploy with Docker Compose
3. Wait for service to be ready (smart retry loop)
4. Run smoke tests (health + prediction)
5. Fail pipeline if smoke tests fail
6. Tear down containers

---

## 📊 Monitoring (M5)

The inference service includes built-in monitoring:

- **Logging:** Every request logged with timestamp, filename, prediction, confidence, and latency to both console and `app.log`
- **Request counter:** Tracks total, successful, and failed requests in memory
- **Latency tracking:** Per-request and average latency in milliseconds
- **Prediction log:** Rolling window of last 100 predictions with cat/dog distribution

---

## 🔧 Environment & Dependencies

All dependencies are pinned in `requirements.txt`. Key libraries:

- `torch`, `torchvision` — Model inference
- `flask`, `gunicorn` — Web service
- `Pillow`, `numpy` — Image processing

---

## 🐳 Docker Hub

Image available at: `vaishnavi06/mlops-classifier:latest`

```bash
docker pull vaishnavi06/mlops-classifier:latest
```

---

## 👩‍💻 Author

**Vaishnavi Gaikwad**  
MLOps Assignment — 2026