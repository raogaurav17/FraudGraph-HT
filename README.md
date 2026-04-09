# FraudGraph - HTGNN Fraud Detection Platform

Production-oriented fraud detection platform with:

- FastAPI backend for real-time scoring
- React dashboard for monitoring and exploration
- HTGNN training pipeline with configurable losses
- Automated training artifacts (metrics + plots)

For full setup details, use [SETUP.md](SETUP.md).

## Tech Stack

- Frontend: React, Vite, Tailwind, Recharts, Zustand
- Backend: FastAPI, SQLAlchemy, Alembic, Redis
- ML: PyTorch, PyTorch Geometric, scikit-learn
- Infra: Docker Compose, Nginx
- Python package manager: uv

## Quick Start (uv + Docker)

```bash
# 1) Clone
git clone <repo-url> fraud-detection
cd fraud-detection

# 2) Configure env
cp backend/.env.example backend/.env

# 3) Install deps
cd backend && uv sync && cd ..
cd frontend && npm install && cd ..

# 4) Check dataset status
uv run --project backend python data/download.py --status

# 5) Start services
docker compose up -d

# 6) Run training (with artifacts)
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss weighted_bce --pos-weight 8.0
```

Default app URLs:

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Training Options

Supported loss functions:

- focal
- bce
- weighted_bce
- dice

Examples:

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss focal
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss bce
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss weighted_bce --pos-weight 8.0
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss dice
```

## Training Artifacts

Artifacts are saved per run under:

```text
artifacts/training/<run_name>/
```

Includes:

- training_validation_curves.png
- precision_recall_curve.png
- threshold_vs_precision_recall_f1.png
- score_distribution_fraud_vs_nonfraud.png
- roc_curve.png
- calibration_curve_reliability_diagram.png
- confusion_matrix.png
- test_metrics_bar_chart.png
- metrics_summary.json
- threshold_scan_metrics.json

Override artifacts location:

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --artifacts-dir ../artifacts/custom
```

## Useful Make Targets

```bash
make up
make logs
make train
make train-focal
make train-bce
make train-weighted-bce
make train-dice
make train-fast
```

Parameterized example:

```bash
make train DATASET=ieee_cis EPOCHS=5 TRAIN_LOSS=weighted_bce POS_WEIGHT=8.0 ARTIFACTS_DIR=artifacts
```

## API Endpoints

- GET /health
- POST /api/predict
- POST /api/predict/batch
- GET /api/transactions
- GET /api/model/info
- GET /api/stats/overview
- GET /api/stats/timeseries
- WS /api/ws/scores

## Repository Layout

```text
fraud-detection/
     backend/
     frontend/
     data/
     docker/
     artifacts/
     docker-compose.yml
     Makefile
     SETUP.md
```
