# FraudGraph Setup Guide (uv-first)

This guide is aligned with the current repository state and uses `uv` for Python dependency management.

## Documentation Index

- Project overview and quick start: [README.md](README.md)
- Backend-focused quick guide: [backend/README.md](backend/README.md)
- Full setup and troubleshooting: [SETUP.md](SETUP.md)

## Prerequisites

| Tool | Recommended | Notes |
|---|---|---|
| Python | 3.11+ | Backend runtime |
| uv | latest | Dependency and virtualenv management |
| Node.js | 20+ | Frontend local dev |
| npm | 10+ | Frontend packages |

## 1. Clone and Environment

```bash
git clone <repo-url> fraud-detection
cd fraud-detection

cp backend/.env.example backend/.env
```

Important values in `backend/.env`:

- `DATABASE_URL`
- `REDIS_URL`
- `MODEL_PATH`
- `DEVICE`
- `DATA_DIR`, `IEEE_CIS_PATH`, `PAYSIM_PATH`, `ELLIPTIC_PATH`

## 2. Install Dependencies

### Backend (uv)

```bash
cd backend
uv sync
cd ..
```

Notes:

- Dependencies are maintained through `uv` (`backend/pyproject.toml` + `backend/uv.lock`).
- If you add packages, use `uv add ...` in `backend/`.

### Frontend

```bash
cd frontend
npm install
cd ..
```

## 3. Dataset Preparation

Check what is present:

```bash
uv run --project backend python data/download.py --status
```

Print dataset-specific instructions:

```bash
uv run --project backend python data/download.py --instructions ieee_cis paysim elliptic
```

Expected locations:

- `data/ieee_cis/`
- `data/paysim/`
- `data/elliptic/`

## 4. Start Services

Run services locally in separate terminals:

Terminal 1:

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev
```

## 5. Database Migration

If needed for local DB state:

```bash
cd backend
uv run alembic upgrade head
```

Or from root:

```bash
make db-migrate
```

## 6. ML Pipeline and Training

### Run preprocessing pipeline

```bash
cd backend
uv run python -m app.ml.pipeline
```

### Train model

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 50
```

### Train with selectable loss functions

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss focal
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss bce
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss weighted_bce --pos-weight 8.0
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss dice
```

### Artifacts output

- By default, training stores plots/metrics under `artifacts/training/<run_name>/`.
- Override path:

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --artifacts-dir ../artifacts/custom
```

Generated artifacts include:

- Training/validation curves
- Precision-Recall curve
- Threshold vs Precision/Recall/F1
- Score distribution (fraud vs non-fraud)
- ROC curve
- Calibration curve
- Confusion matrix
- Test metric bar chart
- `metrics_summary.json`
- `threshold_scan_metrics.json`

## 7. Makefile Shortcuts

Common targets from project root:

```bash
make up
make down
make logs
make pipeline
make train
make train-focal
make train-bce
make train-weighted-bce
make train-dice
make train-fast
```

Examples with overrides:

```bash
make train DATASET=ieee_cis EPOCHS=5 TRAIN_LOSS=weighted_bce POS_WEIGHT=8.0 ARTIFACTS_DIR=artifacts
```

## 8. API Sanity Checks

Health endpoint:

```bash
curl http://localhost:8000/health
```

Single prediction:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 4999.99,
    "channel": "online",
    "hour_of_day": 3,
    "velocity_1h": 12,
    "velocity_24h": 35,
    "country_mismatch": true,
    "merchant_fraud_rate": 0.18
  }'
```

Useful API routes:

- `POST /api/predict`
- `POST /api/predict/batch`
- `GET /api/transactions`
- `GET /api/model/info`
- `GET /api/stats/overview`
- `GET /api/stats/timeseries`
- `WS /api/ws/scores`

## 9. Troubleshooting

`ModuleNotFoundError` when running backend commands:

- Run via `uv run ...` from `backend/`, or ensure `backend/.venv` is active.

IEEE-CIS missing files:

- Ensure `train_transaction.csv` and `train_identity.csv` are in `data/ieee_cis/`.

Training is slow at startup:

- Initial CSV loading/preprocessing can take time on first run.

No file logs for training:

- Current training logs are printed to console; metrics/plots are saved in `artifacts/...`.
