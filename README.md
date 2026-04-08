# FraudGraph — HTGNN Credit Card Fraud Detection Platform

A production-grade full-stack fraud detection system built on Heterogeneous Temporal Graph Neural Networks (HTGNN). Ingests multiple real-world datasets, trains a PyG-based GNN model, and exposes a real-time scoring API with a React dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│   Dashboard · Transaction Explorer · Model Monitor      │
└────────────────────┬────────────────────────────────────┘
                     │  REST + WebSocket
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                          │
│   /api/predict  /api/transactions  /api/model/metrics   │
└──────┬──────────────┬──────────────────┬────────────────┘
       │              │                  │
  PostgreSQL      Redis Cache       ML Service
  (transactions)  (recent scores)   (HTGNN model)
       │
  DuckDB OLAP
  (analytics queries)
```

## Datasets Used

| Dataset | Source | Size | Fraud Rate | Role |
|---------|--------|------|------------|------|
| IEEE-CIS | Kaggle / Vesta | 590K txns | 3.5% | Primary training |
| YelpChi | DGL (CIKM 2020) | 45K nodes | 14.5% | GNN benchmark eval |
| FraudAmazon | DGL (CIKM 2020) | 11K nodes | 9.5% | GNN benchmark eval |
| PaySim | Kaggle | 6.3M txns | 0.13% | Scale + streaming sim |
| Elliptic | Kaggle / MIT | 203K nodes | 16.4% | Temporal GNN eval |

## Stack

- **Frontend**: React 18, Vite, TailwindCSS, Recharts, React Query, Zustand
- **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic v2, Redis
- **ML**: PyTorch, PyTorch Geometric, scikit-learn, DGL
- **Database**: PostgreSQL (operational), DuckDB (analytics)
- **Infra**: Docker Compose, Nginx

---

## Quick Start

```bash
# 1. Clone and enter
git clone <repo> && cd fraud-detection

# 2. Download datasets (see data/README.md)
python data/download.py

# 3. Start all services
docker compose up -d

# 4. Run data pipeline
docker compose exec backend python -m app.ml.pipeline

# 5. Train model
docker compose exec backend python -m app.ml.train

# 6. Open dashboard
open http://localhost:5173
```

---

## Project Structure

```
fraud-detection/
├── frontend/               # React app
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Dashboard, Explorer, Monitor
│   │   ├── hooks/          # useTransactions, useModel, useWebSocket
│   │   ├── store/          # Zustand state management
│   │   └── utils/          # API client, formatters
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routers
│   │   ├── core/           # Config, DB, Redis
│   │   ├── models/         # SQLAlchemy models
│   │   ├── services/       # Business logic
│   │   └── ml/             # HTGNN model, pipeline, training
│   ├── requirements.txt
│   └── Dockerfile
├── data/                   # Dataset download + preprocessing scripts
├── docker/                 # Nginx config
└── docker-compose.yml
```
