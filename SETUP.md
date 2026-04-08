# FraudGraph — Setup Guide

Complete step-by-step setup for the HTGNN Credit Card Fraud Detection platform.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker + Docker Compose | v24+ | All services |
| Python | 3.11+ | ML pipeline, scripts |
| Node.js | 20+ | Frontend dev |
| 8 GB RAM | — | Model training |
| GPU (optional) | CUDA 11.8+ | Faster training |

---

## 1. Clone and configure

```bash
git clone <repo-url> fraud-detection
cd fraud-detection

# Copy environment template
cp backend/.env.example backend/.env
# Edit backend/.env — set POSTGRES password, Redis URL, etc.
```

**`backend/.env` template:**
```env
DATABASE_URL=postgresql+asyncpg://fraud:fraud_secret@localhost:5432/fraudgraph
REDIS_URL=redis://localhost:6379/0
MODEL_PATH=models/htgnn_latest.pt
DEVICE=cpu            # or cuda
ENV=development
```

---

## 2. Start infrastructure

```bash
# Start PostgreSQL + Redis only first
docker compose up postgres redis -d

# Wait for postgres to be ready
docker compose logs postgres | grep "ready to accept connections"
```

---

## 3. Dataset download

```bash
# Check what's needed
python data/download.py --status

# Show download instructions per dataset
python data/download.py --instructions ieee_cis paysim elliptic
```

### IEEE-CIS (primary, required)
1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
2. Accept competition rules
3. Download `train_transaction.csv` + `train_identity.csv`
4. Place in `data/ieee_cis/`

### PaySim (synthetic, large scale)
1. Go to https://www.kaggle.com/datasets/ealaxi/paysim1
2. Download the CSV file
3. Place in `data/paysim/`

### Elliptic Bitcoin (temporal GNN benchmark)
1. Go to https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Download all three CSVs
3. Place in `data/elliptic/`

### YelpChi + FraudAmazon (auto-downloaded via DGL)
```bash
# These download automatically when the pipeline runs
pip install dgl
python -c "from dgl.data import FraudDataset; FraudDataset('yelp'); FraudDataset('amazon')"
```

---

## 4. Run the ML pipeline

```bash
# Install Python deps
pip install -r backend/requirements.txt

# Install PyG extras (CPU)
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# Run full data pipeline (IEEE-CIS + PaySim + Elliptic + YelpChi)
cd backend
python -m app.ml.pipeline

# Or train directly (auto-runs pipeline if model missing)
python -m app.ml.train --dataset ieee_cis --epochs 50

# For GPU training
python -m app.ml.train --dataset ieee_cis --epochs 100 --hidden 256 --heads 8
```

Training output example:
```
Epoch 001/050 | loss=0.4821 | val_auprc=0.4231 | val_auroc=0.7812 | time=28.3s
Epoch 010/050 | loss=0.2341 | val_auprc=0.7123 | val_auroc=0.8934 | time=24.1s
Epoch 030/050 | loss=0.1124 | val_auprc=0.8756 | val_auroc=0.9312 | time=22.8s
...
Test metrics: {'auprc': 0.891, 'auroc': 0.938, 'f1': 0.742}
Model saved to models/htgnn_latest.pt
```

---

## 5. Start all services

```bash
# Start everything
docker compose up -d

# Check all services are healthy
docker compose ps

# View logs
docker compose logs -f backend
docker compose logs -f frontend
```

Services:
- **Frontend**: http://localhost:5173 (dev) or http://localhost (via nginx)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

---

## 6. Populate demo data

```bash
# Install simulation deps
pip install httpx

# Run 500 simulated transactions (fast batch)
python scripts/simulate.py --count 500 --dataset mixed

# Stream live transactions to demo the real-time dashboard
python scripts/simulate.py --stream --interval 0.5

# Only IEEE-CIS-style transactions
python scripts/simulate.py --count 200 --dataset ieee_cis
```

---

## 7. Verify everything works

```bash
# Health check
curl http://localhost:8000/health

# Score a test transaction
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

# Expected response
{
  "transaction_id": "...",
  "fraud_probability": 0.8742,
  "risk_level": "CRITICAL",
  "model_version": "...",
  "latency_ms": 12.3,
  "explanation": {...}
}
```

---

## Architecture details

### Backend (`backend/app/`)

```
app/
├── main.py           # FastAPI app + lifespan
├── api/
│   └── routes.py     # All API endpoints + WebSocket
├── core/
│   ├── config.py     # Pydantic settings
│   ├── database.py   # Async SQLAlchemy + session
│   └── redis.py      # Redis cache + pubsub
├── models/
│   └── db.py         # SQLAlchemy ORM models
└── ml/
    ├── model.py       # HTGNN architecture (PyTorch Geometric)
    ├── pipeline.py    # Multi-dataset loaders (IEEE-CIS, PaySim, Elliptic, YelpChi)
    ├── train.py       # Training loop with focal loss + early stopping
    └── inference.py   # Real-time scoring service
```

### Frontend (`frontend/src/`)

```
src/
├── App.tsx            # Router + WebSocket connection + nav
├── pages/
│   ├── Dashboard.tsx  # Overview stats, charts, live feed
│   ├── Predict.tsx    # Transaction scoring UI with explanations
│   ├── Explorer.tsx   # Paginated transaction browser with filters
│   └── Monitor.tsx    # Model architecture, training curves, per-dataset metrics
├── components/
│   └── ui.tsx         # RiskBadge, ProbBar, StatTile, Charts, etc.
├── store/index.ts     # Zustand global state (live scores, WS status)
└── utils/api.ts       # Axios client + TypeScript types + WebSocket factory
```

### ML pipeline

```
IEEE-CIS CSV  →  IEEECISPipeline.engineer_features()
                     ↓
              build_hetero_data()   →  HeteroData(
                                         txn, card, merchant, device
                                         card→txn, txn→merchant, txn→device,
                                         card→card (shared device)
                                       )
                     ↓
PaySim CSV    →  PaySimPipeline     →  HeteroData(account, txn)
                     ↓
Elliptic CSVs →  EllipticPipeline   →  HeteroData(txn)  [homogeneous]
                     ↓
YelpChi/Amazon → load_dgl_fraud_dataset() → HeteroData(review/user, 3 relations)
                     ↓
              HTGNN.forward()
                     ↓
              focal_loss()  →  AdamW  →  checkpoint
```

---

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Score single transaction |
| `/api/predict/batch` | POST | Score up to 100 transactions |
| `/api/transactions` | GET | List transactions (paginated) |
| `/api/transactions/{id}` | GET | Get single transaction + prediction |
| `/api/stats/overview` | GET | Dashboard summary stats |
| `/api/stats/timeseries` | GET | Hourly transaction/fraud counts |
| `/api/model/info` | GET | Current model version + metrics |
| `/api/ws/scores` | WS | Real-time fraud score stream |
| `/docs` | GET | Interactive Swagger UI |

---

## Training tips

### On 6GB RAM (your Lenovo IdeaPad)

```bash
# Use smaller batch size, fewer neighbours
python -m app.ml.train \
  --dataset ieee_cis \
  --epochs 30 \
  --hidden 64 \
  --heads 2 \
  --batch-size 256
```

This keeps peak RAM under 4GB. With your 2GB zram + 4GB swapfile you can push to hidden=128.

### GPU training (if available)

```bash
# Update backend/.env
DEVICE=cuda

# Use GPU PyG wheels
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

python -m app.ml.train \
  --dataset ieee_cis \
  --epochs 100 \
  --hidden 256 \
  --heads 8 \
  --batch-size 1024
```

### Multi-dataset training (advanced)

```bash
# Train on IEEE-CIS first, then fine-tune on YelpChi
python -m app.ml.train --dataset ieee_cis --epochs 50
python -m app.ml.train --dataset yelp    --epochs 20 \
  --pretrained models/htgnn_latest.pt
```

---

## Common issues

**`FileNotFoundError: IEEE-CIS data not found`**
→ Place `train_transaction.csv` in `data/ieee_cis/`. See step 3.

**`OOM during training`**
→ Reduce `--batch-size` to 128 and `--hidden` to 64.

**`torch_scatter not found`**
→ Install with exact PyTorch version: `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html`

**Frontend shows empty dashboard**
→ Run `python scripts/simulate.py --count 200` to seed demo data.

**WebSocket not connecting**
→ Ensure Redis is running: `docker compose up redis -d`

**Model shows "demo_v0"**
→ No trained model found. Either run the training pipeline or use demo mode (scores are random but UI is fully functional).
