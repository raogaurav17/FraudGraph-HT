.PHONY: up down logs train simulate lint test clean

# ── Infrastructure ────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "Services started. Dashboard: http://localhost:5173  API: http://localhost:8000"

down:
	docker compose down

logs:
	docker compose logs -f backend frontend

restart-backend:
	docker compose restart backend

# ── Database ──────────────────────────────────────────────────────
db-migrate:
	cd backend && alembic upgrade head

db-reset:
	docker compose down -v
	docker compose up postgres -d
	sleep 3
	cd backend && alembic upgrade head
	@echo "Database reset complete"

# ── ML Pipeline ───────────────────────────────────────────────────
pipeline:
	cd backend && python -m app.ml.pipeline

train:
	cd backend && python -m app.ml.train --dataset ieee_cis --epochs 50

train-all:
	cd backend && python -m app.ml.train --dataset ieee_cis --epochs 50
	cd backend && python -m app.ml.train --dataset paysim   --epochs 30
	cd backend && python -m app.ml.train --dataset elliptic --epochs 40

train-fast:
	cd backend && python -m app.ml.train \
		--dataset ieee_cis --epochs 20 --hidden 64 --heads 2 --batch-size 128

# ── Demo data ─────────────────────────────────────────────────────
simulate:
	python scripts/simulate.py --count 500 --dataset mixed

simulate-stream:
	python scripts/simulate.py --stream --interval 0.5

# ── Development ───────────────────────────────────────────────────
dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

install-backend:
	pip install -r backend/requirements.txt
	pip install torch-scatter torch-sparse \
		-f https://data.pyg.org/whl/torch-2.3.0+cpu.html

install-frontend:
	cd frontend && npm install

# ── Datasets ──────────────────────────────────────────────────────
dataset-status:
	python data/download.py --status

dataset-instructions:
	python data/download.py --instructions

# ── Quality ──────────────────────────────────────────────────────
lint:
	cd backend && ruff check app/
	cd frontend && npx tsc --noEmit

test:
	cd backend && pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	cd frontend && rm -rf dist/ node_modules/.cache/
	@echo "Cleaned"
