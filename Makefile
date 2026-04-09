.PHONY: up down logs train train-all train-fast train-focal train-bce train-weighted-bce train-dice simulate lint test clean

# ── Training defaults (override from CLI) ─────────────────────────
# Example:
#   make train TRAIN_LOSS=weighted_bce POS_WEIGHT=8.0 EPOCHS=5
DATASET ?= ieee_cis
EPOCHS ?= 20
TRAIN_LOSS ?= weighted_bce
FOCAL_ALPHA ?= 0.25
FOCAL_GAMMA ?= 2.0
POS_WEIGHT ?= 8.0
ARTIFACTS_DIR ?=
SAVE_PATH ?=
LOGS_DIR ?=

# ── Infrastructure ────────────────────────────────────────────────
up:
	@echo "Use local dev commands instead: make dev-backend and make dev-frontend"

down:
	@echo "No containerized services configured. Stop local processes manually."

logs:
	@echo "Check logs from your local backend/frontend terminals."

restart-backend:
	@echo "Restart backend by rerunning: make dev-backend"

# ── Database ──────────────────────────────────────────────────────
db-migrate:
	cd backend && uv run alembic upgrade head

db-reset:
	@echo "No automated database reset configured."
	@echo "If using local Postgres, recreate DB manually and run migrations."
	cd backend && uv run alembic upgrade head
	@echo "Database reset complete"

# ── ML Pipeline ───────────────────────────────────────────────────
pipeline:
	cd backend && uv run python -m app.ml.pipeline

train:
	cd backend && uv run python -m app.ml.train \
		--dataset $(DATASET) --epochs $(EPOCHS) \
		--loss $(TRAIN_LOSS) --focal-alpha $(FOCAL_ALPHA) --focal-gamma $(FOCAL_GAMMA) \
		$(if $(strip $(SAVE_PATH)),--save-path $(SAVE_PATH),) \
		$(if $(strip $(ARTIFACTS_DIR)),--artifacts-dir $(ARTIFACTS_DIR),) \
		$(if $(strip $(LOGS_DIR)),--logs-dir $(LOGS_DIR),) \
		$(if $(strip $(POS_WEIGHT)),--pos-weight $(POS_WEIGHT),)

train-focal:
	$(MAKE) train TRAIN_LOSS=focal

train-bce:
	$(MAKE) train TRAIN_LOSS=bce

train-weighted-bce:
	$(MAKE) train TRAIN_LOSS=weighted_bce POS_WEIGHT=8.0

train-dice:
	$(MAKE) train TRAIN_LOSS=dice

train-all:
	cd backend && uv run python -m app.ml.train --dataset ieee_cis --epochs 50 --loss $(TRAIN_LOSS) --focal-alpha $(FOCAL_ALPHA) --focal-gamma $(FOCAL_GAMMA) $(if $(strip $(SAVE_PATH)),--save-path $(SAVE_PATH),) $(if $(strip $(ARTIFACTS_DIR)),--artifacts-dir $(ARTIFACTS_DIR),) $(if $(strip $(LOGS_DIR)),--logs-dir $(LOGS_DIR),) $(if $(strip $(POS_WEIGHT)),--pos-weight $(POS_WEIGHT),)
	cd backend && uv run python -m app.ml.train --dataset paysim   --epochs 30 --loss $(TRAIN_LOSS) --focal-alpha $(FOCAL_ALPHA) --focal-gamma $(FOCAL_GAMMA) $(if $(strip $(SAVE_PATH)),--save-path $(SAVE_PATH),) $(if $(strip $(ARTIFACTS_DIR)),--artifacts-dir $(ARTIFACTS_DIR),) $(if $(strip $(LOGS_DIR)),--logs-dir $(LOGS_DIR),) $(if $(strip $(POS_WEIGHT)),--pos-weight $(POS_WEIGHT),)
	cd backend && uv run python -m app.ml.train --dataset elliptic --epochs 40 --loss $(TRAIN_LOSS) --focal-alpha $(FOCAL_ALPHA) --focal-gamma $(FOCAL_GAMMA) $(if $(strip $(SAVE_PATH)),--save-path $(SAVE_PATH),) $(if $(strip $(ARTIFACTS_DIR)),--artifacts-dir $(ARTIFACTS_DIR),) $(if $(strip $(LOGS_DIR)),--logs-dir $(LOGS_DIR),) $(if $(strip $(POS_WEIGHT)),--pos-weight $(POS_WEIGHT),)

train-fast:
	cd backend && uv run python -m app.ml.train \
		--dataset ieee_cis --epochs 20 --hidden 64 --heads 2 --batch-size 128 --max-rows 200000 \
		--loss $(TRAIN_LOSS) --focal-alpha $(FOCAL_ALPHA) --focal-gamma $(FOCAL_GAMMA) \
		$(if $(strip $(SAVE_PATH)),--save-path $(SAVE_PATH),) \
		$(if $(strip $(ARTIFACTS_DIR)),--artifacts-dir $(ARTIFACTS_DIR),) \
		$(if $(strip $(LOGS_DIR)),--logs-dir $(LOGS_DIR),) \
		$(if $(strip $(POS_WEIGHT)),--pos-weight $(POS_WEIGHT),)

# ── Demo data ─────────────────────────────────────────────────────
simulate:
	uv run --project backend python scripts/simulate.py --count 500 --dataset mixed

simulate-stream:
	uv run --project backend python scripts/simulate.py --stream --interval 0.5

# ── Development ───────────────────────────────────────────────────
dev-backend:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

install-backend:
	cd backend && uv sync

install-frontend:
	cd frontend && npm install

# ── Datasets ──────────────────────────────────────────────────────
dataset-status:
	uv run --project backend python data/download.py --status

dataset-instructions:
	uv run --project backend python data/download.py --instructions

# ── Quality ──────────────────────────────────────────────────────
lint:
	cd backend && uv run ruff check app/
	cd frontend && npx tsc --noEmit

test:
	cd backend && uv run pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	cd frontend && rm -rf dist/ node_modules/.cache/
	@echo "Cleaned"
