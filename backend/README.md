## Backend Quick Guide (uv)

This backend uses uv for dependency and environment management.

### Setup

```bash
cd backend
uv sync
cp .env.example .env
```

### Run API locally

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

### Train model

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss focal
```

Loss options:

- focal
- bce
- weighted_bce
- dice

Example with weighted BCE:

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --loss weighted_bce --pos-weight 8.0
```

### Artifacts

Training produces plots and metrics JSON under:

```text
artifacts/training/<run_name>/
```

Override output directory:

```bash
cd backend
uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --artifacts-dir ../artifacts/custom
```
