FROM python:3.11-slim

WORKDIR /app

# System deps for PyTorch Geometric
RUN apt-get update && apt-get install -y \
    gcc g++ git curl ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r requirements.txt

# PyG extras (CPU build for portability; swap for cu118 wheel URL for GPU)
RUN uv pip install --system --no-cache \
    --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    torch-scatter==2.1.2 torch-sparse==0.6.18

# DGL Linux wheels are published on the DGL wheel index, not PyPI.
RUN uv pip install --system --no-cache \
    --find-links https://data.dgl.ai/wheels/torch-2.3/repo.html \
    dgl==2.2.1

COPY . .

# Create model directory
RUN mkdir -p /app/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
