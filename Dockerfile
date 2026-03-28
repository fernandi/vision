FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and immediately verify it imports correctly
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu && \
    python -c "import torch; print('torch OK:', torch.__version__)"

# Install remaining dependencies and verify torch is still importable
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import torch; print('torch still OK after requirements.txt:', torch.__version__)"

COPY . .

CMD ["/bin/sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
