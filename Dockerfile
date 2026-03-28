FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install CPU-only PyTorch FIRST with --index-url (not --extra-index-url)
# This prevents pip from later resolving the CUDA version from PyPI when
# installing transformers, which would override this installation.
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining dependencies
# torch is already present → pip won't reinstall it
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/bin/sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
