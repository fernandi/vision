#!/bin/bash
# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run project initialization first."
    exit 1
fi

echo "Starting Visual Search Engine..."
echo "Access at http://localhost:8000"

# Run Uvicorn from venv
# app.backend.main:app -> module app.backend.main, instance app
# Fix for potential OpenMP duplicate library error on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

venv/bin/uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
