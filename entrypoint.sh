#!/bin/sh
set -e  # Exit on any error

# ✅ Start periodic FAISS indexing in background
while true; do
    echo "🚀 Starting FAISS indexing..."
    python -u build_faiss_index.py
    echo "✅ FAISS indexing complete. Sleeping for 60 seconds..."
    sleep 30
done &

# ✅ Start the API normally (in foreground)
echo "🚀 Starting API..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
