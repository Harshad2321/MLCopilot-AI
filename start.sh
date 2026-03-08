#!/bin/bash
# MLCopilot AI — Container Startup Script
# Starts the FastAPI backend on port 8000, then launches the Streamlit
# dashboard on port 7860 (the port exposed to Hugging Face Spaces).

set -e

cd /home/user/app

echo "==> Initialising database …"
python -c "from database.storage import init_db; init_db()"

echo "==> Starting FastAPI backend on port 8000 …"
uvicorn backend.main_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 &

BACKEND_PID=$!

# Wait until the backend is accepting connections (up to 30 s)
echo "==> Waiting for backend to be ready …"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "    Backend is ready (attempt $i)."
        break
    fi
    sleep 1
done

echo "==> Starting Streamlit dashboard on port 7860 …"
exec streamlit run dashboard/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
