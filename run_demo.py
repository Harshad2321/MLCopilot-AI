"""
MLCopilot AI — Application Launcher
====================================
Starts the FastAPI backend server and opens the Streamlit dashboard.

Usage:
    python run_demo.py            # Launches the full app
    python run_demo.py --api-only # Starts only the backend API
"""

import subprocess
import sys
import os
import time
import argparse
import signal
import requests

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
API_URL = "http://localhost:8000"
DASHBOARD_PORT = 8501


def wait_for_server(url: str, timeout: int = 20) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="MLCopilot AI — Launcher")
    parser.add_argument("--api-only", action="store_true",
                        help="Start only the FastAPI backend server")
    parser.add_argument("--port", type=int, default=8501,
                        help="Streamlit dashboard port (default: 8501)")
    args = parser.parse_args()

    procs = []

    def cleanup(signum=None, frame=None):
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 56)
    print("  🤖  MLCopilot AI")
    print("=" * 56)
    print()

    # ── Start backend ──
    print("[1] Starting FastAPI backend server …")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main_api:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT_DIR,
    )
    procs.append(backend)

    if not wait_for_server(API_URL):
        print("✗  Backend failed to start. Check port 8000.")
        cleanup()
        return

    print(f"✓  Backend running → {API_URL}")
    print(f"   API docs       → {API_URL}/docs")
    print()

    if args.api_only:
        print("Running in API-only mode. Press Ctrl+C to stop.")
        try:
            backend.wait()
        except KeyboardInterrupt:
            cleanup()
        return

    # ── Start dashboard ──
    print("[2] Starting Streamlit dashboard …")
    dashboard = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join(ROOT_DIR, "dashboard", "streamlit_app.py"),
         "--server.port", str(args.port),
         "--server.headless", "false",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT_DIR,
    )
    procs.append(dashboard)
    time.sleep(3)

    print(f"✓  Dashboard running → http://localhost:{args.port}")
    print()
    print("=" * 56)
    print("  MLCopilot AI is ready!")
    print()
    print(f"  Dashboard : http://localhost:{args.port}")
    print(f"  API       : {API_URL}")
    print(f"  API Docs  : {API_URL}/docs")
    print()
    print("  Press Ctrl+C to stop all services.")
    print("=" * 56)

    try:
        dashboard.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
