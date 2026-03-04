"""
MLCopilot AI - Demo Runner
Starts the server, runs all training scenarios, and displays results.

Usage:  python run_demo.py
"""

import subprocess
import sys
import os
import time
import requests

ROOT_DIR = os.path.dirname(__file__)
os.chdir(ROOT_DIR)

# Ensure project root is on path
sys.path.insert(0, ROOT_DIR)

API_URL = "http://localhost:8000"


def wait_for_server(url, timeout=15):
    """Wait until the server is ready."""
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
    print("=" * 60)
    print("  🤖 MLCopilot AI — Full Demo")
    print("=" * 60)
    print()

    # ---- Step 1: Start server ----
    print("[1/4] Starting FastAPI server...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main_api:app",
         "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT_DIR,
    )
    time.sleep(1)

    if not wait_for_server(API_URL):
        print("❌ Server failed to start. Check if port 8000 is available.")
        server_proc.terminate()
        sys.exit(1)

    print("✅ Server is running at http://localhost:8000\n")

    # ---- Step 2: Run training scenarios ----
    scenarios = ["exploding_gradients", "overfitting", "healthy"]

    for i, scenario in enumerate(scenarios, 2):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(scenarios)+1}] Running scenario: {scenario}")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [sys.executable, "ml_pipeline/sample_training.py",
             "--scenario", scenario,
             "--api-url", API_URL],
            cwd=ROOT_DIR,
            timeout=120,
        )
        time.sleep(0.5)

    # ---- Step 3: Show summary ----
    print(f"\n{'='*60}")
    print("  📊 Experiments Summary")
    print(f"{'='*60}\n")

    try:
        experiments = requests.get(f"{API_URL}/api/experiments", timeout=5).json()
        for exp in experiments:
            print(f"  #{exp['id']} | {exp['name']:30s} | {exp['status']}")
            analyses = requests.get(f"{API_URL}/api/analysis/{exp['id']}", timeout=5).json()
            if analyses:
                total_issues = sum(len(a.get("issues", [])) for a in analyses)
                print(f"       → {total_issues} issue(s) detected across {len(analyses)} analysis run(s)")
            else:
                print(f"       → No issues detected ✅")
    except Exception as e:
        print(f"  Error fetching summary: {e}")

    print(f"\n{'='*60}")
    print("  ✅ Demo complete!")
    print()
    print("  Next steps:")
    print("    • Open http://localhost:8000/docs for API docs")
    print("    • Run: streamlit run dashboard/streamlit_app.py")
    print("    • Run: python optimizer/hyperparameter_optimizer.py")
    print(f"{'='*60}\n")

    print("Press Ctrl+C to stop the server...")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print("Done.")


if __name__ == "__main__":
    main()
