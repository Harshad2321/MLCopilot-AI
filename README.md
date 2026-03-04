# 🤖 MLCopilot AI

> **An AI assistant that watches ML models train and suggests fixes when problems occur.**

MLCopilot AI is a prototype intelligent debugging assistant for ML engineers. It monitors model training in real time, detects anomalies, performs root-cause analysis, and provides actionable fix suggestions — all automatically.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **Training Monitor** | Real-time tracking of loss, accuracy, gradients, learning rate |
| **Anomaly Detector** | Rule-based + statistical detection of 8+ common ML problems |
| **Root Cause Engine** | Infers likely causes with confidence scoring |
| **Suggestion Engine** | Generates fixes, parameter changes, code snippets & explanations |
| **Hyperparameter Optimizer** | Optuna-powered search for optimal training configs |
| **Streamlit Dashboard** | Interactive web UI for monitoring and analysis |
| **REST API** | FastAPI backend for programmatic access |

### Detectable Problems
- 🔴 Exploding Gradients
- 🔴 Vanishing Gradients
- 🟠 Overfitting
- 🟠 Underfitting
- 🟡 Loss Stagnation
- 🟠 Learning Rate Too High
- 🔴 Loss Divergence
- 🔴 NaN/Inf Values

---

## 📁 Project Structure

```
mlcopilot/
├── backend/
│   ├── main_api.py                 # FastAPI entry point
│   ├── routes/
│   │   ├── analyze_training.py     # Training analysis endpoints
│   │   └── suggestions.py          # Suggestion endpoints
│   └── services/
│       ├── metrics_monitor.py      # Real-time metrics tracking
│       ├── anomaly_detector.py     # Issue detection engine
│       ├── root_cause_engine.py    # Root cause analysis
│       └── suggestion_engine.py    # Fix suggestions & explanations
├── ml_pipeline/
│   ├── sample_training.py          # Sample PyTorch training (with bugs!)
│   └── metrics_logger.py           # Training metrics logger client
├── optimizer/
│   └── hyperparameter_optimizer.py # Optuna hyperparameter search
├── database/
│   └── storage.py                  # SQLite storage layer
├── dashboard/
│   └── streamlit_app.py            # Interactive web dashboard
├── run_demo.py                     # One-command demo runner
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd mlcopilot
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
uvicorn backend.main_api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3. Run a Training Scenario

Open a **new terminal** and run one of the sample training scenarios:

```bash
# Exploding gradients (high learning rate)
python ml_pipeline/sample_training.py --scenario exploding_gradients

# Overfitting (large model + tiny dataset)
python ml_pipeline/sample_training.py --scenario overfitting

# Vanishing gradients (deep sigmoid network)
python ml_pipeline/sample_training.py --scenario vanishing_gradients

# Healthy training (baseline)
python ml_pipeline/sample_training.py --scenario healthy
```

MLCopilot will **automatically detect issues** and print analysis reports in real time.

### 4. Open the Dashboard (Optional)

```bash
streamlit run dashboard/streamlit_app.py
```

### 5. Run the Full Demo

```bash
python run_demo.py
```

This starts the server, runs all training scenarios, and displays results.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/experiment` | Create a new experiment |
| `GET`  | `/api/experiments` | List all experiments |
| `POST` | `/api/metrics` | Submit training metrics (returns real-time analysis) |
| `POST` | `/api/analyze` | Run full analysis on an experiment |
| `GET`  | `/api/metrics/{id}` | Get metrics history |
| `GET`  | `/api/analysis/{id}` | Get analysis history |
| `POST` | `/api/suggest` | Get suggestions for a specific problem |
| `GET`  | `/api/problems` | List all diagnosable problems |

---

## 📊 Example Output

```
============================================================
  MLCopilot AI - Training Analysis Report
============================================================

🟠 Issue #1: Exploding Gradients
----------------------------------------

  Root Cause Analysis:
    • Learning rate too high (confidence: 88%)
    • Missing gradient clipping (confidence: 80%)

  Suggested Fixes:
    → Reduce learning rate (try 0.0001 or 0.0003)
    → Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    → Add Batch Normalization layers

  Recommended Parameter Changes:
    learning_rate: 0.0003
    max_grad_norm: 1.0

  Explanation:
    Large gradients are destabilizing the training process.
    Reducing the learning rate lowers the magnitude of parameter
    updates. Gradient clipping caps the norm of gradients to
    prevent explosive growth.

============================================================
```

---

## 🧪 Hyperparameter Optimization

```bash
python optimizer/hyperparameter_optimizer.py
```

Uses Optuna TPE sampler with median pruning to search over:
- Learning rate
- Hidden dimensions
- Network depth
- Dropout rate
- Optimizer type (Adam/AdamW/SGD)
- Weight decay
- Batch size

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Interface Layer                     │
│   CLI  │  Streamlit Dashboard  │  REST API (FastAPI) │
├──────────────────────────────────────────────────────┤
│                  AI Intelligence                      │
│  Anomaly Detector  │  Root Cause  │  Suggestion      │
│                    │  Engine      │  Engine           │
├──────────────────────────────────────────────────────┤
│                  Processing Layer                     │
│         FastAPI Server  │  Metrics Monitor            │
├──────────────────────────────────────────────────────┤
│                    Data Layer                         │
│         SQLite Storage  │  Training Logs              │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **FastAPI** — Backend API server
- **PyTorch** — Sample ML training pipeline
- **Optuna** — Hyperparameter optimization
- **Streamlit** — Interactive dashboard
- **SQLite** — Lightweight data storage

---

## 📝 License

MIT — Built for hackathon demonstration purposes.
