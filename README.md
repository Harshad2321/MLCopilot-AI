---
<<<<<<< HEAD
title: MLCopilot AI
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: >-
  Real-time ML training monitor — detects exploding/vanishing gradients,
  overfitting, loss stagnation & more. Powered by FastAPI + Streamlit.
---

# 🤖 MLCopilot AI

> **AI for Bharat Hackathon (powered by AWS) — AI for Learning & Developer Productivity**

MLCopilot AI is an intelligent debugging assistant for ML engineers.  
It monitors model training **in real time**, detects ML issues automatically, performs **root-cause analysis**, and suggests **actionable fixes** — all through a polished interactive dashboard.

Train your model locally → metrics are streamed to the backend → problems are explained → fixes are shown on the dashboard.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **SDK Logger** | Two-line integration — `start_monitoring()` + `log()` |
| **Anomaly Detector** | Rule-based detection of 5+ common ML problems |
| **Root Cause Engine** | Infers likely causes with human-readable explanations |
| **LLM Engine** | Optional GPT-4o / Claude integration for richer explanations |
| **REST API** | FastAPI backend — POST metrics, GET analysis |
| **Live Dashboard** | Streamlit UI with real-time charts and issue cards |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend (Terminal 1)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the dashboard (Terminal 2)

```bash
streamlit run dashboard/app.py
```

### 4. Run a training example (Terminal 3)

```bash
python training_example/train_model.py --scenario exploding_gradients
# or: overfitting | vanishing_gradients | healthy
```

Open **http://localhost:8501** to watch the dashboard update in real time.

---

## 🔌 Integrate with Your Own Model

```python
from sdk.mlcopilot_logger import MLCopilotLogger

logger = MLCopilotLogger(run_id="my_run", api_url="http://localhost:8000")

for epoch in range(num_epochs):
    # ... your training code ...
    logger.log(
        epoch         = epoch,
        train_loss    = train_loss,
        val_loss      = val_loss,
        accuracy      = accuracy,
        learning_rate = current_lr,
        gradient_norm = grad_norm,
    )

logger.finish()
```

That's it — MLCopilot will detect issues and print alerts to your terminal  
while the dashboard shows full charts and suggestions.

---

## 🌐 Using a Remote / Published Backend

If the backend is deployed online, just change `api_url`:

```python
logger = MLCopilotLogger(
    run_id="my_run",
    api_url="https://your-mlcopilot-server.com",
)
```

Your local training script streams metrics to the online server — the analysis  
and dashboard work exactly the same way.

---

## 🧠 LLM Integration (Optional)

Set one of these environment variables for richer AI explanations:

```bash
# OpenAI (GPT-4o-mini)
export OPENAI_API_KEY=sk-...

# Anthropic (Claude 3 Haiku)
export ANTHROPIC_API_KEY=sk-ant-...
```

If no key is set, the system uses built-in rule-based explanations automatically.

---

## 📁 Project Structure

```
mlcopilot-ai/
├── backend/
│   ├── main.py          ← FastAPI entry point (POST /metrics, GET /analysis)
│   ├── analyzer.py      ← Rule-based ML issue detector
│   ├── database.py      ← SQLite storage layer
│   └── llm_engine.py    ← LLM explanation engine (OpenAI / Anthropic / fallback)
│
├── sdk/
│   └── mlcopilot_logger.py  ← Drop-in training logger (the SDK)
│
├── training_example/
│   └── train_model.py   ← PyTorch demo with 4 intentional bug scenarios
│
├── dashboard/
│   └── app.py           ← Streamlit live dashboard
│
├── database/
│   └── mlcopilot.db     ← Auto-created SQLite database
│
├── requirements.txt
└── README.md
```

---

## 🔍 Detectable Issues

| Issue | Detection Condition |
|-------|---------------------|
| **Exploding Gradients** | gradient_norm > 10 |
| **Vanishing Gradients** | gradient_norm < 1e-7 |
| **Overfitting** | val_loss rising while train_loss falling, gap > 0.15 |
| **Underfitting** | accuracy < 55% after 8+ epochs |
| **Loss Stagnation** | loss improves < 0.005 over 5 consecutive epochs |

---

## 🖥️ API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/metrics` | Submit epoch metrics from training script |
| `GET`  | `/metrics?run_id=<id>` | Retrieve metric history |
| `GET`  | `/analysis?run_id=<id>` | Full issue analysis with LLM explanations |
| `GET`  | `/health` | Server health check |
| `GET`  | `/docs` | Interactive Swagger docs |

---

## 🏆 Hackathon Info

**Event:** AI for Bharat — powered by AWS  
**Track:** AI for Learning & Developer Productivity  
**Problem:** ML engineers waste hours debugging training failures with no guidance.  
**Solution:** MLCopilot AI — detect, explain, and fix ML issues automatically.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **Training Monitor** | Real-time tracking of loss, accuracy, gradients, learning rate |
| **Anomaly Detector** | Rule-based + statistical detection of 8+ common ML problems |
| **Root Cause Engine** | Infers likely causes with confidence scoring |
| **Suggestion Engine** | Generates fixes, parameter changes, code snippets & explanations |
| **Hyperparameter Optimizer** | Optuna-powered search for optimal training configs |
| **Interactive Dashboard** | 6-page Streamlit UI with Plotly charts and live monitoring |
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
MLCopilot AI/
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
│   └── streamlit_app.py            # Interactive web dashboard (6 pages)
├── run_demo.py                     # Application launcher
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Full Application

```bash
python run_demo.py
```

This starts both the FastAPI backend and the Streamlit dashboard automatically.

- **Dashboard** → http://localhost:8501
- **API Docs** → http://localhost:8000/docs

### 3. Or Start Components Separately

```bash
# Terminal 1 — Backend API
uvicorn backend.main_api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Dashboard
streamlit run dashboard/streamlit_app.py
```

> **Note:** The dashboard auto-starts the backend if it's not running, so you can also just run `streamlit run dashboard/streamlit_app.py` directly.

---

## 🖥️ Dashboard Pages

### 📊 Dashboard
Overview with KPIs (total experiments, running, completed), recent experiment cards with mini charts, and feature descriptions.

### 🚀 Run Training
Select from 4 built-in scenarios (Exploding Gradients, Overfitting, Vanishing Gradients, Healthy Baseline). Launches training directly from the UI with **live progress bars**, **real-time metric updates**, and **instant anomaly alerts**.

### 📈 Experiment Monitor
Deep-dive into any experiment with interactive Plotly charts:
- **Loss Curves** (train + validation with area fill)
- **Accuracy Curves**
- **Gradient Norm** (with exploding threshold line)
- **Learning Rate Schedule**
- **4-panel Overview**
- **Raw Metrics Table**

### 🔍 Analysis & Diagnostics
Run a full analysis on any experiment. Shows:
- **Detected issues** with severity badges and evidence
- **Root cause analysis** with confidence percentages
- **Fix suggestions** with code snippets, parameter changes, and explanations
- **Downloadable text report**
- **Analysis history**

### ⚙️ Hyperparameter Optimizer
Configure and run Optuna-powered Bayesian optimization:
- Adjustable trials, epochs, and sample count
- Searches over LR, hidden size, depth, dropout, optimizer, batch size
- Shows **best parameters**, **trial scatter plot**, and **optimization progress curve**

### 🧪 AI Advisor
Instant advice for any ML training problem — select a problem, choose severity, and get:
- Root causes with confidence bars
- Ranked fix recommendations
- Ready-to-use code snippets
- Suggested parameter changes
- Plain-English explanations

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

## 🔬 Training Scenarios

| Scenario | What Goes Wrong | Expected Detection |
|----------|----------------|-------------------|
| `exploding_gradients` | LR=0.5, no clipping | Exploding Gradients, LR Too High |
| `overfitting` | 512-hidden model on 50 samples | Overfitting |
| `vanishing_gradients` | 15-layer sigmoid, tiny init | Vanishing Gradients |
| `healthy` | Well-configured baseline | No issues ✅ |

Run any scenario from the CLI:
```bash
python ml_pipeline/sample_training.py --scenario exploding_gradients --api-url http://localhost:8000
```

Or run them directly from the dashboard's **🚀 Run Training** page.

---

## 📊 Example Analysis Output

```
============================================================
  MLCopilot AI - Training Analysis Report
============================================================

[CRITICAL] Issue #1: Exploding Gradients
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

  Code Suggestion:
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

  Explanation:
    Large gradients are destabilizing the training process.
    Reducing the learning rate lowers the magnitude of parameter
    updates. Gradient clipping caps the norm of gradients to
    prevent explosive growth.
============================================================
```

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
│   Metrics Monitor  │  Hyperparameter Optimizer        │
├──────────────────────────────────────────────────────┤
│                    Data Layer                         │
│         SQLite Storage  │  Training Logs              │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **FastAPI** | Backend REST API |
| **PyTorch** | ML training pipeline |
| **Optuna** | Hyperparameter optimization |
| **Streamlit** | Interactive dashboard |
| **Plotly** | Interactive charts & visualizations |
| **SQLite** | Lightweight persistent storage |
| **Pandas** | Data manipulation |

---

## 📝 License

MIT
=======
title: MLCopilot
emoji: 📚
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 677e18ef13400210ef97fef68e54b773bc61f715
