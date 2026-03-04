"""
MLCopilot AI — Interactive Dashboard
=====================================
Professional web interface for ML training monitoring, anomaly detection,
root-cause analysis, and automated fix suggestions.

Run:
    streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import subprocess
import sys
import os
import math

# ============================================================================
# PATH / CONFIG
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

API_URL = "http://localhost:8000"

# ============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="MLCopilot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DESIGN CONSTANTS
# ============================================================================
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "train": "#667eea",
    "val": "#ed8936",
    "gradient": "#e53e3e",
    "accuracy": "#48bb78",
    "lr": "#9f7aea",
    "success": "#48bb78",
    "warning": "#ecc94b",
    "danger": "#e53e3e",
    "info": "#4299e1",
}

SEVERITY_CFG = {
    "critical": {"icon": "🔴", "color": "#e53e3e", "label": "CRITICAL"},
    "high":     {"icon": "🟠", "color": "#ed8936", "label": "HIGH"},
    "medium":   {"icon": "🟡", "color": "#ecc94b", "label": "MEDIUM"},
    "low":      {"icon": "🟢", "color": "#48bb78", "label": "LOW"},
}

SCENARIOS = {
    "exploding_gradients": {
        "title": "💥 Exploding Gradients",
        "desc": "High learning rate (0.5) with no gradient clipping causes gradient norms to explode.",
        "lr": 0.5, "epochs": 30, "samples": 500, "batch": 32,
        "expect": ["Exploding Gradients", "Learning Rate Too High"],
    },
    "overfitting": {
        "title": "📉 Overfitting",
        "desc": "Large model (5 layers, 512 hidden) trained on only 50 samples memorizes data.",
        "lr": 0.001, "epochs": 60, "samples": 50, "batch": 16,
        "expect": ["Overfitting"],
    },
    "vanishing_gradients": {
        "title": "🔻 Vanishing Gradients",
        "desc": "15-layer deep sigmoid network with tiny weight initialization.",
        "lr": 0.01, "epochs": 30, "samples": 500, "batch": 32,
        "expect": ["Vanishing Gradients"],
    },
    "healthy": {
        "title": "✅ Healthy Baseline",
        "desc": "Well-configured model (2 layers, ReLU, LR 0.001) for comparison.",
        "lr": 0.001, "epochs": 30, "samples": 1000, "batch": 64,
        "expect": [],
    },
}

CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=13),
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .gradient-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.25;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #999;
        margin-bottom: 1.5rem;
    }
    .sev-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #fff;
    }
    .sev-critical { background: #e53e3e; }
    .sev-high     { background: #ed8936; }
    .sev-medium   { background: #ecc94b; color: #333; }
    .sev-low      { background: #48bb78; }
    footer { visibility: hidden; }
    [data-testid="stMetricLabel"] p { font-size: 0.82rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE DEFAULTS
# ============================================================================
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = "exploding_gradients"
if "last_exp_id" not in st.session_state:
    st.session_state.last_exp_id = None

# ============================================================================
# BACKEND MANAGEMENT
# ============================================================================

def _backend_alive() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _start_backend() -> bool:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main_api:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state["_backend_proc"] = proc
    for _ in range(30):
        if _backend_alive():
            return True
        time.sleep(0.5)
    return False


def ensure_backend():
    if _backend_alive():
        return
    with st.spinner("Starting backend server …"):
        ok = _start_backend()
    if ok:
        st.toast("Backend is ready", icon="✅")
    else:
        st.error(
            "Could not start the backend on port 8000.  \n"
            "Start it manually: `uvicorn backend.main_api:app --port 8000`"
        )
        st.stop()


# ============================================================================
# API HELPERS
# ============================================================================

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(endpoint: str, data: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ============================================================================
# CHART BUILDERS
# ============================================================================

def _safe(series):
    """Replace inf/nan with None so Plotly skips them cleanly."""
    def _clean(v):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return series.apply(_clean)


def chart_loss(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "loss" in df.columns and df["loss"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=_safe(df["loss"]), name="Train Loss",
            line=dict(color=COLORS["train"], width=2.5),
            fill="tozeroy", fillcolor="rgba(102,126,234,0.07)"))
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=_safe(df["val_loss"]), name="Val Loss",
            line=dict(color=COLORS["val"], width=2.5, dash="dash")))
    fig.update_layout(**CHART_LAYOUT, title="Loss Curves",
                      xaxis_title="Epoch", yaxis_title="Loss")
    return fig


def chart_accuracy(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=_safe(df["accuracy"]), name="Train Accuracy",
            line=dict(color=COLORS["success"], width=2.5),
            fill="tozeroy", fillcolor="rgba(72,187,120,0.07)"))
    if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=_safe(df["val_accuracy"]), name="Val Accuracy",
            line=dict(color=COLORS["info"], width=2.5, dash="dash")))
    fig.update_layout(**CHART_LAYOUT, title="Accuracy Curves",
                      xaxis_title="Epoch", yaxis_title="Accuracy")
    return fig


def chart_gradient(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "grad_norm" in df.columns and df["grad_norm"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=_safe(df["grad_norm"]), name="Gradient Norm",
            line=dict(color=COLORS["danger"], width=2.5),
            fill="tozeroy", fillcolor="rgba(229,62,62,0.07)"))
        fig.add_hline(y=10.0, line_dash="dot", line_color="rgba(229,62,62,0.5)",
                      annotation_text="Exploding threshold (10.0)")
    fig.update_layout(**CHART_LAYOUT, title="Gradient Norm",
                      xaxis_title="Epoch", yaxis_title="Gradient Norm")
    return fig


def chart_lr(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "lr" in df.columns and df["lr"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=_safe(df["lr"]), name="Learning Rate",
            line=dict(color=COLORS["lr"], width=2.5),
            mode="lines+markers", marker=dict(size=4)))
    fig.update_layout(**CHART_LAYOUT, title="Learning Rate Schedule",
                      xaxis_title="Epoch", yaxis_title="Learning Rate")
    return fig


def chart_overview(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "Accuracy", "Gradient Norm", "Learning Rate"),
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )
    if "loss" in df.columns and df["loss"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=_safe(df["loss"]), name="Train Loss",
                                 line=dict(color=COLORS["train"], width=2)), row=1, col=1)
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=_safe(df["val_loss"]), name="Val Loss",
                                 line=dict(color=COLORS["val"], width=2, dash="dash")), row=1, col=1)
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=_safe(df["accuracy"]), name="Train Acc",
                                 line=dict(color=COLORS["success"], width=2)), row=1, col=2)
    if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=_safe(df["val_accuracy"]), name="Val Acc",
                                 line=dict(color=COLORS["info"], width=2, dash="dash")), row=1, col=2)
    if "grad_norm" in df.columns and df["grad_norm"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=_safe(df["grad_norm"]), name="Grad Norm",
                                 line=dict(color=COLORS["danger"], width=2)), row=2, col=1)
    if "lr" in df.columns and df["lr"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=_safe(df["lr"]), name="LR",
                                 line=dict(color=COLORS["lr"], width=2)), row=2, col=2)
    fig.update_layout(
        height=600, showlegend=False, template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=30))
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🤖 MLCopilot AI")
    st.caption("Intelligent ML Training Assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🚀 Run Training",
            "📈 Experiment Monitor",
            "🔍 Analysis & Diagnostics",
            "⚙️ Hyperparameter Optimizer",
            "🧪 AI Advisor",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    if _backend_alive():
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
        if st.button("Start Backend"):
            ensure_backend()
            st.rerun()
    st.divider()
    st.caption("v1.0.0 · Built with Streamlit + FastAPI + PyTorch")

# ── Ensure backend is running ──
ensure_backend()


# ============================================================================
# PAGE — DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown('<p class="gradient-header">MLCopilot AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Real-time ML training monitoring · Anomaly detection '
        '· Root-cause analysis · Automated fix suggestions</p>',
        unsafe_allow_html=True,
    )

    experiments = api_get("/api/experiments") or []
    total = len(experiments)
    running = sum(1 for e in experiments if e.get("status") == "running")
    completed = sum(1 for e in experiments if e.get("status") == "completed")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📋 Experiments", total)
    k2.metric("🔄 Running", running)
    k3.metric("✅ Completed", completed)
    k4.metric("🤖 Detectable Issues", 8)

    st.divider()

    if experiments:
        st.subheader("📋 Recent Experiments")
        for exp in experiments[:8]:
            status = exp.get("status", "unknown")
            icon = {"running": "🔵", "completed": "🟢"}.get(status, "⚪")
            label = (
                f"{icon} #{exp['id']} — {exp['name']}  ·  "
                f"{status.upper()}  ·  {exp.get('created_at', '')}"
            )
            with st.expander(label):
                metrics = api_get(f"/api/metrics/{exp['id']}")
                if metrics:
                    df = pd.DataFrame(metrics)
                    if not df.empty:
                        latest = df.iloc[-1]
                        mc = st.columns(5)
                        mc[0].metric("Epochs", len(df))
                        mc[1].metric("Loss",
                                     f"{latest['loss']:.4f}" if latest.get("loss") is not None else "—")
                        mc[2].metric("Val Loss",
                                     f"{latest['val_loss']:.4f}" if latest.get("val_loss") is not None else "—")
                        mc[3].metric("Accuracy",
                                     f"{latest['accuracy']:.2%}" if latest.get("accuracy") is not None else "—")
                        mc[4].metric("Grad Norm",
                                     f"{latest['grad_norm']:.4f}" if latest.get("grad_norm") is not None else "—")
                        fig = chart_loss(df)
                        fig.update_layout(height=260, title="")
                        st.plotly_chart(fig, use_container_width=True, key=f"dash_{exp['id']}")
                else:
                    st.info("No metrics recorded yet.")
    else:
        st.info("No experiments yet. Head to **🚀 Run Training** to start!")

    st.divider()

    st.subheader("🛠️ Capabilities")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("#### 🔍 Anomaly Detection")
        st.markdown(
            "Detects **8 common problems** in real time:\n"
            "- Exploding / Vanishing gradients\n"
            "- Overfitting / Underfitting\n"
            "- Loss stagnation & divergence\n"
            "- LR issues · NaN/Inf values"
        )
    with f2:
        st.markdown("#### 🧠 Root Cause Analysis")
        st.markdown(
            "AI-powered diagnosis:\n"
            "- Confidence-scored likely causes\n"
            "- Evidence-based conditional reasoning\n"
            "- Multi-cause detection & ranking"
        )
    with f3:
        st.markdown("#### 💡 Smart Suggestions")
        st.markdown(
            "Actionable recommendations:\n"
            "- Ready-to-use code snippets\n"
            "- Hyperparameter change sets\n"
            "- Plain-English explanations\n"
            "- Best-practice guidance"
        )

    st.divider()
    st.subheader("🚀 Getting Started")
    st.markdown(
        "1. Go to **🚀 Run Training** and pick a scenario.\n"
        "2. Watch live training metrics and anomaly alerts.\n"
        "3. Open **🔍 Analysis & Diagnostics** for a full report.\n"
        "4. Use **⚙️ Hyperparameter Optimizer** to find the best config.\n"
        "5. Use **🧪 AI Advisor** for instant advice on any problem."
    )


# ============================================================================
# PAGE — RUN TRAINING
# ============================================================================
def page_run_training():
    st.markdown('<p class="gradient-header">Run Training Experiment</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Select a scenario to simulate common ML training '
        'problems — MLCopilot detects and diagnoses them in real time</p>',
        unsafe_allow_html=True,
    )

    st.subheader("Select Scenario")
    cols = st.columns(4)
    for i, (key, info) in enumerate(SCENARIOS.items()):
        with cols[i]:
            is_sel = st.session_state.selected_scenario == key
            st.markdown(f"### {info['title']}")
            st.caption(info["desc"])
            st.markdown(
                f"- **LR:** {info['lr']}  \n"
                f"- **Epochs:** {info['epochs']}  \n"
                f"- **Samples:** {info['samples']}  \n"
                f"- **Batch:** {info['batch']}"
            )
            if info["expect"]:
                st.warning(f"Expect: {', '.join(info['expect'])}")
            else:
                st.success("Expect: No issues")
            if st.button(
                "Selected ✓" if is_sel else "Select",
                key=f"sel_{key}",
                type="primary" if is_sel else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_scenario = key
                st.rerun()

    st.divider()

    skey = st.session_state.selected_scenario
    sinfo = SCENARIOS[skey]
    st.info(f"**Selected:** {sinfo['title']} — {sinfo['desc']}")

    if st.button("▶️  Start Training", type="primary", use_container_width=True):
        _execute_training(skey, sinfo)


def _execute_training(scenario_key: str, scenario_info: dict):
    current_exps = api_get("/api/experiments") or []
    current_ids = {e["id"] for e in current_exps}
    total_epochs = scenario_info["epochs"]

    with st.status(f"Training · {scenario_info['title']}", expanded=True) as status:
        st.write("Launching training subprocess …")

        proc = subprocess.Popen(
            [sys.executable,
             os.path.join(PROJECT_ROOT, "ml_pipeline", "sample_training.py"),
             "--scenario", scenario_key,
             "--api-url", API_URL],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        exp_id = None
        for _ in range(40):
            time.sleep(0.5)
            new_exps = api_get("/api/experiments") or []
            new_ids = {e["id"] for e in new_exps}
            diff = new_ids - current_ids
            if diff:
                exp_id = max(diff)
                break

        if not exp_id:
            st.error("Could not detect new experiment — is the backend running?")
            proc.kill()
            status.update(label="Training failed", state="error")
            return

        st.write(f"Experiment **#{exp_id}** created. Streaming metrics …")

        progress = st.progress(0.0, text="Starting …")
        metric_ph = st.empty()
        chart_ph = st.empty()
        issues_ph = st.empty()

        last_n = 0
        collected_issues: list[dict] = []

        while proc.poll() is None:
            metrics = api_get(f"/api/metrics/{exp_id}")
            if metrics and len(metrics) > last_n:
                last_n = len(metrics)
                df = pd.DataFrame(metrics)
                epoch_now = int(df["epoch"].max()) if "epoch" in df.columns else last_n
                pct = min(epoch_now / total_epochs, 1.0)
                progress.progress(pct, text=f"Epoch {epoch_now} / {total_epochs}")

                latest = df.iloc[-1]
                with metric_ph.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Loss",
                              f"{latest['loss']:.4f}" if latest.get("loss") is not None else "—")
                    m2.metric("Val Loss",
                              f"{latest['val_loss']:.4f}" if latest.get("val_loss") is not None else "—")
                    m3.metric("Accuracy",
                              f"{latest['accuracy']:.2%}" if latest.get("accuracy") is not None else "—")
                    m4.metric("Grad Norm",
                              f"{latest['grad_norm']:.6f}" if latest.get("grad_norm") is not None else "—")

                fig = chart_loss(df)
                fig.update_layout(height=320, title="")
                chart_ph.plotly_chart(fig, use_container_width=True)

                analysis = api_get(f"/api/analysis/{exp_id}")
                if analysis:
                    for a in analysis:
                        for iss in a.get("issues", []):
                            if iss not in collected_issues:
                                collected_issues.append(iss)
                    if collected_issues:
                        with issues_ph.container():
                            st.warning(f"⚠️ {len(collected_issues)} issue(s) detected so far")
                            for iss in collected_issues:
                                sev = SEVERITY_CFG.get(
                                    iss.get("severity", "medium"), SEVERITY_CFG["medium"]
                                )
                                st.markdown(
                                    f"{sev['icon']} **{iss['name']}** · "
                                    f"Epoch {iss.get('epoch','?')} · "
                                    f"_{iss.get('description','')}_"
                                )
            time.sleep(1.0)

        proc.wait()
        progress.progress(1.0, text="Complete!")
        status.update(label=f"Training complete — Experiment #{exp_id}", state="complete")

    st.session_state.last_exp_id = exp_id

    result = api_post("/api/analyze", {"experiment_id": exp_id, "window": total_epochs})
    if result:
        if result.get("status") == "healthy":
            st.balloons()
            st.success("🎉 No issues detected — training was healthy!")
        else:
            st.warning(f"⚠️ {len(result.get('issues', []))} issue(s) found")
            _render_suggestions(result.get("suggestions", []))
            if result.get("report"):
                st.download_button(
                    "📥 Download Report", result["report"],
                    file_name=f"mlcopilot_exp{exp_id}.txt", mime="text/plain",
                )
    st.info(
        "💡 Open **📈 Experiment Monitor** or **🔍 Analysis & Diagnostics** "
        "in the sidebar for a deep dive."
    )


# ============================================================================
# PAGE — EXPERIMENT MONITOR
# ============================================================================
def page_monitor():
    st.markdown('<p class="gradient-header">Experiment Monitor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Deep-dive into training metrics with interactive charts</p>',
        unsafe_allow_html=True,
    )

    experiments = api_get("/api/experiments")
    if not experiments:
        st.info("No experiments yet. Run a training first!")
        return

    exp_map = {
        e["id"]: f"#{e['id']} — {e['name']}  ({e.get('status', '')})"
        for e in experiments
    }
    default = st.session_state.last_exp_id or experiments[0]["id"]
    if default not in exp_map:
        default = experiments[0]["id"]

    sel_id = st.selectbox(
        "Experiment", list(exp_map.keys()),
        format_func=lambda x: exp_map[x],
        index=list(exp_map.keys()).index(default),
    )

    metrics = api_get(f"/api/metrics/{sel_id}")
    if not metrics:
        st.warning("No metrics recorded for this experiment.")
        return

    df = pd.DataFrame(metrics)

    st.divider()
    latest = df.iloc[-1]
    c = st.columns(5)
    c[0].metric("Epochs", len(df))
    c[1].metric("Loss", f"{latest['loss']:.4f}" if latest.get("loss") is not None else "—")
    c[2].metric("Val Loss", f"{latest['val_loss']:.4f}" if latest.get("val_loss") is not None else "—")
    c[3].metric("Accuracy", f"{latest['accuracy']:.2%}" if latest.get("accuracy") is not None else "—")
    c[4].metric("Grad Norm", f"{latest['grad_norm']:.6f}" if latest.get("grad_norm") is not None else "—")

    st.divider()

    t1, t2, t3, t4, t5 = st.tabs([
        "📉 Loss", "📈 Accuracy", "🔥 Gradients", "📊 Learning Rate", "🔎 Overview",
    ])
    with t1:
        st.plotly_chart(chart_loss(df).update_layout(height=460), use_container_width=True)
    with t2:
        st.plotly_chart(chart_accuracy(df).update_layout(height=460), use_container_width=True)
    with t3:
        st.plotly_chart(chart_gradient(df).update_layout(height=460), use_container_width=True)
    with t4:
        st.plotly_chart(chart_lr(df).update_layout(height=460), use_container_width=True)
    with t5:
        st.plotly_chart(chart_overview(df), use_container_width=True)

    with st.expander("📋 Raw Metrics Table"):
        show = [
            c for c in [
                "epoch", "loss", "val_loss", "accuracy", "val_accuracy",
                "grad_norm", "lr", "timestamp",
            ]
            if c in df.columns
        ]
        st.dataframe(df[show], use_container_width=True, hide_index=True)


# ============================================================================
# PAGE — ANALYSIS & DIAGNOSTICS
# ============================================================================
def page_analysis():
    st.markdown(
        '<p class="gradient-header">Analysis & Diagnostics</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Run anomaly detection, root-cause analysis, '
        'and get actionable fix suggestions</p>',
        unsafe_allow_html=True,
    )

    experiments = api_get("/api/experiments")
    if not experiments:
        st.info("No experiments yet.")
        return

    exp_map = {
        e["id"]: f"#{e['id']} — {e['name']}  ({e.get('status', '')})"
        for e in experiments
    }
    default = st.session_state.last_exp_id or experiments[0]["id"]
    if default not in exp_map:
        default = experiments[0]["id"]

    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        sel_id = st.selectbox(
            "Experiment", list(exp_map.keys()),
            format_func=lambda x: exp_map[x],
            index=list(exp_map.keys()).index(default),
            key="an_exp",
        )
    with c2:
        window = st.number_input("Window", min_value=5, max_value=200, value=30)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Analyzing …"):
            result = api_post("/api/analyze", {"experiment_id": sel_id, "window": window})

        if result:
            if result.get("status") == "healthy":
                st.success("✅ No issues detected — training looks healthy!")
                st.balloons()
            else:
                issues = result.get("issues", [])
                st.warning(f"⚠️ {len(issues)} issue(s) detected")

                st.subheader("🚨 Detected Issues")
                for iss in issues:
                    sev = iss.get("severity", "medium")
                    cfg = SEVERITY_CFG.get(sev, SEVERITY_CFG["medium"])
                    with st.expander(
                        f"{cfg['icon']} {iss['name']}  ·  {cfg['label']}  ·  "
                        f"Epoch {iss.get('epoch', '?')}",
                        expanded=True,
                    ):
                        st.markdown(f"**Description:** {iss.get('description', '')}")
                        if iss.get("evidence"):
                            st.json(iss["evidence"])

                if result.get("root_causes"):
                    st.subheader("🧠 Root Cause Analysis")
                    for rc in result["root_causes"]:
                        conf = rc.get("confidence", 0)
                        icon = "🟢" if conf >= 0.8 else ("🟡" if conf >= 0.6 else "🔴")
                        st.markdown(
                            f"{icon} **{rc['cause']}** — {conf:.0%} confidence  \n"
                            f"_{rc.get('description', '')}_"
                        )

                st.subheader("💡 Fix Suggestions")
                _render_suggestions(result.get("suggestions", []))

                if result.get("report"):
                    st.divider()
                    st.download_button(
                        "📥 Download Full Report", result["report"],
                        file_name=f"mlcopilot_report_exp{sel_id}.txt",
                        mime="text/plain",
                    )

    st.divider()
    st.subheader("📜 Analysis History")
    history = api_get(f"/api/analysis/{sel_id}")
    if history:
        for entry in reversed(history[-10:]):
            n_iss = len(entry.get("issues", []))
            icon = "✅" if n_iss == 0 else "⚠️"
            with st.expander(
                f"{icon} Epoch {entry.get('epoch', '?')} — "
                f"{n_iss} issue(s) — {entry.get('timestamp', '')}"
            ):
                for iss in entry.get("issues", []):
                    scfg = SEVERITY_CFG.get(
                        iss.get("severity", "medium"), SEVERITY_CFG["medium"]
                    )
                    st.markdown(
                        f"{scfg['icon']} **{iss['name']}**: {iss.get('description', '')}"
                    )
                for s in entry.get("suggestions", []):
                    st.markdown(
                        f"💡 **{s.get('problem', '')}**: "
                        f"{', '.join(s.get('fixes', [])[:2])}"
                    )
    else:
        st.info("No analyses yet — click **Analyze** above.")


# ============================================================================
# PAGE — HYPERPARAMETER OPTIMIZER
# ============================================================================
def page_optimizer():
    st.markdown(
        '<p class="gradient-header">Hyperparameter Optimizer</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Optuna-powered Bayesian optimisation '
        'to find the best training configuration</p>',
        unsafe_allow_html=True,
    )

    st.subheader("⚙️ Settings")
    s1, s2, s3 = st.columns(3)
    with s1:
        n_trials = st.slider("Trials", 5, 100, 25, step=5)
    with s2:
        max_epochs = st.slider("Epochs / trial", 5, 50, 15)
    with s3:
        n_samples = st.slider("Train samples", 200, 2000, 800, step=100)

    st.markdown(
        "**Search space:** LR `1e-5 → 0.1` · Hidden `32–256` · Depth `1–5` · "
        "Dropout `0–0.5` · Optimizer `Adam / AdamW / SGD` · Batch `16 / 32 / 64 / 128`"
    )

    st.divider()

    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        with st.status(f"Running {n_trials} trials …", expanded=True) as status:
            try:
                import torch
                from optimizer.hyperparameter_optimizer import HyperparameterOptimizer

                st.write("Generating synthetic dataset …")
                total_n = n_samples + 200
                X = torch.randn(total_n, 20)
                w = torch.randn(20)
                y = (X @ w > 0).long()
                X_tr, y_tr = X[:n_samples], y[:n_samples]
                X_va, y_va = X[n_samples:], y[n_samples:]

                st.write(f"Launching Optuna ({n_trials} trials, {max_epochs} epochs each) …")
                opt = HyperparameterOptimizer(
                    X_train=X_tr, y_train=y_tr,
                    X_val=X_va, y_val=y_va,
                    max_epochs=max_epochs,
                    n_trials=n_trials,
                )
                result = opt.optimize()
                status.update(label="Optimization complete!", state="complete")
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")
                import traceback
                st.code(traceback.format_exc())
                status.update(label="Failed", state="error")
                return

        st.success("🎉 Optimization complete!")

        st.subheader("🏆 Best Hyperparameters")
        best = result["best_params"]
        pcols = st.columns(min(len(best), 6))
        for i, (k, v) in enumerate(best.items()):
            with pcols[i % len(pcols)]:
                st.metric(k, f"{v:.6f}" if isinstance(v, float) else str(v))
        st.metric("Best Validation Loss", f"{result['best_val_loss']:.5f}")

        st.divider()

        trials_df = pd.DataFrame(result["all_trials"])
        valid = trials_df[trials_df["value"].notna()].copy()

        if not valid.empty:
            st.subheader("📊 Trial Analysis")
            g1, g2 = st.columns(2)

            with g1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=valid["number"], y=valid["value"],
                    mode="markers+lines",
                    marker=dict(
                        size=7, color=valid["value"],
                        colorscale="RdYlGn_r", showscale=True,
                        colorbar=dict(title="Val Loss"),
                    ),
                    line=dict(color="rgba(120,120,120,0.25)", width=1),
                    name="Trial",
                ))
                fig.add_hline(
                    y=result["best_val_loss"], line_dash="dot",
                    line_color=COLORS["success"],
                    annotation_text=f"Best: {result['best_val_loss']:.4f}",
                )
                fig.update_layout(
                    **CHART_LAYOUT, title="Validation Loss per Trial",
                    xaxis_title="Trial #", yaxis_title="Val Loss", height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            with g2:
                running_best = valid["value"].cummin()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=valid["number"], y=running_best,
                    fill="tozeroy", fillcolor="rgba(72,187,120,0.10)",
                    line=dict(color=COLORS["success"], width=2.5),
                    name="Best So Far",
                ))
                fig2.update_layout(
                    **CHART_LAYOUT, title="Optimization Progress",
                    xaxis_title="Trial #", yaxis_title="Best Val Loss", height=400,
                )
                st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 All Trials"):
            st.dataframe(trials_df, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE — AI ADVISOR
# ============================================================================
def page_advisor():
    st.markdown('<p class="gradient-header">AI Advisor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Get instant AI-powered diagnosis and fix suggestions '
        'for any ML training problem</p>',
        unsafe_allow_html=True,
    )

    problems_resp = api_get("/api/problems")
    if problems_resp:
        problems = problems_resp.get("problems", [])
    else:
        problems = [
            {"name": "Exploding Gradients", "description": "Gradient norms grow uncontrollably."},
            {"name": "Vanishing Gradients", "description": "Gradients become too small."},
            {"name": "Overfitting", "description": "Model memorizes training data."},
            {"name": "Underfitting", "description": "Model fails to learn patterns."},
            {"name": "Loss Stagnation", "description": "Training loss stops decreasing."},
            {"name": "Learning Rate Too High", "description": "LR causes unstable training."},
            {"name": "Loss Divergence", "description": "Loss grows out of control."},
            {"name": "NaN/Inf Detected", "description": "Numerical instability in training."},
        ]

    p_names = [p["name"] for p in problems]
    sel_problem = st.selectbox("Select a problem", p_names)
    for p in problems:
        if p["name"] == sel_problem:
            st.caption(p.get("description", ""))
            break

    severity = st.select_slider(
        "Severity level",
        options=["low", "medium", "high", "critical"],
        value="high",
    )

    if st.button("🔍 Get AI Suggestions", type="primary", use_container_width=True):
        with st.spinner("Generating suggestions …"):
            result = api_post("/api/suggest", {
                "problem": sel_problem,
                "severity": severity,
                "context": {},
            })

        if result and result.get("suggestions"):
            _render_suggestions(result["suggestions"])
        else:
            st.warning("No suggestions available for this problem.")

    st.divider()
    st.subheader("📚 Problem Reference")
    ref = pd.DataFrame(problems)
    st.dataframe(ref, use_container_width=True, hide_index=True)


# ============================================================================
# SHARED — RENDER SUGGESTIONS
# ============================================================================
def _render_suggestions(suggestions: list):
    for s in suggestions:
        sev = s.get("severity", "medium")
        cfg = SEVERITY_CFG.get(sev, SEVERITY_CFG["medium"])

        with st.expander(f"{cfg['icon']} {s['problem']} — Fix Suggestions", expanded=True):
            if s.get("root_causes"):
                st.markdown("**🧠 Root Causes**")
                for rc in s["root_causes"]:
                    conf = rc.get("confidence", 0)
                    st.progress(conf, text=f"{rc['cause']} — {conf:.0%}")

            st.markdown("**✅ Recommended Fixes**")
            for i, fix in enumerate(s.get("fixes", []), 1):
                st.markdown(f"{i}. {fix}")

            if s.get("param_changes"):
                st.markdown("**📊 Suggested Parameter Changes**")
                st.json(s["param_changes"])

            if s.get("code_suggestion"):
                st.markdown("**💻 Code Snippet**")
                st.code(s["code_suggestion"], language="python")

            if s.get("explanation"):
                st.markdown("**📖 Explanation**")
                st.info(s["explanation"])


# ============================================================================
# PAGE ROUTER
# ============================================================================
if page == "📊 Dashboard":
    page_dashboard()
elif page == "🚀 Run Training":
    page_run_training()
elif page == "📈 Experiment Monitor":
    page_monitor()
elif page == "🔍 Analysis & Diagnostics":
    page_analysis()
elif page == "⚙️ Hyperparameter Optimizer":
    page_optimizer()
elif page == "🧪 AI Advisor":
    page_advisor()
