"""
MLCopilot AI — Streamlit Dashboard  (dashboard/app.py)
======================================================
Interactive web interface for monitoring ML training runs, visualising
metrics, and viewing detected issues with fix suggestions.

Sections
--------
1. Sidebar      — run selector + auto-refresh toggle
2. Metrics tab  — live loss / accuracy / gradient charts
3. Analysis tab — detected issues + LLM-enhanced explanations + suggestions

Run:
    streamlit run dashboard/app.py
"""

import sys
import os
import time

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

API_URL = os.getenv("MLCOPILOT_API_URL", "http://localhost:8000")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MLCopilot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "train":    "#667eea",
    "val":      "#ed8936",
    "accuracy": "#48bb78",
    "grad":     "#e53e3e",
    "lr":       "#9f7aea",
    "critical": "#e53e3e",
    "high":     "#ed8936",
    "medium":   "#ecc94b",
    "low":      "#48bb78",
}

SEVERITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str, params: dict = None):
    """Return JSON from the backend or None on error."""
    try:
        resp = requests.get(f"{API_URL}{path}", params=params, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_all_metrics() -> pd.DataFrame:
    data = _get("/metrics")
    if not data or not data.get("metrics"):
        return pd.DataFrame()
    return pd.DataFrame(data["metrics"])


def fetch_run_metrics(run_id: str) -> pd.DataFrame:
    data = _get("/metrics", params={"run_id": run_id})
    if not data or not data.get("metrics"):
        return pd.DataFrame()
    return pd.DataFrame(data["metrics"])


def fetch_analysis(run_id: str) -> dict:
    return _get("/analysis", params={"run_id": run_id}) or {}


def server_healthy() -> bool:
    result = _get("/health")
    return result is not None


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_loss(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "train_loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["train_loss"],
            name="Train Loss", line=dict(color=C["train"], width=2),
        ))
    if "val_loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["val_loss"],
            name="Val Loss", line=dict(color=C["val"], width=2, dash="dash"),
        ))
    fig.update_layout(
        title="Loss Curves", xaxis_title="Epoch", yaxis_title="Loss",
        legend=dict(x=0.01, y=0.99), template="plotly_dark",
        margin=dict(t=40, b=30),
    )
    return fig


def plot_accuracy(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "accuracy" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["accuracy"],
            name="Accuracy", line=dict(color=C["accuracy"], width=2),
            fill="tozeroy", fillcolor="rgba(72,187,120,0.1)",
        ))
    fig.update_layout(
        title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]), template="plotly_dark",
        margin=dict(t=40, b=30),
    )
    return fig


def plot_grad_norm(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "gradient_norm" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["gradient_norm"],
            name="Gradient Norm", line=dict(color=C["grad"], width=2),
        ))
        # Exploding gradient threshold line
        fig.add_hline(y=10.0, line_dash="dot", line_color="orange",
                      annotation_text="Exploding threshold (10)")
    fig.update_layout(
        title="Gradient Norm", xaxis_title="Epoch", yaxis_title="‖∇‖",
        template="plotly_dark", margin=dict(t=40, b=30),
    )
    return fig


def plot_lr(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "learning_rate" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["learning_rate"],
            name="Learning Rate", line=dict(color=C["lr"], width=2),
        ))
    fig.update_layout(
        title="Learning Rate Schedule", xaxis_title="Epoch", yaxis_title="LR",
        template="plotly_dark", margin=dict(t=40, b=30),
    )
    return fig


# ── Issue cards ───────────────────────────────────────────────────────────────

def render_issue(issue: dict):
    sev   = issue.get("severity", "low")
    icon  = SEVERITY_ICON.get(sev, "⚪")
    color = C.get(sev, "#ccc")

    with st.container():
        st.markdown(
            f"""
            <div style="border-left: 4px solid {color}; padding: 12px 16px;
                        background: rgba(255,255,255,0.03); border-radius: 6px;
                        margin-bottom: 12px;">
              <h4 style="margin:0; color:{color};">{icon} {issue.get('issue')}
                <span style="font-size:0.7em; font-weight:normal;
                             background:{color}20; padding:2px 8px;
                             border-radius:4px; margin-left:8px;">{sev.upper()}</span>
              </h4>
              <p style="margin:6px 0 0; color:#ccc;">{issue.get('reason','')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # LLM explanation
        explanation = issue.get("llm_explanation", "")
        if explanation:
            with st.expander("💡 Explanation"):
                st.info(explanation)

        # Suggestions
        suggestions = issue.get("suggestions", [])
        if suggestions:
            with st.expander("🔧 Fix Suggestions"):
                for s in suggestions:
                    st.markdown(f"- {s}")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='text-align:center; background: linear-gradient(135deg,#667eea,#764ba2);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   font-size:2.5rem;'>
          🤖 MLCopilot AI
        </h1>
        <p style='text-align:center; color:#aaa; margin-top:-10px;'>
          Real-time ML Training Monitor &amp; Debugger
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Server health ─────────────────────────────────────────────────────────
    if not server_healthy():
        st.error(
            f"⚠️  Cannot reach the backend at **{API_URL}**. "
            "Start it with: `uvicorn backend.main:app --reload --port 8000`"
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/robot.png", width=80)
        st.title("MLCopilot AI")
        st.caption("AI for Bharat Hackathon — 2026")
        st.markdown("---")

        # Pull all metric rows to derive run IDs
        all_metrics = fetch_all_metrics()
        run_ids = sorted(all_metrics["run_id"].unique().tolist()) if not all_metrics.empty else []

        if not run_ids:
            st.warning("No training runs found yet.\n\nRun:\n```\npython training_example/train_model.py\n```")
            st.stop()

        selected_run = st.selectbox("📁 Select Training Run", run_ids, index=len(run_ids) - 1)

        auto_refresh = st.toggle("🔄 Auto-refresh (5 s)", value=False)

        st.markdown("---")
        st.markdown(f"**Backend:** `{API_URL}`")
        st.markdown("[API Docs](%s/docs)" % API_URL)

    # ── Load data for selected run ─────────────────────────────────────────────
    df = fetch_run_metrics(selected_run)

    if df.empty:
        st.info("No metrics logged for this run yet.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    latest = df.iloc[-1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Epochs Logged",  int(df.shape[0]))
    k2.metric("Latest Train Loss", f"{latest.get('train_loss', 0):.4f}")
    k3.metric("Latest Val Loss",
              f"{latest['val_loss']:.4f}" if latest.get("val_loss") else "—")
    k4.metric("Latest Accuracy",
              f"{latest['accuracy']:.2%}" if latest.get("accuracy") else "—")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_metrics, tab_analysis = st.tabs(["📈 Live Metrics", "🔍 Analysis & Suggestions"])

    with tab_metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_loss(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_accuracy(df), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_grad_norm(df), use_container_width=True)
        with col4:
            st.plotly_chart(plot_lr(df), use_container_width=True)

        with st.expander("📋 Raw Metrics Table"):
            st.dataframe(df.drop(columns=["id"], errors="ignore"), use_container_width=True)

    with tab_analysis:
        if st.button("▶  Run Analysis", type="primary"):
            with st.spinner("Analysing training run…"):
                result = fetch_analysis(selected_run)

            if not result:
                st.error("Could not reach the analysis endpoint.")
            elif result.get("total_issues", 0) == 0:
                st.success("✅ No issues detected — the training run looks healthy!")
            else:
                st.error(f"⚠️  **{result['total_issues']} issue(s) detected**")
                for issue in result.get("results", []):
                    render_issue(issue)

        st.caption("Click **Run Analysis** to fetch the latest root-cause report.")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
