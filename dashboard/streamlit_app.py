"""
MLCopilot AI - Streamlit Dashboard
Interactive web dashboard for monitoring training and viewing analysis results.

Run with:  streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import json
import time

API_URL = "http://localhost:8000"

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="MLCopilot AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================
# Sidebar
# ============================
st.sidebar.title("🤖 MLCopilot AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Training Monitor", "🔍 Analysis", "⚙️ Optimizer", "🧪 Quick Diagnose"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**API Status:**")
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    st.sidebar.success(f"✅ Backend connected")
except Exception:
    st.sidebar.error("❌ Backend offline")
    st.sidebar.info("Start the server:\n```\nuvicorn backend.main_api:app --reload\n```")


# ============================
# Helper functions
# ============================

def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint, data):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ============================
# Pages
# ============================

def page_overview():
    st.title("🤖 MLCopilot AI Dashboard")
    st.markdown(
        """
        **MLCopilot AI** is an intelligent assistant for ML engineers that:
        - 📊 Monitors model training in real time
        - 🔍 Detects anomalies in training metrics
        - 🧠 Performs root-cause analysis
        - 💡 Suggests fixes to code and hyperparameters
        - 📝 Explains problems in natural language
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    experiments = api_get("/api/experiments")
    if experiments:
        with col1:
            st.metric("Total Experiments", len(experiments))
        running = sum(1 for e in experiments if e.get("status") == "running")
        with col2:
            st.metric("Running", running)
        completed = sum(1 for e in experiments if e.get("status") == "completed")
        with col3:
            st.metric("Completed", completed)

        st.markdown("### Recent Experiments")
        df = pd.DataFrame(experiments)
        if not df.empty:
            display_cols = [c for c in ["id", "name", "status", "created_at"] if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No experiments found. Run a training script to get started!")

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.code(
        """# 1. Start the backend server:
uvicorn backend.main_api:app --reload

# 2. Run a sample training scenario:
python ml_pipeline/sample_training.py --scenario exploding_gradients
python ml_pipeline/sample_training.py --scenario overfitting
python ml_pipeline/sample_training.py --scenario healthy

# 3. Open this dashboard:
streamlit run dashboard/streamlit_app.py""",
        language="bash",
    )


def page_training_monitor():
    st.title("📊 Training Monitor")

    experiments = api_get("/api/experiments")
    if not experiments:
        st.info("No experiments found.")
        return

    exp_names = {e["id"]: f"#{e['id']} - {e['name']} ({e['status']})" for e in experiments}
    selected_id = st.selectbox(
        "Select Experiment",
        options=list(exp_names.keys()),
        format_func=lambda x: exp_names[x],
    )

    if selected_id:
        metrics = api_get(f"/api/metrics/{selected_id}")
        if not metrics:
            st.warning("No metrics recorded for this experiment yet.")
            return

        df = pd.DataFrame(metrics)

        # Key metrics cards
        if not df.empty:
            latest = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                val = latest.get("loss")
                st.metric("Loss", f"{val:.4f}" if val else "N/A")
            with col2:
                val = latest.get("val_loss")
                st.metric("Val Loss", f"{val:.4f}" if val else "N/A")
            with col3:
                val = latest.get("accuracy")
                st.metric("Accuracy", f"{val:.2%}" if val else "N/A")
            with col4:
                val = latest.get("grad_norm")
                st.metric("Grad Norm", f"{val:.4f}" if val else "N/A")

        st.markdown("---")

        # Loss curves
        st.subheader("📈 Loss Curves")
        loss_cols = [c for c in ["loss", "val_loss"] if c in df.columns and df[c].notna().any()]
        if loss_cols and "epoch" in df.columns:
            chart_df = df.set_index("epoch")[loss_cols]
            st.line_chart(chart_df)

        # Accuracy curves
        acc_cols = [c for c in ["accuracy", "val_accuracy"] if c in df.columns and df[c].notna().any()]
        if acc_cols:
            st.subheader("📈 Accuracy Curves")
            chart_df = df.set_index("epoch")[acc_cols]
            st.line_chart(chart_df)

        # Gradient norm
        if "grad_norm" in df.columns and df["grad_norm"].notna().any():
            st.subheader("📈 Gradient Norm")
            chart_df = df.set_index("epoch")[["grad_norm"]]
            st.line_chart(chart_df)

        # Raw metrics table
        with st.expander("📋 Raw Metrics Table"):
            display_cols = [c for c in ["epoch", "loss", "val_loss", "accuracy", "val_accuracy", "grad_norm", "lr"] if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)


def page_analysis():
    st.title("🔍 Analysis & Diagnostics")

    experiments = api_get("/api/experiments")
    if not experiments:
        st.info("No experiments found.")
        return

    exp_names = {e["id"]: f"#{e['id']} - {e['name']} ({e['status']})" for e in experiments}
    selected_id = st.selectbox(
        "Select Experiment",
        options=list(exp_names.keys()),
        format_func=lambda x: exp_names[x],
        key="analysis_exp",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        window = st.number_input("Analysis Window (epochs)", min_value=5, max_value=100, value=20)
    with col2:
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Analyzing training metrics..."):
                result = api_post("/api/analyze", {"experiment_id": selected_id, "window": window})

            if result:
                if result.get("status") == "healthy":
                    st.success("✅ No issues detected! Training looks healthy.")
                else:
                    st.warning(f"⚠️ {len(result.get('issues', []))} issue(s) detected")

                    # Display issues
                    for issue in result.get("issues", []):
                        severity_color = {
                            "critical": "🔴",
                            "high": "🟠",
                            "medium": "🟡",
                            "low": "🟢",
                        }.get(issue.get("severity", "medium"), "⚪")

                        with st.expander(f"{severity_color} {issue['name']} (Epoch {issue.get('epoch', '?')})"):
                            st.markdown(f"**Severity:** {issue['severity']}")
                            st.markdown(f"**Description:** {issue['description']}")
                            if issue.get("evidence"):
                                st.json(issue["evidence"])

                    # Display suggestions
                    st.markdown("---")
                    st.subheader("💡 Suggestions")
                    for s in result.get("suggestions", []):
                        with st.expander(f"Fix: {s['problem']}"):
                            if s.get("root_causes"):
                                st.markdown("**Root Causes:**")
                                for rc in s["root_causes"]:
                                    st.markdown(f"- {rc['cause']} (confidence: {rc['confidence']:.0%})")

                            st.markdown("**Recommended Fixes:**")
                            for fix in s.get("fixes", []):
                                st.markdown(f"→ {fix}")

                            if s.get("code_suggestion"):
                                st.markdown("**Code Suggestion:**")
                                st.code(s["code_suggestion"], language="python")

                            st.markdown("**Explanation:**")
                            st.info(s.get("explanation", ""))

                    # Full report
                    if result.get("report"):
                        with st.expander("📄 Full Text Report"):
                            st.code(result["report"], language="text")

    # Show previous analyses
    st.markdown("---")
    st.subheader("📜 Analysis History")
    history = api_get(f"/api/analysis/{selected_id}")
    if history:
        for entry in reversed(history[-5:]):
            with st.expander(f"Analysis @ Epoch {entry.get('epoch', '?')} — {entry.get('timestamp', '')}"):
                if entry.get("issues"):
                    st.markdown(f"**Issues:** {len(entry['issues'])}")
                    for issue in entry["issues"]:
                        st.markdown(f"- {issue.get('name', '?')}: {issue.get('description', '')}")
    else:
        st.info("No analyses run yet for this experiment.")


def page_optimizer():
    st.title("⚙️ Hyperparameter Optimizer")
    st.markdown("Use Optuna to find optimal hyperparameters for your model.")

    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.slider("Number of trials", 5, 100, 20)
    with col2:
        max_epochs = st.slider("Max epochs per trial", 5, 50, 15)

    if st.button("🚀 Start Optimization", type="primary"):
        with st.spinner(f"Running {n_trials} optimization trials..."):
            try:
                import torch
                from optimizer.hyperparameter_optimizer import HyperparameterOptimizer

                # Generate data
                X = torch.randn(1000, 20)
                weights = torch.randn(20)
                y = (X @ weights > 0).long()
                X_train, y_train = X[:800], y[:800]
                X_val, y_val = X[800:], y[800:]

                opt = HyperparameterOptimizer(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    max_epochs=max_epochs,
                    n_trials=n_trials,
                )
                result = opt.optimize()

                st.success("Optimization complete!")

                st.subheader("🏆 Best Hyperparameters")
                st.json(result["best_params"])
                st.metric("Best Validation Loss", f"{result['best_val_loss']:.4f}")

                # Show all trials
                trials_df = pd.DataFrame(result["all_trials"])
                if not trials_df.empty and "value" in trials_df.columns:
                    st.subheader("📊 Trial History")
                    valid_trials = trials_df[trials_df["value"].notna()]
                    if not valid_trials.empty:
                        st.line_chart(valid_trials.set_index("number")["value"])
                    st.dataframe(trials_df, use_container_width=True)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def page_quick_diagnose():
    st.title("🧪 Quick Diagnose")
    st.markdown("Get instant suggestions for a specific ML training problem.")

    problems = api_get("/api/problems")
    if problems:
        problem_names = [p["name"] for p in problems.get("problems", [])]
    else:
        problem_names = [
            "Exploding Gradients", "Vanishing Gradients", "Overfitting",
            "Underfitting", "Loss Stagnation", "Learning Rate Too High",
        ]

    selected_problem = st.selectbox("Select a problem", problem_names)

    if st.button("🔍 Get Suggestions", type="primary"):
        with st.spinner("Generating suggestions..."):
            result = api_post("/api/suggest", {
                "problem": selected_problem,
                "severity": "high",
                "context": {},
            })

        if result and result.get("suggestions"):
            for s in result["suggestions"]:
                st.subheader(f"Problem: {s['problem']}")

                if s.get("root_causes"):
                    st.markdown("**Root Causes:**")
                    for rc in s["root_causes"]:
                        st.markdown(f"- {rc['cause']} ({rc['confidence']:.0%})")

                st.markdown("**Fixes:**")
                for fix in s.get("fixes", []):
                    st.markdown(f"→ {fix}")

                if s.get("param_changes"):
                    st.markdown("**Parameter Changes:**")
                    st.json(s["param_changes"])

                if s.get("code_suggestion"):
                    st.markdown("**Code:**")
                    st.code(s["code_suggestion"], language="python")

                st.info(s.get("explanation", ""))


# ============================
# Page Router
# ============================

if page == "🏠 Overview":
    page_overview()
elif page == "📊 Training Monitor":
    page_training_monitor()
elif page == "🔍 Analysis":
    page_analysis()
elif page == "⚙️ Optimizer":
    page_optimizer()
elif page == "🧪 Quick Diagnose":
    page_quick_diagnose()
