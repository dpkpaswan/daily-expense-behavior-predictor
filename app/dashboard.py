"""
Interactive Streamlit Dashboard
Real-time expense prediction and financial stress forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import io
from pathlib import Path
import torch

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from utils.data_pipeline import DataPipeline
from utils.feature_engineering import FeatureEngineer
from utils.helpers import AlertSystem, ReportGenerator, ModelPersistence
from optimization.savings_optimizer import SavingsPlanOptimizer
from models.neural_networks import create_model

# Page configuration
st.set_page_config(
    page_title="Expense & Stress Predictor",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .alert-high {
        background-color: #ffecec;
        border-left: 5px solid #ff0000;
    }
    .alert-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .alert-low {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)


def _in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _infer_checkpoint_input_size(state_dict: dict) -> int | None:
    try:
        key = "models.transformer.embedding.weight"
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) == 2:
            return int(state_dict[key].shape[1])
        key = "models.gnn.node_encoder.weight"
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) == 2:
            return int(state_dict[key].shape[1])
        key = "models.tcn.tcn.0.weight"
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) == 3:
            return int(state_dict[key].shape[1])
    except Exception:
        return None
    return None


@st.cache_resource
def _load_ensemble_checkpoint(checkpoint_path: str, input_size: int, device: str = "cpu"):
    """Load an ensemble checkpoint safely.

    Returns (model_or_none, error_message_or_none, feature_cols_or_none).
    """
    try:
        loaded = torch.load(checkpoint_path, map_location=device)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded.get("state_dict")
            feature_cols = loaded.get("feature_cols")
            expected = loaded.get("input_size")
        else:
            state_dict = loaded
            feature_cols = None
            expected = None

        if not isinstance(state_dict, dict):
            return None, "Invalid checkpoint format.", None

        inferred = _infer_checkpoint_input_size(state_dict)
        expected_size = int(expected) if isinstance(expected, (int, float)) else inferred
        if expected_size is not None and int(expected_size) != int(input_size):
            return (
                None,
                f"Checkpoint expects {int(expected_size)} features, but current data has {int(input_size)}. "
                "Re-train the model to match current feature engineering.",
                feature_cols if isinstance(feature_cols, list) else None,
            )

        model = create_model("ensemble", input_size=input_size)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, None, feature_cols if isinstance(feature_cols, list) else None
    except Exception as e:
        return None, f"Failed to load checkpoint: {e}", None


def load_or_create_demo_data():
    """Load real dataset (if provided) or create demo dataset.

    Real data sources supported:
    - Sidebar CSV upload (stored in session_state)
    - Local CSV path set in session_state['data_path']
    """
    required_cols = {'daily_income_raw', 'total_spending_raw', 'impulse_frequency_raw', 'mood_score_raw'}

    def _raw_cols_look_invalid(existing: pd.DataFrame) -> bool:
        """Detect stale cached data where *_raw columns were accidentally scaled."""
        try:
            if not isinstance(existing, pd.DataFrame):
                return True
            if not required_cols.issubset(set(existing.columns)):
                return True

            income_raw = pd.to_numeric(existing.get('daily_income_raw'), errors='coerce')
            spending_raw = pd.to_numeric(existing.get('total_spending_raw'), errors='coerce')

            # Synthetic/raw income is expected to be in dollars (tens to hundreds).
            # If it's centered near 0, it's almost certainly scaled.
            if float(income_raw.mean(skipna=True)) < 10.0:
                return True

            # Spending should also be positive dollars (not ~0).
            if float(spending_raw.mean(skipna=True)) < 5.0:
                return True

            return False
        except Exception:
            return True

    if 'data' in st.session_state:
        try:
            existing = st.session_state['data']
            if _raw_cols_look_invalid(existing):
                del st.session_state['data']
                if 'engineered_data' in st.session_state:
                    del st.session_state['engineered_data']
        except Exception:
            if 'data' in st.session_state:
                del st.session_state['data']
            if 'engineered_data' in st.session_state:
                del st.session_state['engineered_data']

    if 'data' not in st.session_state:
        with st.spinner("Loading data..."):
            pipeline = DataPipeline()

            raw_df = None
            source_label = None
            try:
                if 'uploaded_csv_bytes' in st.session_state and st.session_state.get('uploaded_csv_bytes'):
                    raw_df = pd.read_csv(io.BytesIO(st.session_state['uploaded_csv_bytes']))
                    source_label = st.session_state.get('uploaded_csv_name', 'uploaded CSV')
                else:
                    data_path = st.session_state.get('data_path')
                    if isinstance(data_path, str) and data_path.strip() and Path(data_path).exists():
                        raw_df = pd.read_csv(data_path)
                        source_label = data_path
                    else:
                        # Convenience: auto-load a prepared dataset if present in the repo
                        default_path = Path(__file__).resolve().parents[1] / "data" / "expenses.csv"
                        if default_path.exists():
                            raw_df = pd.read_csv(default_path)
                            source_label = str(default_path)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                raw_df = None

            if raw_df is not None:
                try:
                    df, metadata = pipeline.preprocess_full_pipeline(df=raw_df)
                    metadata = dict(metadata)
                    metadata['data_source'] = f"csv:{source_label}" if source_label else "csv"
                except Exception as e:
                    st.error(
                        "Your CSV could not be processed. Ensure it includes at least 'daily_income' and either "
                        "'total_spending' or expense-category columns (e.g., grocery/utilities/entertainment/impulse_purchases).\n\n"
                        f"Details: {e}"
                    )
                    st.stop()
            else:
                df, metadata = pipeline.preprocess_full_pipeline(n_synthetic_samples=500)
                metadata = dict(metadata)
                metadata['data_source'] = 'synthetic'

            st.session_state['data'] = df
            st.session_state['metadata'] = metadata
    
    return st.session_state['data']


def compute_brs_and_features(df):
    """Compute BRS and features"""
    if 'engineered_data' in st.session_state:
        try:
            engineered = st.session_state['engineered_data']
            if not isinstance(engineered, pd.DataFrame) or 'behavioral_risk_score' not in engineered.columns:
                del st.session_state['engineered_data']
        except Exception:
            if 'engineered_data' in st.session_state:
                del st.session_state['engineered_data']

    if 'engineered_data' not in st.session_state:
        with st.spinner("Engineering features..."):
            engineer = FeatureEngineer()
            df = engineer.compute_behavioral_risk_score(
                df,
                impulse_freq_col="impulse_frequency_raw" if "impulse_frequency_raw" in df.columns else "impulse_frequency",
                overspend_col="total_spending_raw" if "total_spending_raw" in df.columns else "total_spending",
                income_col="daily_income_raw" if "daily_income_raw" in df.columns else "daily_income",
                mood_var_col="mood_variance_raw" if "mood_variance_raw" in df.columns else "mood_variance",
            )
            df = engineer.compute_rolling_features(df, window_size=7)
            df = engineer.compute_domain_specific_features(df)
            st.session_state['engineered_data'] = df
    
    return st.session_state['engineered_data']


def _predict_with_checkpoint(df: pd.DataFrame, daily_income: float, impulse_freq: float, mood_score: float, seq_len: int = 14):
    """Predict next-day spending using the trained ensemble if a checkpoint exists.

    Returns (predicted_spending, stress_risk_score, warning_message) or (None, None, warning_message).
    """
    checkpoint = Path(__file__).resolve().parents[1] / "models" / "best_model.pth"
    if not checkpoint.exists():
        return None, None, None

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    target_cols = [c for c in ["total_spending", "total_spending_raw"] if c in numeric_df.columns]
    feature_cols = [c for c in numeric_df.columns if c not in target_cols]
    if len(feature_cols) == 0:
        return None, None, None

    window = df.tail(seq_len).copy()
    if len(window) < seq_len:
        return None, None, None

    # Override the last row with user inputs (raw cols preferred)
    if "daily_income_raw" in window.columns:
        window.loc[window.index[-1], "daily_income_raw"] = float(daily_income)
    if "impulse_frequency_raw" in window.columns:
        window.loc[window.index[-1], "impulse_frequency_raw"] = float(impulse_freq)
    if "mood_score_raw" in window.columns:
        window.loc[window.index[-1], "mood_score_raw"] = float(mood_score)

    X = window[feature_cols].values.astype(np.float32)
    X = X.reshape(1, seq_len, -1)

    model, ckpt_err, _ = _load_ensemble_checkpoint(str(checkpoint), input_size=X.shape[2], device="cpu")
    if model is None:
        return None, None, ckpt_err
    with torch.no_grad():
        out = model(torch.tensor(X))
        y_pred = (out[0] if isinstance(out, tuple) else out).cpu().numpy().flatten()[0]

    # Compute stress risk score using the required BRS formula
    # BRS_raw = 0.42*ImpulseFreq + 0.35*(Overspend/Income) + 0.23*MoodVar
    mood_series = df["mood_score_raw"] if "mood_score_raw" in df.columns else df.get("mood_score")
    if mood_series is not None:
        last7 = pd.Series(mood_series.tail(6).tolist() + [float(mood_score)])
        mood_var = float(last7.std(ddof=1) if len(last7) > 1 else 0.0)
    else:
        mood_var = 0.0

    brs_raw_pred = 0.42 * float(impulse_freq) + 0.35 * (float(y_pred) / (float(daily_income) + 1e-8)) + 0.23 * mood_var

    # Normalize stress score to [0,1] based on historical BRS_raw values
    hist_imp = df["impulse_frequency_raw"] if "impulse_frequency_raw" in df.columns else df.get("impulse_frequency")
    hist_income = df["daily_income_raw"] if "daily_income_raw" in df.columns else df.get("daily_income")
    hist_spend = df["total_spending_raw"] if "total_spending_raw" in df.columns else df.get("total_spending")
    hist_mvar = df["mood_variance_raw"] if "mood_variance_raw" in df.columns else df.get("mood_variance")

    if hist_imp is not None and hist_income is not None and hist_spend is not None and hist_mvar is not None:
        hist_brs_raw = 0.42 * hist_imp.astype(float) + 0.35 * (hist_spend.astype(float) / (hist_income.astype(float) + 1e-8)) + 0.23 * hist_mvar.astype(float)
        lo, hi = float(hist_brs_raw.min()), float(hist_brs_raw.max())
        stress = (brs_raw_pred - lo) / (hi - lo + 1e-8)
    else:
        stress = brs_raw_pred

    stress = float(np.clip(stress, 0.0, 1.0))
    return float(y_pred), stress, None


def main():
    # Header
    st.markdown("# AI-Based Daily Expense Behavior Predictor")
    st.markdown("### Financial Stress Forecaster and Personalized Savings Planner")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        section = st.radio(
            "Select Section:",
            ["Dashboard", "Predictions", "Savings Plan", "Analysis", "Reports"]
        )
        
        st.markdown("---")
        st.markdown("### Data Controls")

        def _safe_rerun() -> None:
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        try:
            template_path = Path(__file__).resolve().parents[1] / "data" / "expense_template.csv"
            if template_path.exists():
                st.download_button(
                    "Download template CSV",
                    data=template_path.read_bytes(),
                    file_name="expense_template.csv",
                    mime="text/csv",
                    key="download_template_csv",
                )
        except Exception:
            pass
        
        uploaded = st.file_uploader("Upload expense CSV", type=["csv"], key="csv_uploader")
        if uploaded is not None:
            st.session_state['uploaded_csv_bytes'] = uploaded.getvalue()
            st.session_state['uploaded_csv_name'] = getattr(uploaded, 'name', 'uploaded.csv')

        st.text_input("Or local CSV path", key="data_path", placeholder="data/expenses.csv")

        if st.button("Refresh Data", key="refresh_btn"):
            if 'data' in st.session_state:
                del st.session_state['data']
            if 'engineered_data' in st.session_state:
                del st.session_state['engineered_data']
            _safe_rerun()
    
    # Main Content
    if section == "Dashboard":
        dashboard_section()
    elif section == "Predictions":
        predictions_section()
    elif section == "Savings Plan":
        savings_plan_section()
    elif section == "Analysis":
        analysis_section()
    elif section == "Reports":
        reports_section()


def dashboard_section():
    """Main dashboard section"""
    st.markdown("## Financial Dashboard")
    
    # Load data
    df = load_or_create_demo_data()
    df = compute_brs_and_features(df)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    income_col = 'daily_income_raw' if 'daily_income_raw' in df.columns else 'daily_income'
    spend_col = 'total_spending_raw' if 'total_spending_raw' in df.columns else 'total_spending'

    with col1:
        avg_income = float(df[income_col].mean())
        st.metric("Avg Daily Income", f"${avg_income:.2f}")
    
    with col2:
        avg_spending = float(df[spend_col].mean())
        st.metric("Avg Daily Spending", f"${avg_spending:.2f}")
    
    with col3:
        avg_brs = df['behavioral_risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_brs:.2%}")
    
    with col4:
        savings_rate = float(((df[income_col] - df[spend_col]) / (df[income_col] + 1e-8)).mean())
        st.metric("Avg Savings Rate", f"{savings_rate:.2%}")
    
    st.markdown("---")
    
    # Trend Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending vs Income
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            y=df[income_col].head(60),
            mode='lines',
            name='Daily Income',
            line=dict(color='green', width=2)
        ))
        fig1.add_trace(go.Scatter(
            y=df[spend_col].head(60),
            mode='lines',
            name='Total Spending',
            line=dict(color='red', width=2)
        ))
        fig1.update_layout(title="Income vs Spending Trend", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Behavioral Risk Score
        fig2 = go.Figure()
        colors = df['behavioral_risk_score'].head(60).apply(
            lambda x: 'red' if x > 0.6 else 'orange' if x > 0.3 else 'green'
        )
        fig2.add_trace(go.Bar(
            y=df['behavioral_risk_score'].head(60),
            marker_color=colors,
            name='BRS'
        ))
        fig2.update_layout(title="Behavioral Risk Score Trend", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Recent Activity
    st.markdown("### Recent Activity (Last 7 Days)")
    recent_df = df.tail(7)[['date', income_col, spend_col, 'behavioral_risk_score', 'impulse_purchases']]
    recent_df.columns = ['Date', 'Income', 'Spending', 'Risk Score', 'Impulse Purchases']
    st.dataframe(recent_df, use_container_width=True)


def predictions_section():
    """Predictions section"""
    st.markdown("## Expense and Stress Predictions")
    
    df = load_or_create_demo_data()
    df = compute_brs_and_features(df)
    
    # Input Parameters
    col1, col2, col3 = st.columns(3)

    income_col = 'daily_income_raw' if 'daily_income_raw' in df.columns else 'daily_income'
    impulse_col = 'impulse_frequency_raw' if 'impulse_frequency_raw' in df.columns else 'impulse_frequency'
    mood_col = 'mood_score_raw' if 'mood_score_raw' in df.columns else 'mood_score'

    def _safe_float(x, fallback: float) -> float:
        try:
            x = float(x)
            if np.isnan(x) or np.isinf(x):
                return float(fallback)
            return x
        except Exception:
            return float(fallback)

    def _safe_mean(df_: pd.DataFrame, col: str, fallback: float) -> float:
        if col not in df_.columns:
            return float(fallback)
        series = pd.to_numeric(df_[col], errors='coerce')
        m = float(series.mean(skipna=True))
        return _safe_float(m, fallback)

    # Guard against stale widget state from earlier runs
    try:
        if 'daily_income_input' in st.session_state and _safe_float(st.session_state['daily_income_input'], 10.0) < 10.0:
            st.session_state['daily_income_input'] = 10.0
    except Exception:
        st.session_state['daily_income_input'] = 10.0
    with col1:
        daily_income_default = max(10.0, _safe_mean(df, income_col, 120.0))
        daily_income = st.number_input(
            "Daily Income ($)",
            min_value=10.0,
            value=daily_income_default,
            step=5.0,
            key="daily_income_input"
        )
    
    with col2:
        impulse_default = _safe_mean(df, impulse_col, 0.3)
        impulse_default = float(np.clip(impulse_default, 0.0, 1.0))
        impulse_freq = st.slider(
            "Impulse Frequency (0-1)",
            0.0, 1.0, impulse_default, 0.1,
            key="impulse_freq_input"
        )
    
    with col3:
        mood_default = _safe_mean(df, mood_col, 6.0)
        mood_default = int(np.clip(round(mood_default), 1, 10))
        mood_score = st.slider(
            "Mood Score (1-10)",
            1, 10, mood_default, 1,
            key="mood_score_input"
        )
    
    # Predictions
    if st.button("Predict", key="predict_btn"):
        predicted_spending, predicted_stress, ckpt_warning = _predict_with_checkpoint(
            df,
            daily_income=daily_income,
            impulse_freq=impulse_freq,
            mood_score=mood_score,
            seq_len=14,
        )

        if predicted_spending is None:
            if ckpt_warning:
                st.warning(ckpt_warning)
            # Fallback heuristic if no checkpoint exists
            predicted_spending = daily_income * (0.3 + impulse_freq * 0.4)
            predicted_stress = impulse_freq * 0.5 + (1 - mood_score/10) * 0.3 + (predicted_spending/daily_income - 0.5) * 0.2
            predicted_stress = float(np.clip(predicted_stress, 0.0, 1.0))
        
        # Display results
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Daily Spending",
                f"${predicted_spending:.2f}",
                f"{(predicted_spending/daily_income)*100:.1f}% of income"
            )
        
        with col2:
            stress_level = AlertSystem.get_stress_level(predicted_stress)
            st.metric(
                "Financial Stress Level",
                stress_level.upper(),
                f"Risk: {predicted_stress:.2%}"
            )
        
        with col3:
            daily_surplus = daily_income - predicted_spending
            st.metric(
                "Predicted Daily Surplus",
                f"${daily_surplus:.2f}",
                f"+{(daily_surplus/daily_income)*100:.1f}%" if daily_surplus > 0 else f"{(daily_surplus/daily_income)*100:.1f}%"
            )
        
        # Alert Message
        st.markdown("---")
        alert_msg = AlertSystem.generate_alert_message(predicted_stress, predicted_spending, daily_income)
        
        if predicted_stress > 0.6:
            st.error(alert_msg)
        elif predicted_stress > 0.3:
            st.warning(alert_msg)
        else:
            st.success(alert_msg)
        
        # Recommendations
        st.markdown("### Personalized Recommendations")
        recommendations = AlertSystem.get_recommendations(predicted_stress, predicted_spending/daily_income)
        for rec in recommendations:
            st.info(rec)


def savings_plan_section():
    """Savings plan section"""
    st.markdown("## 30-Day Personalized Savings Plan")
    
    df = load_or_create_demo_data()
    df = compute_brs_and_features(df)
    
    # Input parameters
    col1, col2, col3 = st.columns(3)
    
    income_col = 'daily_income_raw' if 'daily_income_raw' in df.columns else 'daily_income'
    grocery_col = 'grocery_expense_raw' if 'grocery_expense_raw' in df.columns else 'grocery_expense'
    util_col = 'utilities_expense_raw' if 'utilities_expense_raw' in df.columns else 'utilities_expense'
    ent_col = 'entertainment_expense_raw' if 'entertainment_expense_raw' in df.columns else 'entertainment_expense'
    impulse_col = 'impulse_purchases_raw' if 'impulse_purchases_raw' in df.columns else 'impulse_purchases'

    def _safe_float(x, fallback: float) -> float:
        try:
            x = float(x)
            if np.isnan(x) or np.isinf(x):
                return float(fallback)
            return x
        except Exception:
            return float(fallback)

    def _safe_mean(df_: pd.DataFrame, col: str, fallback: float) -> float:
        if col not in df_.columns:
            return float(fallback)
        series = pd.to_numeric(df_[col], errors='coerce')
        m = float(series.mean(skipna=True))
        return _safe_float(m, fallback)

    try:
        if 'fixed_expenses_input' in st.session_state and _safe_float(st.session_state['fixed_expenses_input'], 0.0) < 0.0:
            st.session_state['fixed_expenses_input'] = 0.0
    except Exception:
        st.session_state['fixed_expenses_input'] = 0.0

    try:
        if 'variable_expenses_input' in st.session_state and _safe_float(st.session_state['variable_expenses_input'], 0.0) < 0.0:
            st.session_state['variable_expenses_input'] = 0.0
    except Exception:
        st.session_state['variable_expenses_input'] = 0.0

    try:
        if 'sp_income' in st.session_state and _safe_float(st.session_state['sp_income'], 10.0) < 10.0:
            st.session_state['sp_income'] = 10.0
    except Exception:
        st.session_state['sp_income'] = 10.0

    with col1:
        daily_income = st.number_input(
            "Daily Income ($)",
            min_value=10.0,
            value=max(10.0, _safe_mean(df, income_col, 120.0)),
            step=5.0,
            key="sp_income"
        )
    
    with col2:
        fixed_expenses = st.number_input(
            "Daily Fixed Expenses ($)",
            min_value=0.0,
            value=max(0.0, _safe_mean(df, util_col, 30.0) + _safe_mean(df, grocery_col, 20.0) * 0.3),
            step=5.0,
            key="fixed_expenses_input"
        )
    
    with col3:
        variable_expenses = st.number_input(
            "Daily Variable Expenses ($)",
            min_value=0.0,
            value=max(0.0, _safe_mean(df, ent_col, 10.0) + _safe_mean(df, impulse_col, 10.0)),
            step=5.0,
            key="variable_expenses_input"
        )
    
    # Generate plan
    if st.button("Generate Savings Plan", key="gen_plan_btn"):
        optimizer = SavingsPlanOptimizer(days=30)
        brs = df['behavioral_risk_score'].mean()
        
        plan_df = optimizer.generate_savings_plan(
            daily_income=daily_income,
            avg_fixed_expenses=fixed_expenses,
            avg_variable_expenses=variable_expenses,
            behavioral_risk_score=brs,
            behavioral_persona=0
        )
        
        # Summary metrics
        summary = optimizer.get_savings_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "30-Day Total Savings",
                f"${summary['total_savings_30_days']:.2f}"
            )
        
        with col2:
            st.metric(
                "Daily Avg Savings",
                f"${summary['average_daily_savings']:.2f}"
            )
        
        with col3:
            st.metric(
                "Total Fixed Expenses",
                f"${summary['total_fixed_expenses']:.2f}"
            )
        
        with col4:
            st.metric(
                "Total Variable Expenses",
                f"${summary['total_variable_expenses']:.2f}"
            )
        
        st.markdown("---")
        
        # Plan visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=plan_df['day'],
            y=plan_df['recommended_savings'],
            name='Daily Savings',
            marker_color='green'
        ))
        
        fig.add_trace(go.Scatter(
            x=plan_df['day'],
            y=plan_df['cumulative_savings'],
            name='Cumulative Savings',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="30-Day Savings Plan",
            xaxis_title="Day",
            yaxis_title="Daily Savings ($)",
            yaxis2=dict(title="Cumulative Savings ($)", overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plan table
        st.markdown("### Daily Breakdown")
        display_df = plan_df[['day', 'daily_income', 'fixed_expenses', 
                              'variable_expenses', 'recommended_savings', 
                              'cumulative_savings']].copy()
        display_df.columns = ['Day', 'Income', 'Fixed', 'Variable', 'Savings', 'Cumulative']
        st.dataframe(display_df, use_container_width=True)


def analysis_section():
    """Analysis section"""
    st.markdown("## Behavioral Analysis")
    
    df = load_or_create_demo_data()
    df = compute_brs_and_features(df)
    
    # Persona Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'behavioral_persona' in df.columns:
            persona_counts = df['behavioral_persona'].value_counts()
            fig = px.pie(
                values=persona_counts.values,
                names=[f"Persona {i}" for i in persona_counts.index],
                title="Behavioral Persona Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # BRS Distribution
        fig = px.histogram(
            df['behavioral_risk_score'],
            nbins=20,
            title="Risk Score Distribution",
            labels={'value': 'BRS', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### Feature Correlations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_cols = [col for col in numeric_cols 
                 if col not in ['date', 'behavioral_persona', 'day_of_week']][:10]
    
    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title="Feature Correlation Matrix", height=500)
        st.plotly_chart(fig, use_container_width=True)


def reports_section():
    """Reports section"""
    st.markdown("## Reports and Insights")
    
    df = load_or_create_demo_data()
    df = compute_brs_and_features(df)
    
    # Generate Daily Report
    st.markdown("### Daily Report")
    
    selected_date = st.date_input("Select Date", value=datetime.now().date())
    
    if st.button("Generate Daily Report", key="gen_report_btn"):
        # Get data for selected date
        idx = min(len(df) - 1, 0)  # Demo: use latest data
        row = df.iloc[idx]
        
        daily_report = ReportGenerator.generate_daily_report(
            date=str(selected_date),
            brs=row['behavioral_risk_score'],
            daily_spending=row['total_spending'],
            daily_income=row['daily_income'],
            predicted_stress=row['behavioral_risk_score'],
            recommendations=AlertSystem.get_recommendations(
                row['behavioral_risk_score'],
                row['total_spending'] / row['daily_income']
            )
        )
        
        # Display report
        st.json(daily_report)
    
    # Generate Monthly Report
    st.markdown("### Monthly Summary")
    
    if st.button("Generate Monthly Report", key="gen_monthly_btn"):
        # Simulate daily reports
        daily_reports = []
        for i in range(min(30, len(df))):
            row = df.iloc[i]
            daily_reports.append({
                'date': str((datetime.now() - timedelta(days=30-i)).date()),
                'behavioral_risk_score': row['behavioral_risk_score'],
                'daily_income': row['daily_income'],
                'daily_spending': row['total_spending'],
                'spending_ratio': row['total_spending'] / row['daily_income'],
                'daily_surplus': row['daily_income'] - row['total_spending'],
            })
        
        monthly_report = ReportGenerator.generate_monthly_report(daily_reports)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Income", f"${monthly_report['total_income']:.2f}")
        
        with col2:
            st.metric("Total Spending", f"${monthly_report['total_spending']:.2f}")
        
        with col3:
            st.metric("Total Surplus", f"${monthly_report['total_surplus']:.2f}")
        
        with col4:
            st.metric("Savings Rate", f"{monthly_report['savings_rate']:.2%}")
        
        st.markdown("---")
        st.json(monthly_report)


if __name__ == "__main__":
    if not _in_streamlit_runtime():
        print("This app must be launched with Streamlit.")
        print("Run: streamlit run app/dashboard.py")
        raise SystemExit(1)
    main()
