"""
Production-Grade Monitoring Dashboard
- PSI + KS drift detection
- Dual Phase (German Credit + LendingClub)
- All metrics loaded from JSON files
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp


st.set_page_config(page_title="Credit Risk Monitor (Production)", layout="wide")


# DRIFT FUNCTIONS


def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index (PSI) - Industry standard for drift detection"""
    expected = np.array(expected)
    actual = np.array(actual)
    # Create bins based on expected distribution percentiles
    breakpoints = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    # Avoid division by zero
    expected_counts = np.clip(expected_counts, 1e-6, 1)
    actual_counts = np.clip(actual_counts, 1e-6, 1)
    psi_values = (actual_counts - expected_counts) * np.log(actual_counts / expected_counts)
    return np.sum(psi_values)


def calculate_ks(expected, actual):
    """Kolmogorov-Smirnov test for distribution shift"""
    stat, p_value = ks_2samp(expected, actual)
    return stat, p_value


def drift_status(psi):
    """Interpret PSI score - Industry standard thresholds"""
    if psi < 0.1:
        return "🟢 Stable"
    elif psi < 0.25:
        return "🟡 Moderate"
    else:
        return "🔴 Significant"


def drift_severity(psi):
    """Return severity level for coloring"""
    if psi < 0.1:
        return "low"
    elif psi < 0.25:
        return "medium"
    else:
        return "high"


# LOAD DATA FUNCTIONS


def load_metrics_phase1():
    """
    Load Phase 1 metrics from reports/metrics.json
    Phase 1: German Credit dataset
    """
    metrics_path = Path("reports/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            data = json.load(f)
            return data
    return {}


def load_metrics_phase2():
    """
    Load Phase 2 metrics from reports/metrics_full.json
    Phase 2: LendingClub dataset
    """
    metrics_path = Path("reports/metrics_full.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            data = json.load(f)
            return data
    return {}


def load_reference_data_phase1():
    ref_path = Path("data/processed/features_german.csv")
    if ref_path.exists():
        return pd.read_csv(ref_path)
    return None


def load_reference_data_phase2():
    ref_path = Path("data/processed/features_lending.csv")
    if ref_path.exists():
        return pd.read_csv(ref_path)
    return None


def load_streaming_data_phase1():
    stream_path = Path("data/streaming/german/all_loans.csv")
    if stream_path.exists():
        return pd.read_csv(stream_path)
    return None


def load_streaming_data_phase2():
    stream_path = Path("data/streaming/all_loans.csv")
    if stream_path.exists():
        return pd.read_csv(stream_path)
    return None


# DRIFT ANALYSIS FUNCTION


def run_drift_analysis(ref_df, stream_df, numeric_cols, phase="phase1"):
    """
    Run PSI + KS drift analysis

    Args:
        ref_df: Reference/baseline dataframe
        stream_df: Streaming/current dataframe
        numeric_cols: List of numeric columns to analyze
        phase: "phase1" or "phase2" to handle different column naming
    """
    results = []

    ref_filtered = ref_df.copy()
    stream_filtered = stream_df.copy()

    for col in numeric_cols:
        # Map column names based on phase
        if phase == "phase1":
            # Phase 1 uses capitalized column names: Age, Credit amount, Duration, Job
            col_ref = col
            col_stream = col
        else:
            # Phase 2 uses lowercase with underscores: age, credit_amount, duration
            col_ref = col
            col_stream = col

        # Check if column exists in both dataframes
        if col_ref not in ref_filtered.columns or col_stream not in stream_filtered.columns:
            continue

        # Extract numeric values
        ref_vals = pd.to_numeric(ref_filtered[col_ref], errors="coerce").dropna()
        cur_vals = pd.to_numeric(stream_filtered[col_stream], errors="coerce").dropna()

        # Only calculate if we have enough samples
        if len(ref_vals) > 5 and len(cur_vals) > 5:
            try:
                psi = calculate_psi(ref_vals, cur_vals)
                ks_stat, ks_p = calculate_ks(ref_vals, cur_vals)
                results.append(
                    {
                        "Feature": col,
                        "PSI": round(psi, 4),
                        "KS Stat": round(ks_stat, 4),
                        "p-value": round(ks_p, 4),
                        "Status": drift_status(psi),
                        "Severity": drift_severity(psi),
                    }
                )
            except Exception:
                # Skip features that cause calculation errors
                continue

    return pd.DataFrame(results)


# DISPLAY METRIC ROW


def display_metric_row(metrics, phase_name):
    """Dynamically display metrics from JSON - no hardcoding"""
    if not metrics:
        st.info(f"📌 {phase_name} metrics not available. Run training first.")
        return

    # Get available metrics (exclude non-numeric and internal fields)
    exclude_keys = [
        "tn",
        "fp",
        "fn",
        "tp",
        "n_total",
        "n_pos",
        "lift_by_decile",
        "ks_threshold",
        "threshold",
    ]

    available_metrics = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and k not in exclude_keys and not k.startswith("_")
    }

    # Create columns dynamically
    num_metrics = len(available_metrics)
    cols = st.columns(min(num_metrics, 5))

    for idx, (key, value) in enumerate(list(available_metrics.items())[:5]):
        with cols[idx % 5]:
            # Format display name
            display_name = key.replace("_", " ").upper()
            # Format value
            if "auc" in key.lower() or "f1" in key.lower() or "f2" in key.lower():
                formatted_value = f"{value:.4f}"
            elif "lift" in key.lower():
                formatted_value = f"{value:.2f}x"
            elif "cost" in key.lower():
                formatted_value = f"${value:,.0f}"
            elif "pct" in key.lower():
                formatted_value = f"{value:.1f}%"
            elif isinstance(value, float) and value < 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)

            st.metric(display_name, formatted_value)


# DISPLAY ADDITIONAL METRICS


def display_additional_metrics(metrics, phase_name):
    """Display additional metrics like lift_by_decile"""
    if not metrics:
        return

    # Display lift by decile if available
    if "lift_by_decile" in metrics and metrics["lift_by_decile"]:
        st.subheader("📊 Lift by Decile")
        lift_data = metrics["lift_by_decile"]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"{d * 10}%" for d in range(1, 11)],
                y=lift_data,
                marker_color="#7C3AED",
                marker_line_width=1.5,
                text=[f"{x:.2f}x" for x in lift_data],
                textposition="outside",
            )
        )
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Random baseline",
            annotation_position="right",
        )
        fig.update_layout(
            title=f"{phase_name}: Lift by Decile",
            xaxis_title="Decile",
            yaxis_title="Lift",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display confusion matrix if available
    tn = metrics.get("tn")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    tp = metrics.get("tp")

    if all(v is not None for v in [tn, fp, fn, tp]):
        st.subheader("📋 Confusion Matrix")
        cm_data = [[tn, fp], [fn, tp]]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm_data,
                x=["Predicted Good", "Predicted Bad"],
                y=["Actual Good", "Actual Bad"],
                text=cm_data,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Blues",
                showscale=False,
            )
        )
        fig.update_layout(
            title=f"{phase_name}: Confusion Matrix",
            height=400,
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display threshold if available
    threshold = metrics.get("threshold")
    if threshold is not None:
        st.info(f"⚙️ Decision Threshold: **{threshold:.3f}**")


# MAIN CONTENT

st.title("📊 Production Credit Risk Monitoring")
st.markdown("**PSI + KS Drift Detection | Dual Pipeline | Industry Standard Metrics**")
st.markdown("---")


# PHASE 1

st.header("🏦 PHASE 1: German Credit (Baseline Model)")

ref_data_1 = load_reference_data_phase1()
stream_data_1 = load_streaming_data_phase1()
metrics_1 = load_metrics_phase1()

# Display Phase 1 metrics from JSON
display_metric_row(metrics_1, "Phase 1")

if ref_data_1 is not None and stream_data_1 is not None:
    numeric_cols_1 = ["Age", "Credit amount", "Duration", "Job"]
    drift_df_1 = run_drift_analysis(ref_data_1, stream_data_1, numeric_cols_1, phase="phase1")

    if len(drift_df_1) > 0:
        st.subheader("📋 Drift Analysis Table (PSI + KS)")
        st.caption("PSI < 0.1: Stable | PSI 0.1-0.25: Moderate | PSI > 0.25: Significant Drift")

        st.dataframe(
            drift_df_1,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn(width="medium"),
                "PSI": st.column_config.NumberColumn(width="small"),
                "KS Stat": st.column_config.NumberColumn(width="small"),
                "p-value": st.column_config.NumberColumn(width="small"),
                "Status": st.column_config.TextColumn(width="medium"),
                "Severity": st.column_config.TextColumn(width="small"),
            },
        )

        # PSI Bar Chart
        st.subheader("📈 PSI Drift Visualization")
        fig1 = go.Figure()
        colors = ["#2ecc71" if psi < 0.1 else "#f39c12" if psi < 0.25 else "#e74c3c" for psi in drift_df_1["PSI"]]
        fig1.add_trace(
            go.Bar(
                x=drift_df_1["Feature"],
                y=drift_df_1["PSI"],
                marker_color=colors,
                marker_line_width=1.5,
                marker_line_color="rgba(0,0,0,0.2)",
                text=drift_df_1["PSI"],
                textposition="outside",
                textfont=dict(size=11),
                name="PSI",
            )
        )
        fig1.add_hline(
            y=0.1,
            line_dash="dash",
            line_color="green",
            annotation_text="Warning (0.1)",
            annotation_position="right",
        )
        fig1.add_hline(
            y=0.25,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical (0.25)",
            annotation_position="right",
        )
        fig1.update_layout(
            title="Phase 1: PSI Drift by Feature",
            xaxis_title="Features",
            yaxis_title="PSI Score",
            height=500,
            showlegend=False,
            margin=dict(l=60, r=80, t=80, b=100),
            xaxis=dict(tickangle=-45, tickfont=dict(size=11), showgrid=False),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.1)"),
            plot_bgcolor="rgba(240,240,240,0.5)",
            hovermode="x unified",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Overall PSI Summary
        overall_psi = drift_df_1["PSI"].mean()
        st.subheader("📊 Overall Drift Assessment")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average PSI", f"{overall_psi:.4f}")
        with col2:
            drifting_features = len(drift_df_1[drift_df_1["PSI"] > 0.1])
            st.metric("Drifting Features", f"{drifting_features} / {len(drift_df_1)}")
        with col3:
            st.metric("Overall Status", drift_status(overall_psi))

        if overall_psi > 0.25:
            st.error(f"🚨 SIGNIFICANT DRIFT DETECTED (PSI={overall_psi:.3f}) → Retraining strongly recommended")
        elif overall_psi > 0.1:
            st.warning(f"⚠️ MODERATE DRIFT DETECTED (PSI={overall_psi:.3f}) → Monitor closely")
        else:
            st.success(f"✅ SYSTEM STABLE (PSI={overall_psi:.3f}) → No action needed")
    else:
        st.warning("⚠️ No valid numeric columns found for Phase 1 analysis")

    # Display additional metrics for Phase 1
    display_additional_metrics(metrics_1, "Phase 1")

else:
    st.info("📌 Phase 1 data not available. Check data files exist.")

st.markdown("---")


# PHASE 2

st.header("🏭 PHASE 2: LendingClub (Production Model)")

ref_data_2 = load_reference_data_phase2()
stream_data_2 = load_streaming_data_phase2()
metrics_2 = load_metrics_phase2()

# Display Phase 2 metrics from JSON
display_metric_row(metrics_2, "Phase 2")

if ref_data_2 is not None and stream_data_2 is not None:
    numeric_cols_2 = [
        "age",
        "credit_amount",
        "duration",
        "income",
        "emp_length",
        "dti",
        "int_rate",
    ]
    drift_df_2 = run_drift_analysis(ref_data_2, stream_data_2, numeric_cols_2, phase="phase2")

    if len(drift_df_2) > 0:
        st.subheader("📋 Drift Analysis Table (PSI + KS)")
        st.caption("PSI < 0.1: Stable | PSI 0.1-0.25: Moderate | PSI > 0.25: Significant Drift")

        st.dataframe(
            drift_df_2,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn(width="medium"),
                "PSI": st.column_config.NumberColumn(width="small"),
                "KS Stat": st.column_config.NumberColumn(width="small"),
                "p-value": st.column_config.NumberColumn(width="small"),
                "Status": st.column_config.TextColumn(width="medium"),
                "Severity": st.column_config.TextColumn(width="small"),
            },
        )

        # PSI Bar Chart
        st.subheader("📈 PSI Drift Visualization")
        fig2 = go.Figure()
        colors_2 = ["#2ecc71" if psi < 0.1 else "#f39c12" if psi < 0.25 else "#e74c3c" for psi in drift_df_2["PSI"]]
        fig2.add_trace(
            go.Bar(
                x=drift_df_2["Feature"],
                y=drift_df_2["PSI"],
                marker_color=colors_2,
                marker_line_width=1.5,
                marker_line_color="rgba(0,0,0,0.2)",
                text=drift_df_2["PSI"],
                textposition="outside",
                textfont=dict(size=11),
                name="PSI",
            )
        )
        fig2.add_hline(
            y=0.1,
            line_dash="dash",
            line_color="green",
            annotation_text="Warning (0.1)",
            annotation_position="right",
        )
        fig2.add_hline(
            y=0.25,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical (0.25)",
            annotation_position="right",
        )
        fig2.update_layout(
            title="Phase 2: PSI Drift by Feature",
            xaxis_title="Features",
            yaxis_title="PSI Score",
            height=500,
            showlegend=False,
            margin=dict(l=60, r=80, t=80, b=100),
            xaxis=dict(tickangle=-45, tickfont=dict(size=11), showgrid=False),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.1)"),
            plot_bgcolor="rgba(240,240,240,0.5)",
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Overall PSI Summary
        overall_psi_2 = drift_df_2["PSI"].mean()
        st.subheader("📊 Overall Drift Assessment")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average PSI", f"{overall_psi_2:.4f}")
        with col2:
            drifting_features_2 = len(drift_df_2[drift_df_2["PSI"] > 0.1])
            st.metric("Drifting Features", f"{drifting_features_2} / {len(drift_df_2)}")
        with col3:
            st.metric("Overall Status", drift_status(overall_psi_2))

        if overall_psi_2 > 0.25:
            st.error(f"🚨 SIGNIFICANT DRIFT DETECTED (PSI={overall_psi_2:.3f}) → Retraining strongly recommended")
        elif overall_psi_2 > 0.1:
            st.warning(f"⚠️ MODERATE DRIFT DETECTED (PSI={overall_psi_2:.3f}) → Monitor closely")
        else:
            st.success(f"✅ SYSTEM STABLE (PSI={overall_psi_2:.3f}) → No action needed")
    else:
        st.warning("⚠️ No valid numeric columns found for Phase 2 analysis")

    # Display additional metrics for Phase 2
    display_additional_metrics(metrics_2, "Phase 2")

else:
    st.info("📌 Phase 2 data not available. Check data files exist.")

st.markdown("---")


# DRIFT EXPLANATION SECTION

st.header("📖 The Idea: PSI and KS Drift Detection")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **What is PSI (Population Stability Index)?**

    PSI is the industry standard for measuring data drift. It compares the distribution of a feature between two time periods.

    **PSI Thresholds (Industry Standard):**
    - 🟢 **< 0.1** - Stable, no action needed
    - 🟡 **0.1 - 0.25** - Moderate drift, monitor closely
    - 🔴 **> 0.25** - Significant drift, retrain model

    **What is KS Test?**
    The Kolmogorov-Smirnov test measures if two distributions are statistically different. Lower p-value = more likely drift exists.
    """)

with col2:
    st.markdown("""
    **Interpretation Guide:**

    | Status | Action |
    |--------|--------|
    | 🟢 Stable | Continue monitoring |
    | 🟡 Moderate | Increase monitoring frequency |
    | 🔴 Significant | Schedule model retraining |

    **Common Drift Causes:**
    - Market/economic changes
    - Seasonal patterns
    - Data quality issues
    - Distribution shifts in population
    """)

st.markdown("---")


# FOOTER

st.caption(
    f"🔄 Production Monitoring Dashboard | PSI + KS Drift Detection | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
st.caption(
    "💡 PSI is the industry standard for drift detection - used by banks, fintechs, and financial institutions worldwide"
)
