import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FUSION_FILE = (
    PROJECT_ROOT
    / "fusion"
    / "outputs"
    / "fusion_results.csv"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Firewall Security Dashboard",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #00ffcc;
}

h2, h3 {
    color: #ffffff;
}

[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 15px;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #aaaaaa;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("🛡️ Firewall-Centric Network Security")

st.subheader(
    "Insider Threat & Anomaly Detection Dashboard"
)

st.markdown("---")


# =========================================================
# CHECK FUSION FILE
# =========================================================

if not FUSION_FILE.exists():

    st.error("Fusion results file not found.")

    st.write("Expected location:")

    st.code(str(FUSION_FILE))

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = pd.read_csv(FUSION_FILE)

except Exception as e:

    st.error("Unable to read the Fusion Results file.")

    st.code(str(e))

    st.stop()


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "insider_score",
    "is_insider",
    "anomaly_score",
    "is_anomaly",
    "final_score",
    "decision"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error("Some required columns are missing.")

    st.write(missing_columns)

    st.stop()


st.success("Fusion results loaded successfully!")


# =========================================================
# CONVERT SCORE COLUMNS
# =========================================================

df["insider_score"] = pd.to_numeric(
    df["insider_score"],
    errors="coerce"
)

df["anomaly_score"] = pd.to_numeric(
    df["anomaly_score"],
    errors="coerce"
)

df["final_score"] = pd.to_numeric(
    df["final_score"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "insider_score",
        "anomaly_score",
        "final_score"
    ]
)


# =========================================================
# SECURITY OVERVIEW
# =========================================================

st.header("📊 Security Overview")


total_users = len(df)

high_risk = len(
    df[df["final_score"] >= 80]
)

medium_risk = len(
    df[
        (df["final_score"] >= 60) &
        (df["final_score"] < 80)
    ]
)

low_risk = len(
    df[
        (df["final_score"] >= 30) &
        (df["final_score"] < 60)
    ]
)

normal_risk = len(
    df[df["final_score"] < 30]
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "👥 Total Users",
        total_users
    )


with col2:

    st.metric(
        "🔴 High Risk",
        high_risk
    )


with col3:

    st.metric(
        "🟠 Medium Risk",
        medium_risk
    )


with col4:

    st.metric(
        "🟡 Low Risk",
        low_risk
    )


with col5:

    st.metric(
        "🟢 Normal",
        normal_risk
    )


# =========================================================
# THREAT ANALYSIS
# =========================================================

st.markdown("---")

st.header("📈 Threat Analysis")


col1, col2 = st.columns(2)


# =========================================================
# INSIDER SCORE CHART
# =========================================================

with col1:

    st.subheader("👤 Insider Threat Score")

    insider_chart = px.bar(
        df,
        x=df.index,
        y="insider_score",
        labels={
            "index": "User",
            "insider_score": "Insider Score"
        },
        title="Insider Threat Score"
    )

    insider_chart.update_layout(
        yaxis_range=[0, 100],
        xaxis_title="User",
        yaxis_title="Score"
    )

    st.plotly_chart(
        insider_chart,
        width="stretch"
    )


# =========================================================
# ANOMALY SCORE CHART
# =========================================================

with col2:

    st.subheader("🚨 Anomaly Score")

    anomaly_chart = px.bar(
        df,
        x=df.index,
        y="anomaly_score",
        labels={
            "index": "User",
            "anomaly_score": "Anomaly Score"
        },
        title="Network Anomaly Score"
    )

    anomaly_chart.update_layout(
        yaxis_range=[0, 100],
        xaxis_title="User",
        yaxis_title="Score"
    )

    st.plotly_chart(
        anomaly_chart,
        width="stretch"
    )


# =========================================================
# INSIDER VS ANOMALY
# =========================================================

st.markdown("---")

st.header("⚖️ Insider vs Anomaly Comparison")


comparison_df = df[
    [
        "insider_score",
        "anomaly_score"
    ]
].copy()

comparison_df["User"] = comparison_df.index.astype(str)

comparison_df = comparison_df.melt(
    id_vars="User",
    value_vars=[
        "insider_score",
        "anomaly_score"
    ],
    var_name="Threat Type",
    value_name="Score"
)


comparison_chart = px.bar(
    comparison_df,
    x="User",
    y="Score",
    color="Threat Type",
    barmode="group",
    title="Insider Score vs Anomaly Score"
)

comparison_chart.update_layout(
    yaxis_range=[0, 100],
    xaxis_title="User",
    yaxis_title="Score"
)

st.plotly_chart(
    comparison_chart,
    width="stretch"
)


# =========================================================
# FINAL RISK SCORE
# =========================================================

st.markdown("---")

st.header("🎯 Final Risk Score")


final_chart = px.bar(
    df,
    x=df.index,
    y="final_score",
    title="Combined Risk Score",
    labels={
        "index": "User",
        "final_score": "Final Risk Score"
    }
)

final_chart.update_layout(
    yaxis_range=[0, 100],
    xaxis_title="User",
    yaxis_title="Final Score"
)

st.plotly_chart(
    final_chart,
    width="stretch"
)


# =========================================================
# HIGH RISK USERS
# =========================================================

st.markdown("---")

st.header("🚨 High Risk Users")


high_risk_df = df[
    df["final_score"] >= 80
].copy()


if len(high_risk_df) > 0:

    st.warning(
        str(len(high_risk_df))
        + " high-risk user(s) detected."
    )

    st.dataframe(
        high_risk_df,
        width="stretch"
    )

else:

    st.success(
        "No high-risk users detected."
    )


# =========================================================
# USER SEARCH
# =========================================================

st.markdown("---")

st.header("🔎 Search Fusion Results")


search_text = st.text_input(
    "Search for a user"
)


if search_text:

    search_result = df[
        df.astype(str)
        .apply(
            lambda row:
            row.str.contains(
                search_text,
                case=False,
                na=False
            ).any(),
            axis=1
        )
    ]

else:

    search_result = df


st.dataframe(
    search_result,
    width="stretch"
)


# =========================================================
# COMPLETE FUSION RESULTS
# =========================================================

st.markdown("---")

st.header("📋 Complete Fusion Results")

st.dataframe(
    df,
    width="stretch"
)