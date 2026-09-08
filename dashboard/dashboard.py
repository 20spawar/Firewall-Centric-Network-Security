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

# ============================================================
# XAI EXPLANATIONS
# Added after the existing dashboard code
# ============================================================

st.markdown("---")
st.header("🔍 Explainable AI (XAI) Analysis")

st.write(
    "Select a user to view the explanations generated by the "
    "Insider Detection, Anomaly Detection, and Fusion modules."
)

# ------------------------------------------------------------
# XAI output paths
# ------------------------------------------------------------

XAI_DIR = PROJECT_ROOT / "xai" / "outputs"

XAI_FILES = {
    "fusion": XAI_DIR / "fusion_explanations.csv",
    "insider_shap": XAI_DIR / "insider_shap_values.csv",
    "insider_lime": XAI_DIR / "insider_lime_explanations.csv",
    "anomaly_shap": XAI_DIR / "anomaly_shap_values.csv",
    "anomaly_lime": XAI_DIR / "anomaly_lime_explanations.csv",
}

# ------------------------------------------------------------
# Check whether XAI files exist
# ------------------------------------------------------------

missing_xai_files = [
    name for name, path in XAI_FILES.items()
    if not path.exists()
]

if missing_xai_files:

    st.warning(
        "Some XAI output files are missing: "
        + ", ".join(missing_xai_files)
    )

    st.info(
        "Run the XAI scripts first to generate the explanation files."
    )

else:

    # --------------------------------------------------------
    # Load XAI outputs
    # --------------------------------------------------------

    fusion_xai_df = pd.read_csv(XAI_FILES["fusion"])

    insider_shap_df = pd.read_csv(
        XAI_FILES["insider_shap"]
    )

    insider_lime_df = pd.read_csv(
        XAI_FILES["insider_lime"]
    )

    anomaly_shap_df = pd.read_csv(
        XAI_FILES["anomaly_shap"]
    )

    anomaly_lime_df = pd.read_csv(
        XAI_FILES["anomaly_lime"]
    )

    # Convert user IDs to strings for reliable matching
    fusion_xai_df["user"] = fusion_xai_df["user"].astype(str)
    insider_shap_df["user"] = insider_shap_df["user"].astype(str)
    insider_lime_df["user"] = insider_lime_df["user"].astype(str)
    anomaly_shap_df["user"] = anomaly_shap_df["user"].astype(str)
    anomaly_lime_df["user"] = anomaly_lime_df["user"].astype(str)

    # --------------------------------------------------------
    # User selection
    # --------------------------------------------------------

    available_xai_users = sorted(
        fusion_xai_df["user"].unique()
    )

    selected_xai_user = st.selectbox(
        "Select User for XAI Explanation",
        available_xai_users,
        key="xai_user_selector"
    )

    # --------------------------------------------------------
    # Get selected user's data
    # --------------------------------------------------------

    fusion_user_data = fusion_xai_df[
        fusion_xai_df["user"] == selected_xai_user
    ]

    insider_shap_user = insider_shap_df[
        insider_shap_df["user"] == selected_xai_user
    ]

    insider_lime_user = insider_lime_df[
        insider_lime_df["user"] == selected_xai_user
    ]

    anomaly_shap_user = anomaly_shap_df[
        anomaly_shap_df["user"] == selected_xai_user
    ]

    anomaly_lime_user = anomaly_lime_df[
        anomaly_lime_df["user"] == selected_xai_user
    ]

    # --------------------------------------------------------
    # XAI Analysis
    # --------------------------------------------------------

    if not fusion_user_data.empty:

        fusion_row = fusion_user_data.iloc[0]

        st.subheader(
            f"XAI Explanation for User: {selected_xai_user}"
        )

        # ----------------------------------------------------
        # Risk Summary
        # ----------------------------------------------------

        st.markdown("### 📊 Risk Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Final Risk Score",
                f"{fusion_row['final_score']:.2f}"
            )

        with col2:
            st.metric(
                "Decision",
                str(fusion_row["decision"])
            )

        with col3:
            st.metric(
                "Insider Score",
                f"{fusion_row['insider_score']:.2f}"
            )

        with col4:
            st.metric(
                "Anomaly Score",
                f"{fusion_row['anomaly_score']:.2f}"
            )

        # ----------------------------------------------------
        # Fusion Explanation
        # ----------------------------------------------------

        st.markdown("### 🔗 Fusion Explanation")

        st.info(
            str(fusion_row["explanation"])
        )

        st.write(
            f"**Dominant Contributor:** "
            f"{fusion_row['dominant_contributor']}"
        )

        # ----------------------------------------------------
        # Fusion Contribution Chart
        # ----------------------------------------------------

        contribution_df = pd.DataFrame({
            "Detection Source": [
                "Insider Detection",
                "Anomaly Detection"
            ],
            "Contribution": [
                fusion_row["insider_contribution"],
                fusion_row["anomaly_contribution"]
            ]
        })

        fig_fusion = px.bar(
            contribution_df,
            x="Detection Source",
            y="Contribution",
            title="Contribution to Final Risk Score",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig_fusion,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Tabs for Insider and Anomaly XAI
        # ----------------------------------------------------

        tab1, tab2 = st.tabs([
            "🕵️ Insider Detection XAI",
            "🚨 Anomaly Detection XAI"
        ])

        # ====================================================
        # INSIDER XAI
        # ====================================================

        with tab1:

            st.subheader("Insider Detection Explanation")

            # ------------------------------------------------
            # Insider SHAP
            # ------------------------------------------------

            st.markdown("#### SHAP Feature Contributions")

            if not insider_shap_user.empty:

                shap_row = insider_shap_user.iloc[0]

                shap_values = []

                for column in insider_shap_df.columns:

                    if column == "user":
                        continue

                    try:
                        value = float(shap_row[column])

                        shap_values.append({
                            "Feature": column,
                            "SHAP Contribution": value,
                            "Absolute Contribution": abs(value)
                        })

                    except (ValueError, TypeError):
                        continue

                insider_shap_plot_df = pd.DataFrame(
                    shap_values
                )

                if not insider_shap_plot_df.empty:

                    insider_shap_plot_df = (
                        insider_shap_plot_df
                        .sort_values(
                            "Absolute Contribution",
                            ascending=False
                        )
                        .head(10)
                    )

                    fig_insider_shap = px.bar(
                        insider_shap_plot_df.sort_values(
                            "Absolute Contribution"
                        ),
                        x="SHAP Contribution",
                        y="Feature",
                        orientation="h",
                        title="Top 10 Insider Detection Features",
                        text_auto=".3f"
                    )

                    st.plotly_chart(
                        fig_insider_shap,
                        use_container_width=True
                    )

                    st.caption(
                        "Larger absolute SHAP values indicate "
                        "features with greater influence on the model output."
                    )

                    st.dataframe(
                        insider_shap_plot_df[
                            [
                                "Feature",
                                "SHAP Contribution"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.warning(
                    "No Insider SHAP explanation found for this user."
                )

            # ------------------------------------------------
            # Insider LIME
            # ------------------------------------------------

            st.markdown("#### LIME Explanation")

            if not insider_lime_user.empty:

                insider_lime_display = (
                    insider_lime_user
                    .sort_values("rank")
                    .head(10)
                    .copy()
                )

                st.dataframe(
                    insider_lime_display[
                        [
                            "rank",
                            "feature_rule",
                            "contribution"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "LIME shows the local feature rules that contributed "
                    "to the explanation for this user."
                )

            else:

                st.warning(
                    "No Insider LIME explanation found for this user."
                )

        # ====================================================
        # ANOMALY XAI
        # ====================================================

        with tab2:

            st.subheader("Anomaly Detection Explanation")

            # ------------------------------------------------
            # Anomaly SHAP
            # ------------------------------------------------

            st.markdown("#### SHAP Feature Contributions")

            if not anomaly_shap_user.empty:

                anomaly_shap_row = anomaly_shap_user.iloc[0]

                anomaly_shap_values = []

                for column in anomaly_shap_df.columns:

                    if column == "user":
                        continue

                    try:
                        value = float(
                            anomaly_shap_row[column]
                        )

                        anomaly_shap_values.append({
                            "Feature": column,
                            "SHAP Contribution": value,
                            "Absolute Contribution": abs(value)
                        })

                    except (ValueError, TypeError):
                        continue

                anomaly_shap_plot_df = pd.DataFrame(
                    anomaly_shap_values
                )

                if not anomaly_shap_plot_df.empty:

                    anomaly_shap_plot_df = (
                        anomaly_shap_plot_df
                        .sort_values(
                            "Absolute Contribution",
                            ascending=False
                        )
                        .head(10)
                    )

                    fig_anomaly_shap = px.bar(
                        anomaly_shap_plot_df.sort_values(
                            "Absolute Contribution"
                        ),
                        x="SHAP Contribution",
                        y="Feature",
                        orientation="h",
                        title="Top 10 Anomaly Detection Features",
                        text_auto=".3f"
                    )

                    st.plotly_chart(
                        fig_anomaly_shap,
                        use_container_width=True
                    )

                    st.caption(
                        "Larger absolute SHAP values indicate "
                        "features with greater influence on the model output."
                    )

                    st.dataframe(
                        anomaly_shap_plot_df[
                            [
                                "Feature",
                                "SHAP Contribution"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.warning(
                    "No Anomaly SHAP explanation found for this user."
                )

            # ------------------------------------------------
            # Anomaly LIME
            # ------------------------------------------------

            st.markdown("#### LIME Explanation")

            if not anomaly_lime_user.empty:

                anomaly_lime_display = (
                    anomaly_lime_user
                    .sort_values("rank")
                    .head(10)
                    .copy()
                )

                st.dataframe(
                    anomaly_lime_display[
                        [
                            "rank",
                            "feature_rule",
                            "contribution"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "LIME shows the local feature rules that contributed "
                    "to the explanation for this user."
                )

            else:

                st.warning(
                    "No Anomaly LIME explanation found for this user."
                )

    else:

        st.warning(
            "No Fusion XAI explanation found for the selected user."
        )