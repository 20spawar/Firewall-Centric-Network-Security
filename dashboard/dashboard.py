
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_FILE = PROJECT_ROOT / "fusion" / "outputs" / "fusion_results.csv"
XAI_DIR = PROJECT_ROOT / "xai" / "outputs"

XAI_FILES = {
    "Fusion explanations": XAI_DIR / "fusion_explanations.csv",
    "Insider SHAP": XAI_DIR / "insider_shap_values.csv",
    "Insider LIME": XAI_DIR / "insider_lime_explanations.csv",
    "Anomaly SHAP": XAI_DIR / "anomaly_shap_values.csv",
    "Anomaly LIME": XAI_DIR / "anomaly_lime_explanations.csv",
    "Payload SHAP": XAI_DIR / "payload_shap_values.csv",
    "Payload LIME": XAI_DIR / "payload_lime_explanations.csv",
}

# =========================================================
# PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(
    page_title="FADE | Security Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    h1 {
        color: #00e5c3;
    }

    h2, h3 {
        color: #f0f3f6;
    }

    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ FADE — Firewall-Centric Network Security")

st.caption(
    "Unified threat monitoring across Payload Detection, "
    "Insider Threat Detection, and Network Anomaly Detection."
)

st.divider()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
@st.cache_data(show_spinner=False)
def read_optional_csv(path_string):
    """Read an optional CSV file without stopping the dashboard."""
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def rows_for_user(frame, user_value):
    """Return rows for a selected user when the file has a user column."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    user_col = next(
        (
            column
            for column in frame.columns
            if column.lower() in {"user", "user_id", "userid"}
        ),
        None,
    )

    if user_col is None:
        return pd.DataFrame()

    return frame[
        frame[user_col].astype(str).str.strip()
        == str(user_value).strip()
    ]


def show_shap_explanation(frame, module_name, user_value, file_label):
    """Display available numeric SHAP contributions for one user."""
    st.markdown("**SHAP Feature Contributions**")

    if frame is None:
        st.info(f"{file_label} file was not found.")
        return

    user_frame = rows_for_user(frame, user_value)

    if user_frame.empty:
        st.warning(
            f"No {file_label} explanation found for user {user_value}."
        )
        return

    user_columns = {"user", "user_id", "userid"}

    numeric_columns = [
        column
        for column in user_frame.columns
        if column.lower() not in user_columns
        and pd.to_numeric(
            user_frame[column], errors="coerce"
        ).notna().any()
    ]

    if not numeric_columns:
        st.dataframe(
            user_frame,
            use_container_width=True,
            hide_index=True,
        )
        return

    shap_values = []

    for column in numeric_columns:
        value = pd.to_numeric(
            user_frame.iloc[0][column],
            errors="coerce",
        )

        if pd.notna(value):
            shap_values.append(
                {
                    "Feature": column,
                    "SHAP Contribution": float(value),
                }
            )

    shap_display = pd.DataFrame(shap_values)

    if shap_display.empty:
        st.info("No numeric SHAP contributions are available.")
        return

    shap_display["Absolute Contribution"] = (
        shap_display["SHAP Contribution"].abs()
    )

    shap_display = (
        shap_display.sort_values(
            "Absolute Contribution",
            ascending=False,
        )
        .head(10)
    )

    fig = px.bar(
        shap_display.sort_values("SHAP Contribution"),
        x="SHAP Contribution",
        y="Feature",
        orientation="h",
        title=f"{module_name}: Top SHAP Contributions",
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        shap_display.drop(
            columns=["Absolute Contribution"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "SHAP contributions show how the available features "
        "influence the model output. They do not independently "
        "prove that an event is malicious."
    )


def show_lime_explanation(frame, user_value, file_label):
    """Display available LIME rules and contribution values."""
    st.markdown("**LIME Local Explanation**")

    if frame is None:
        st.info(f"{file_label} file was not found.")
        return

    user_frame = rows_for_user(frame, user_value)

    if user_frame.empty:
        st.warning(
            f"No {file_label} explanation found for user {user_value}."
        )
        return

    columns_lower = {
        column.lower(): column
        for column in user_frame.columns
    }

    rule_col = next(
        (
            columns_lower[key]
            for key in [
                "feature_rule",
                "feature",
                "rule",
                "feature_name",
            ]
            if key in columns_lower
        ),
        None,
    )

    contribution_col = next(
        (
            columns_lower[key]
            for key in [
                "contribution",
                "weight",
                "importance",
                "lime_weight",
            ]
            if key in columns_lower
        ),
        None,
    )

    rank_col = next(
        (
            columns_lower[key]
            for key in ["rank", "order"]
            if key in columns_lower
        ),
        None,
    )

    if rule_col and contribution_col:
        display_columns = [
            column
            for column in [
                rank_col,
                rule_col,
                contribution_col,
            ]
            if column
        ]

        lime_display = user_frame[display_columns].copy()

        lime_display[contribution_col] = pd.to_numeric(
            lime_display[contribution_col],
            errors="coerce",
        )

        st.dataframe(
            lime_display,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "LIME shows local feature rules and their contributions "
            "for the selected user."
        )
    else:
        # Preserve the original output if its column names differ.
        st.dataframe(
            user_frame,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# LOAD FUSION RESULTS
# =========================================================
if not FUSION_FILE.exists():
    st.error("Fusion results file was not found.")

    st.write("Expected location:")
    st.code(str(FUSION_FILE))

    st.info(
        "Run the Fusion module first, then refresh this dashboard."
    )

    st.stop()

try:
    df = pd.read_csv(FUSION_FILE)

except Exception as exc:
    st.error("Could not read fusion_results.csv.")
    st.exception(exc)
    st.stop()

if df.empty:
    st.warning("The Fusion results file is empty.")
    st.stop()

df.columns = [str(column).strip() for column in df.columns]

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================
if "user" not in df.columns:
    st.error(
        "The Fusion results must contain a 'user' column."
    )

    st.write("Columns found:", df.columns.tolist())
    st.stop()

# Accept payload_probability as an alternative only when
# payload_score is absent. Convert probabilities to 0–100.
if "payload_score" not in df.columns:
    if "payload_probability" in df.columns:
        payload_probability = pd.to_numeric(
            df["payload_probability"],
            errors="coerce",
        )

        if (
            not payload_probability.dropna().empty
            and payload_probability.dropna().between(0, 1).all()
        ):
            df["payload_score"] = payload_probability * 100
        else:
            df["payload_score"] = payload_probability

    else:
        st.error(
            "Payload score is missing from fusion_results.csv. "
            "Run the updated three-module Fusion pipeline first."
        )

        st.write("Columns found:", df.columns.tolist())
        st.stop()

score_columns = [
    "payload_score",
    "insider_score",
    "anomaly_score",
    "final_score",
]

required_columns = score_columns + ["decision"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("Required columns are missing from Fusion results.")

    st.write("Missing columns:", missing_columns)
    st.write("Columns found:", df.columns.tolist())

    st.stop()

# =========================================================
# CLEAN DATA
# =========================================================
for column in score_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df["user"] = df["user"].astype(str).str.strip()

df["decision"] = (
    df["decision"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df = df.dropna(subset=score_columns)

for column in score_columns:
    df[column] = df[column].clip(0, 100)

if df.empty:
    st.warning(
        "There are no valid numeric scores to display."
    )
    st.stop()

# Warn if payload records were mapped to users synthetically.
if "user_mapping_type" in df.columns:
    synthetic_mapping = df["user_mapping_type"].astype(str).str.contains(
        "SYNTHETIC",
        case=False,
        na=False,
    )

    if synthetic_mapping.any():
        st.warning(
            "Synthetic mapping notice: payload samples have been "
            "assigned to users for integration testing. These "
            "payload-to-user associations are not real observations "
            "and must not be interpreted as actual user behavior."
        )

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Dashboard Filters")

decisions = sorted(
    df["decision"].dropna().unique().tolist()
)

selected_decisions = st.sidebar.multiselect(
    "Risk category",
    options=decisions,
    default=decisions,
)

search_user = st.sidebar.text_input(
    "Search user ID"
).strip()

filtered = df[
    df["decision"].isin(selected_decisions)
].copy()

if search_user:
    filtered = filtered[
        filtered["user"].str.contains(
            search_user,
            case=False,
            na=False,
        )
    ]

if filtered.empty:
    st.warning(
        "No rows match the selected filters. "
        "Change the risk category or user search."
    )
    st.stop()

# =========================================================
# SECURITY OVERVIEW
# =========================================================
st.header("📊 Security Overview")

total_users = filtered["user"].nunique()

high = int(
    (filtered["decision"] == "HIGH").sum()
)

medium = int(
    (filtered["decision"] == "MEDIUM").sum()
)

low = int(
    (filtered["decision"] == "LOW").sum()
)

normal = int(
    (filtered["decision"] == "NORMAL").sum()
)

avg_score = filtered["final_score"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Users / Entities", f"{total_users:,}")
k2.metric("High Risk", f"{high:,}")
k3.metric("Medium Risk", f"{medium:,}")
k4.metric("Low Risk", f"{low:,}")
k5.metric("Normal", f"{normal:,}")
k6.metric("Average Final Score", f"{avg_score:.1f}/100")

st.divider()

# =========================================================
# INDIVIDUAL MODULE SCORE DISTRIBUTIONS
# =========================================================
st.header("🧩 Detection Module Scores")

score_labels = {
    "payload_score": "Payload Detection",
    "insider_score": "Insider Threat",
    "anomaly_score": "Network Anomaly",
}

module_columns = st.columns(3)

for column, (score_col, label) in zip(
    module_columns,
    score_labels.items(),
):
    with column:
        st.metric(
            f"{label} — Average",
            f"{filtered[score_col].mean():.2f}/100",
        )

        fig = px.histogram(
            filtered,
            x=score_col,
            nbins=20,
            title=f"{label} Score Distribution",
            labels={
                score_col: "Risk score (0–100)"
            },
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_range=[0, 100],
            yaxis_title="Number of records",
            margin=dict(
                l=15,
                r=15,
                t=55,
                b=15,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

# =========================================================
# THREE-MODULE COMPARISON
# =========================================================
st.divider()
st.header("⚖️ Compare the Three Modules")

comparison = filtered[
    [
        "user",
        "payload_score",
        "insider_score",
        "anomaly_score",
    ]
].melt(
    id_vars="user",
    value_vars=[
        "payload_score",
        "insider_score",
        "anomaly_score",
    ],
    var_name="Module",
    value_name="Score",
)

comparison["Module"] = comparison["Module"].map(
    {
        "payload_score": "Payload",
        "insider_score": "Insider",
        "anomaly_score": "Anomaly",
    }
)

max_points = 3000

if len(comparison) > max_points:
    comparison_plot = comparison.sample(
        max_points,
        random_state=42,
    )

    st.caption(
        "The comparison chart samples up to 3,000 points for "
        "readability. Tables continue to use the filtered data."
    )
else:
    comparison_plot = comparison

fig_compare = px.bar(
    comparison_plot,
    x="user",
    y="Score",
    color="Module",
    barmode="group",
    title="Per-User Module Risk Scores",
    hover_data=["Module", "Score"],
)

fig_compare.update_layout(
    template="plotly_dark",
    yaxis_range=[0, 100],
    xaxis_title="User",
    yaxis_title="Risk score (0–100)",
    xaxis_tickangle=-45,
)

st.plotly_chart(
    fig_compare,
    use_container_width=True,
)

# =========================================================
# SCORE RELATIONSHIP SCATTER PLOTS
# =========================================================
left, right = st.columns(2)

with left:
    fig_payload = px.scatter(
        filtered,
        x="payload_score",
        y="final_score",
        color="decision",
        hover_name="user",
        title="Payload Score vs Final Fused Score",
        labels={
            "payload_score": "Payload score",
            "final_score": "Final fused score",
        },
    )

    fig_payload.update_layout(
        template="plotly_dark"
    )

    fig_payload.update_xaxes(range=[0, 100])
    fig_payload.update_yaxes(range=[0, 100])

    st.plotly_chart(
        fig_payload,
        use_container_width=True,
    )

with right:
    fig_behavior = px.scatter(
        filtered,
        x="insider_score",
        y="anomaly_score",
        color="decision",
        hover_name="user",
        title="Insider Score vs Anomaly Score",
        labels={
            "insider_score": "Insider score",
            "anomaly_score": "Anomaly score",
        },
    )

    fig_behavior.update_layout(
        template="plotly_dark"
    )

    fig_behavior.update_xaxes(range=[0, 100])
    fig_behavior.update_yaxes(range=[0, 100])

    st.plotly_chart(
        fig_behavior,
        use_container_width=True,
    )

# =========================================================
# UNIFIED RISK ASSESSMENT
# =========================================================
st.divider()
st.header("🎯 Unified Risk Assessment")

left, right = st.columns(2)

with left:
    risk_order = [
        "NORMAL",
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    risk_counts = (
        filtered["decision"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Count")
    )

    fig_risk = px.pie(
        risk_counts,
        names="Risk Category",
        values="Count",
        hole=0.55,
        title="Risk Category Breakdown",
    )

    fig_risk.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )

with right:
    top_risk = (
        filtered.sort_values(
            "final_score",
            ascending=False,
        )
        .head(20)
    )

    fig_final = px.bar(
        top_risk.sort_values("final_score"),
        x="final_score",
        y="user",
        color="decision",
        orientation="h",
        title="Top 20 Highest-Risk Users / Entities",
        labels={
            "final_score": "Final fused score",
            "user": "User",
        },
    )

    fig_final.update_layout(
        template="plotly_dark",
        xaxis_range=[0, 100],
        yaxis_title="",
    )

    st.plotly_chart(
        fig_final,
        use_container_width=True,
    )

# =========================================================
# INSPECT A SELECTED USER
# =========================================================
st.divider()
st.header("🔎 Inspect a User")

user_options = sorted(
    filtered["user"].unique().tolist()
)

selected_user = st.selectbox(
    "Select a user",
    user_options,
    key="details_selected_user",
)

user_rows = filtered[
    filtered["user"] == selected_user
]

selected_row = user_rows.iloc[0]

a, b, c, d = st.columns(4)

a.metric(
    "Payload Score",
    f"{selected_row['payload_score']:.2f}/100",
)

b.metric(
    "Insider Score",
    f"{selected_row['insider_score']:.2f}/100",
)

c.metric(
    "Anomaly Score",
    f"{selected_row['anomaly_score']:.2f}/100",
)

d.metric(
    "Final Score",
    f"{selected_row['final_score']:.2f}/100",
)

st.write(
    f"**Risk decision:** {selected_row['decision']}"
)

detail_columns = [
    column
    for column in [
        "user",
        "payload_score",
        "insider_score",
        "anomaly_score",
        "final_score",
        "decision",
        "is_payload",
        "is_insider",
        "is_anomaly",
        "protocol",
        "payload_probability",
        "dominant_contributor",
    ]
    if column in user_rows.columns
]

st.dataframe(
    user_rows[detail_columns],
    use_container_width=True,
    hide_index=True,
)

# =========================================================
# HIGH-RISK RECORDS
# =========================================================
st.divider()
st.header("🚨 High-Risk Records")

high_risk_df = (
    filtered[
        filtered["decision"] == "HIGH"
    ]
    .sort_values(
        "final_score",
        ascending=False,
    )
)

if high_risk_df.empty:
    st.success(
        "No HIGH-risk records in the current filtered results."
    )
else:
    st.warning(
        f"{len(high_risk_df):,} HIGH-risk record(s) "
        "in the current filter."
    )

    st.dataframe(
        high_risk_df,
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# COMPLETE FUSION RESULTS + CSV DOWNLOAD
# =========================================================
st.divider()
st.header("📋 Fusion Results")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download filtered fusion results (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="fade_fusion_results_filtered.csv",
    mime="text/csv",
)

# =========================================================
# EXPLAINABLE AI (XAI)
# =========================================================
st.divider()
st.header("🔍 Explainable AI (XAI) Analysis")

st.write(
    "View explanations generated by the existing Fusion, Insider, "
    "and Anomaly XAI modules. Payload-specific SHAP/LIME explanations "
    "are displayed only if the corresponding output files exist."
)

xai_data = {
    name: read_optional_csv(str(path))
    for name, path in XAI_FILES.items()
}

existing_xai = {
    name: frame
    for name, frame in xai_data.items()
    if frame is not None
}

if not existing_xai:
    st.info(
        "No XAI output CSV files were found. The XAI section is available, "
        "but explanation tables will appear after the XAI pipeline generates "
        "the expected files in xai/outputs/."
    )
else:
    xai_user = st.selectbox(
        "Select user for XAI explanations",
        options=sorted(
            filtered["user"].astype(str).unique().tolist()
        ),
        key="xai_selected_user",
    )

    # -----------------------------------------------------
    # FUSION-LEVEL EXPLANATION
    # -----------------------------------------------------
    st.subheader("Fusion Explanation")

    fusion_xai = xai_data["Fusion explanations"]
    fusion_user_rows = rows_for_user(
        fusion_xai,
        xai_user,
    )

    if not fusion_user_rows.empty:
        st.dataframe(
            fusion_user_rows,
            use_container_width=True,
            hide_index=True,
        )
    elif fusion_xai is None:
        st.info(
            "Fusion explanation file not found: "
            "xai/outputs/fusion_explanations.csv"
        )
    else:
        st.warning(
            "No Fusion XAI explanation found for the selected user."
        )

    # -----------------------------------------------------
    # INSIDER, ANOMALY, AND OPTIONAL PAYLOAD XAI
    # -----------------------------------------------------
    xai_modules = [
        (
            "Insider Detection",
            "Insider SHAP",
            "Insider LIME",
        ),
        (
            "Anomaly Detection",
            "Anomaly SHAP",
            "Anomaly LIME",
        ),
        (
            "Payload Detection",
            "Payload SHAP",
            "Payload LIME",
        ),
    ]

    for module_name, shap_key, lime_key in xai_modules:
        with st.expander(
            f"{module_name} — SHAP and LIME",
            expanded=(module_name != "Payload Detection"),
        ):
            show_shap_explanation(
                xai_data[shap_key],
                module_name,
                xai_user,
                shap_key,
            )

            st.divider()

            show_lime_explanation(
                xai_data[lime_key],
                xai_user,
                lime_key,
            )

    st.caption(
        "XAI visualizations use explanation CSV files already generated "
        "by the project. The dashboard displays explanations; it does "
        "not generate SHAP or LIME explanations itself."
    )

# =========================================================
# FINAL NOTE
# =========================================================
st.divider()

st.caption(
    "Dashboard charts visualize the values saved in fusion_results.csv. "
    "They do not independently validate the detection models or any "
    "synthetic payload-to-user mapping."
)
