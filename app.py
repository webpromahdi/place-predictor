# -*- coding: utf-8 -*-
"""
PlacePredictor — Full Streamlit UI
Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/Placement_Data_Full_Class.csv"
CHARTS_DIR   = Path("charts")

MODELS = {
    "Logistic Regression": "models/logistic_regression_pipeline.pkl",
    "Decision Tree":       "models/decision_tree_model.pkl",
    "Naive Bayes":         "placement_model.pkl",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PlacePredictor", page_icon="🎓", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#eef2ff 0%,#f8fafc 50%,#e0f2fe 100%); }

.main-title {
    text-align:center; font-size:52px; font-weight:900;
    background: linear-gradient(135deg,#1e3a8a,#2563eb);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:4px;
}
.subtitle { text-align:center; font-size:18px; color:#475569; margin-bottom:24px; }

.hero-card {
    background: linear-gradient(135deg,#1e3a8a,#2563eb);
    color:white; padding:22px 28px; border-radius:22px;
    box-shadow:0 15px 35px rgba(37,99,235,.25); margin-bottom:24px; text-align:center;
}
.hero-card h3 { margin:0; color:white; font-size:22px; }
.hero-card p  { margin:8px 0 0; color:#dbeafe; font-size:15px; }

.section-card {
    background:#fff; padding:22px; border-radius:20px;
    border:1px solid #e2e8f0; box-shadow:0 8px 24px rgba(15,23,42,.07); margin-bottom:18px;
}

.result-placed {
    background:linear-gradient(135deg,#15803d,#22c55e);
    padding:28px; border-radius:22px; color:white; text-align:center;
    box-shadow:0 12px 30px rgba(34,197,94,.3); margin:18px 0;
}
.result-notplaced {
    background:linear-gradient(135deg,#dc2626,#f97316);
    padding:28px; border-radius:22px; color:white; text-align:center;
    box-shadow:0 12px 30px rgba(249,115,22,.3); margin:18px 0;
}
.result-placed h2,.result-notplaced h2,
.result-placed h3,.result-notplaced h3,
.result-placed p,.result-notplaced p { color:white; margin:8px 0; }

.ai-card {
    background:#fff; border-left:7px solid #2563eb;
    padding:20px; border-radius:16px; margin:16px 0;
    box-shadow:0 8px 24px rgba(15,23,42,.07);
}
.ai-card h3 { color:#1e3a8a; margin-top:0; }

.positive-box {
    background:#ecfdf5; border-left:6px solid #22c55e;
    padding:12px 16px; border-radius:12px; margin:8px 0;
    color:#064e3b; font-weight:600;
}
.negative-box {
    background:#fff7ed; border-left:6px solid #f97316;
    padding:12px 16px; border-radius:12px; margin:8px 0;
    color:#7c2d12; font-weight:600;
}
.skill-box {
    background:#eff6ff; border:1px solid #bfdbfe;
    padding:12px 16px; border-radius:12px; margin:8px 0;
    color:#1e3a8a; font-weight:600;
}
.champion-badge {
    background:linear-gradient(135deg,#f59e0b,#fbbf24);
    color:#1c1917; padding:6px 14px; border-radius:20px;
    font-weight:800; font-size:13px; display:inline-block; margin-left:8px;
}

.stButton > button {
    background:linear-gradient(135deg,#1e40af,#2563eb);
    color:white; border:none; border-radius:14px;
    padding:14px 28px; font-size:17px; font-weight:800;
    width:100%; transition:.2s;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#1d4ed8,#3b82f6);
    transform:scale(1.01);
}
[data-testid="stMetric"] {
    background:white; padding:18px; border-radius:16px;
    border:1px solid #e2e8f0; box-shadow:0 6px 18px rgba(15,23,42,.07);
}
.stTabs [data-baseweb="tab-list"] { gap:10px; }
.stTabs [data-baseweb="tab"] {
    background:#fff; border-radius:12px; padding:12px 24px;
    font-weight:800; color:#1e3a8a; border:1px solid #dbeafe;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#1e40af,#2563eb); color:white;
}
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    return joblib.load(MODELS[name])

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["sl_no", "salary"], errors="ignore")
    df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})
    return df

@st.cache_data
def get_model_results():
    df = load_dataset()
    X, y = df.drop(columns=["status"]), df["status"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rows = []
    for name, path in MODELS.items():
        try:
            m = joblib.load(path)
            yp = m.predict(X_test)
            rows.append({
                "Model": name,
                "Accuracy":  round(accuracy_score(y_test, yp), 4),
                "Precision": round(precision_score(y_test, yp), 4),
                "Recall":    round(recall_score(y_test, yp), 4),
                "F1 Score":  round(f1_score(y_test, yp), 4),
            })
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)


def shap_waterfall_for_input(pipeline, input_df):
    """Generate SHAP waterfall chart for a single prediction row."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier   = pipeline.named_steps["classifier"]
        X_t = preprocessor.transform(input_df)

        num_cols = input_df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in input_df.columns if c not in num_cols]
        cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        feat_names = num_cols + cat_names

        explainer   = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_t)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]; bv = float(explainer.expected_value[1])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[0, :, 1]; bv = float(explainer.expected_value[1])
        else:
            sv = shap_values[0]; bv = float(np.atleast_1d(explainer.expected_value)[0])

        exp = shap.Explanation(values=sv, base_values=bv, data=X_t[0], feature_names=feat_names)
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.title("SHAP — Why this prediction?", pad=12, fontsize=13)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        return None


def simple_shap_bar(pipeline, input_df):
    """Simple green/red bar chart for SHAP impact."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier   = pipeline.named_steps["classifier"]
        X_t = preprocessor.transform(input_df)

        num_cols  = input_df.select_dtypes(include="number").columns.tolist()
        cat_cols  = [c for c in input_df.columns if c not in num_cols]
        cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        feat_names = num_cols + cat_names

        explainer   = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_t)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        top8    = sorted(zip(feat_names, sv), key=lambda x: abs(x[1]), reverse=True)[:8]
        names   = [t[0] for t in top8]
        values  = [t[1] for t in top8]
        colors  = ["#22c55e" if v > 0 else "#ef4444" for v in values]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(names, values, color=colors, height=0.55, edgecolor="white")
        for bar, val in zip(bars, values):
            xp = val + 0.005 if val >= 0 else val - 0.005
            ha = "left" if val >= 0 else "right"
            ax.text(xp, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Impact on Placement Prediction", fontsize=11)
        ax.set_title("Feature Impact — Simple View", fontsize=12, pad=10)
        ax.invert_yaxis()
        legend = [mpatches.Patch(facecolor="#22c55e", label="Helps (+)"),
                  mpatches.Patch(facecolor="#ef4444", label="Hurts (-)")]
        ax.legend(handles=legend, loc="lower right", fontsize=9)
        plt.tight_layout()
        return fig, list(zip(names, values))
    except Exception:
        return None, []


def rule_explanation(row, prediction, probability):
    reasons, skills = [], []
    if prediction == 0:
        if row["workex"] == "No":
            reasons.append("No work experience is a major reason for Not Placed.")
            skills.append("Gain internship or project work experience.")
        if row["etest_p"] < 60:
            reasons.append("Employability test score is low.")
            skills.append("Improve aptitude, reasoning and communication skills.")
        if row["degree_p"] < 60:
            reasons.append("Degree percentage is below average for placed students.")
            skills.append("Strengthen core subject knowledge.")
        if row["mba_p"] < 60:
            reasons.append("MBA percentage is low.")
            skills.append("Improve management knowledge and business communication.")
        if row["ssc_p"] < 60:
            reasons.append("SSC percentage is comparatively low.")
            skills.append("Build a strong portfolio to compensate.")
        if not reasons:
            reasons.append("Overall profile is weaker than placed students.")
            skills.append("Improve work experience, employability score, and interview preparation.")
        heading = "❌ Why Not Placed?"
        skill_h = "📌 Skills to Improve"
        summary = f"The model predicted Not Placed. Placement probability: {probability*100:.1f}%."
    else:
        if row["workex"] == "Yes":
            reasons.append("Work experience strongly supports placement.")
        if row["etest_p"] >= 65:
            reasons.append("Good employability test score.")
        if row["degree_p"] >= 65:
            reasons.append("Solid degree percentage.")
        if row["mba_p"] >= 65:
            reasons.append("MBA percentage is competitive.")
        if not reasons:
            reasons.append("Overall profile matches placed students.")
        skills = ["Build a strong CV and LinkedIn profile.",
                  "Improve interview confidence and HR preparation.",
                  "Add certifications relevant to your specialisation."]
        heading = "✅ Why Placed?"
        skill_h = "🚀 Further Improvements"
        summary = f"The model predicted Placed. Placement probability: {probability*100:.1f}%."
    return summary, heading, list(dict.fromkeys(reasons)), skill_h, list(dict.fromkeys(skills))


def build_form():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Student Information")
    c1, c2 = st.columns(2)
    with c1:
        gender   = st.selectbox("Gender", ["M", "F"])
        ssc_p    = st.number_input("SSC Percentage",    0.0, 100.0, 70.0, 0.5)
        ssc_b    = st.selectbox("SSC Board", ["Central", "Others"])
        hsc_p    = st.number_input("HSC Percentage",    0.0, 100.0, 70.0, 0.5)
        hsc_b    = st.selectbox("HSC Board", ["Central", "Others"])
        hsc_s    = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
    with c2:
        degree_p  = st.number_input("Degree Percentage",         0.0, 100.0, 65.0, 0.5)
        degree_t  = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
        workex    = st.selectbox("Work Experience", ["Yes", "No"])
        etest_p   = st.number_input("Employability Test %",      0.0, 100.0, 60.0, 0.5)
        specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])
        mba_p     = st.number_input("MBA Percentage",            0.0, 100.0, 60.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)
    return pd.DataFrame([{
        "gender": gender, "ssc_p": ssc_p, "ssc_b": ssc_b,
        "hsc_p": hsc_p, "hsc_b": hsc_b, "hsc_s": hsc_s,
        "degree_p": degree_p, "degree_t": degree_t, "workex": workex,
        "etest_p": etest_p, "specialisation": specialisation, "mba_p": mba_p,
    }])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    try:
        st.sidebar.image("assets/logo.png", width=170)
    except Exception:
        pass
    st.sidebar.title("PlacePredictor")
    st.sidebar.markdown("---")

    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        list(MODELS.keys()),
        index=0,
        help="Logistic Regression has the highest F1 score among available models."
    )
    show_shap = st.sidebar.checkbox(
        "Show SHAP Explanation (Decision Tree)",
        value=True,
        help="SHAP explanation uses the Decision Tree model regardless of prediction model."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Team Straw Hat · PlacePredictor · 2025")

    # Header
    st.markdown('<div class="main-title">🎓 PlacePredictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-Powered Student Placement Prediction System</div>',
        unsafe_allow_html=True,
    )
    try:
        st.image("assets/banner.png", use_container_width=True)
    except Exception:
        pass
    st.markdown("""
    <div class="hero-card">
        <h3>Predict · Explain · Improve</h3>
        <p>Enter student details to predict placement status, see AI explanation, and get actionable improvement tips.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Student Prediction", "📊 Model Comparison", "🔬 EDA & Charts"])

    # ── Tab 1: Prediction ──────────────────────────────────────────────────────
    with tab1:
        input_df = build_form()

        if st.button("🚀 Predict Placement"):
            model = load_model(model_choice)
            try:
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

            # Result card
            if pred == 1:
                st.markdown(f"""
                <div class="result-placed">
                    <h2>✅ Prediction: Placed</h2>
                    <h3>Placement Probability: {prob*100:.1f}%</h3>
                    <p>Model: {model_choice}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-notplaced">
                    <h2>❌ Prediction: Not Placed</h2>
                    <h3>Placement Probability: {prob*100:.1f}%</h3>
                    <p>Model: {model_choice}</p>
                </div>""", unsafe_allow_html=True)

            # Probability meter
            c1, c2, c3 = st.columns(3)
            c1.metric("Placement Probability", f"{prob*100:.1f}%")
            c2.metric("Prediction Model",      model_choice)
            c3.metric("Result",                "Placed ✅" if pred == 1 else "Not Placed ❌")

            # Probability bar
            fig_prob, ax = plt.subplots(figsize=(7, 1.2))
            color = "#22c55e" if pred == 1 else "#ef4444"
            ax.barh(["Placement Chance"], [prob * 100], color=color, height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.axvline(50, color="gray", lw=1, ls="--", alpha=0.5)
            ax.set_title("Placement Probability")
            plt.tight_layout()
            st.pyplot(fig_prob)
            plt.close()

            # SHAP section
            if show_shap:
                st.markdown("---")
                st.subheader("🔍 SHAP Explanation (Decision Tree)")
                st.caption("SHAP shows how each feature pushed the prediction toward Placed (+) or Not Placed (-).")

                dt_model = load_model("Decision Tree")
                fig_simple, impact = simple_shap_bar(dt_model, input_df)
                if fig_simple:
                    st.pyplot(fig_simple)
                    plt.close()

                    # Top factors text
                    helps  = [(f, v) for f, v in impact if v > 0][:2]
                    hurts  = [(f, v) for f, v in reversed(impact) if v < 0][:2]
                    if helps:
                        st.markdown("**📈 Biggest strengths:**")
                        for f, v in helps:
                            st.markdown(f'<div class="positive-box">✅ {f}  →  +{v:.3f}</div>', unsafe_allow_html=True)
                    if hurts:
                        st.markdown("**📉 Main weaknesses:**")
                        for f, v in hurts:
                            st.markdown(f'<div class="negative-box">❌ {f}  →  {v:.3f}</div>', unsafe_allow_html=True)

                # Detailed waterfall
                with st.expander("Show Detailed SHAP Waterfall Chart"):
                    fig_wf = shap_waterfall_for_input(dt_model, input_df)
                    if fig_wf:
                        st.pyplot(fig_wf)
                        plt.close()
                    else:
                        st.info("Waterfall chart not available for this input.")

            # Rule-based AI Explanation
            st.markdown("---")
            st.subheader("🤖 AI Explanation")
            row = input_df.iloc[0]
            summary, heading, reasons, skill_h, skills = rule_explanation(row, pred, prob)
            st.markdown(f'<div class="ai-card"><h3>🤖 AI Explanation</h3><p>{summary}</p></div>',
                        unsafe_allow_html=True)
            st.markdown(f"### {heading}")
            box = "positive-box" if pred == 1 else "negative-box"
            for r in reasons:
                st.markdown(f'<div class="{box}">• {r}</div>', unsafe_allow_html=True)
            st.markdown(f"### {skill_h}")
            for s in skills:
                st.markdown(f'<div class="skill-box">🎯 {s}</div>', unsafe_allow_html=True)

            st.markdown("---")
            with st.expander("View Input Summary"):
                st.dataframe(input_df, use_container_width=True)

    # ── Tab 2: Model Comparison ────────────────────────────────────────────────
    with tab2:
        st.header("📊 Model Comparison Dashboard")
        results = get_model_results()

        if not results.empty:
            champion = results.iloc[0]["Model"]
            st.markdown(f"**Champion Model:** {champion} "
                        f'<span class="champion-badge">🏆 Best F1</span>', unsafe_allow_html=True)
            st.dataframe(results.style.highlight_max(subset=["F1 Score", "Accuracy"], color="#d1fae5"),
                         use_container_width=True)

            # F1 bar chart
            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors = ["#2563eb" if m == champion else "#93c5fd" for m in results["Model"]]
            bars = ax.bar(results["Model"], results["F1 Score"], color=colors, width=0.5)
            for bar, val in zip(bars, results["F1 Score"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("F1 Score")
            ax.set_title("Model F1 Score Comparison", fontsize=13)
            ax.axhline(0.8, color="gray", ls="--", lw=1, alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption("Blue = champion model. Dashed line = 0.80 F1 threshold.")
        else:
            st.warning("No model results available.")

    # ── Tab 3: EDA & Charts ────────────────────────────────────────────────────
    with tab3:
        st.header("🔬 Exploratory Data Analysis")
        df = load_dataset()

        c1, c2 = st.columns(2)
        with c1:
            # Placement count
            fig, ax = plt.subplots(figsize=(5, 4))
            df["status"].value_counts().plot(kind="bar", ax=ax, color=["#2563eb", "#ef4444"])
            ax.set_xticklabels(["Placed", "Not Placed"], rotation=0)
            ax.set_title("Placement Status Count")
            ax.set_ylabel("Students")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption("Dataset: 148 Placed vs 67 Not Placed — class imbalance handled with SMOTE.")

        with c2:
            # Work experience
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df.assign(status=df["status"].map({1:"Placed",0:"Not Placed"})),
                          x="workex", hue="status", ax=ax, palette={"Placed":"#2563eb","Not Placed":"#ef4444"})
            ax.set_title("Work Experience vs Placement")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption("Work experience is a strong predictor of placement.")

        # Score boxplots
        st.subheader("Score Distribution by Placement Status")
        df_plot = df.assign(status=df["status"].map({1:"Placed", 0:"Not Placed"}))

        # Show our pre-generated boxplots chart if available
        boxplot_path = CHARTS_DIR / "decision_tree_boxplots.png"
        if boxplot_path.exists():
            st.image(str(boxplot_path), caption="SSC / HSC / Degree / MBA Score Distribution (by placement status)",
                     use_container_width=True)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, col, title in zip(axes.flatten(),
                ["ssc_p","hsc_p","degree_p","mba_p"],
                ["SSC Score","HSC Score","Degree Score","MBA Score"]):
                sns.boxplot(data=df_plot, x="status", y=col, ax=ax, palette={"Placed":"#2563eb","Not Placed":"#ef4444"})
                ax.set_title(title)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        heatmap_path = CHARTS_DIR / "correlation_heatmap.png"
        if heatmap_path.exists():
            st.image(str(heatmap_path), caption="Numerical feature correlations", use_container_width=True)
        else:
            num_df = df.select_dtypes(include=["float64","int64"])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # SHAP summary
        st.subheader("SHAP Feature Importance (Decision Tree)")
        shap_sum = CHARTS_DIR / "shap_summary.png"
        shap_sim = CHARTS_DIR / "shap_simple.png"
        if shap_sim.exists():
            st.image(str(shap_sim), caption="Simple SHAP feature impact", use_container_width=True)
        if shap_sum.exists():
            st.image(str(shap_sum), caption="Full SHAP summary — all test students", use_container_width=True)

        st.markdown("---")
        st.subheader("Dataset Preview")
        raw = pd.read_csv(DATASET_PATH)
        st.dataframe(raw.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
