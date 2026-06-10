"""
PlacePredictor Streamlit App
Prediction model: Naive Bayes
Explanation: AI-style rule-based explanation for why Placed / Not Placed and what skills are needed.

Run:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

MODEL_PATH = "placement_model.pkl"
DATASET_PATH = "dataset/Placement_Data_Full_Class.csv"
RESULTS_PATH = "model_results.csv"


st.set_page_config(
    page_title="PlacePredictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #e0f2fe 100%);
        font-family: 'Segoe UI', sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        color: #1e3a8a;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #475569;
        margin-bottom: 28px;
    }

    .hero-card {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        padding: 22px 28px;
        border-radius: 22px;
        box-shadow: 0 15px 35px rgba(37, 99, 235, 0.25);
        margin-bottom: 26px;
        text-align: center;
    }

    .hero-card h3 {
        margin: 0;
        color: white;
        font-size: 23px;
    }

    .hero-card p {
        margin: 8px 0 0 0;
        color: #dbeafe;
        font-size: 15px;
    }

    .section-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }

    .result-placed {
        background: linear-gradient(135deg, #15803d, #22c55e);
        padding: 28px;
        border-radius: 22px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 30px rgba(34, 197, 94, 0.35);
        margin: 18px 0;
    }

    .result-notplaced {
        background: linear-gradient(135deg, #dc2626, #f97316);
        padding: 28px;
        border-radius: 22px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 30px rgba(249, 115, 22, 0.35);
        margin: 18px 0;
    }

    .result-placed h2, .result-notplaced h2,
    .result-placed h3, .result-notplaced h3,
    .result-placed p, .result-notplaced p {
        color: white;
        margin: 8px 0;
    }

    .ai-card {
        background: #ffffff;
        border-left: 7px solid #2563eb;
        padding: 22px;
        border-radius: 16px;
        margin: 18px 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }

    .ai-card h3 {
        color: #1e3a8a;
        margin-top: 0;
    }

    .positive-box {
        background: #ecfdf5;
        border-left: 6px solid #22c55e;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 9px 0;
        color: #064e3b;
        font-weight: 600;
    }

    .negative-box {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 9px 0;
        color: #7c2d12;
        font-weight: 600;
    }

    .skill-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 9px 0;
        color: #1e3a8a;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 14px 28px;
        font-size: 17px;
        font-weight: 800;
        width: 100%;
        transition: 0.25s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        color: white;
        transform: scale(1.01);
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 800;
        color: #1e3a8a;
        border: 1px solid #dbeafe;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        color: white;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


def draw_probability_chart(probability: float):
    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.barh(["Placement Chance"], [probability * 100])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Predicted Placement Probability")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)


def explain_prediction(input_data: pd.DataFrame, prediction: int, probability: float):
    """Return explanation and skill suggestions based on input features."""
    row = input_data.iloc[0]

    reasons = []
    skills_needed = []

    if prediction == 0:
        if row["workex"] == "No":
            reasons.append("No work experience is a major reason for the Not Placed prediction.")
            skills_needed.append("Gain internship, project work, freelancing, or part-time job experience.")

        if row["etest_p"] < 60:
            reasons.append("Employability test score is low, which reduces placement chance.")
            skills_needed.append("Improve aptitude, logical reasoning, problem-solving, and communication skills.")
        elif row["etest_p"] < 70:
            reasons.append("Employability test score is moderate but can be improved.")
            skills_needed.append("Practice employability tests, group discussion, and interview questions.")

        if row["degree_p"] < 60:
            reasons.append("Degree percentage is low compared with stronger placement profiles.")
            skills_needed.append("Strengthen core subject knowledge and academic performance.")

        if row["mba_p"] < 60:
            reasons.append("MBA percentage is low and may reduce employer confidence.")
            skills_needed.append("Improve management knowledge, presentation skill, and business communication.")

        if row["ssc_p"] < 60:
            reasons.append("SSC percentage is comparatively low.")
            skills_needed.append("Build a strong portfolio to compensate for weaker previous academic history.")

        if row["hsc_p"] < 60:
            reasons.append("HSC percentage is comparatively low.")
            skills_needed.append("Improve academic consistency and subject fundamentals.")

        if row["specialisation"] == "Mkt&HR":
            skills_needed.append("Develop HR analytics, negotiation, people-management, and communication skills.")
        else:
            skills_needed.append("Develop financial analysis, Excel, business analytics, and marketing strategy skills.")

        if not reasons:
            reasons.append("The overall profile is weaker than placed students in the dataset.")
            skills_needed.append("Improve work experience, employability score, interview preparation, and portfolio.")

        heading = "❌ Why Not Placed?"
        skill_heading = "📌 Skills Needed to Become Placed"
        summary = f"The model predicted Not Placed because some placement factors are weak. Placement probability is {probability * 100:.2f}%."

    else:
        if row["workex"] == "Yes":
            reasons.append("Work experience supports the Placed prediction.")
        if row["etest_p"] >= 60:
            reasons.append("Employability test score supports placement chance.")
        if row["degree_p"] >= 60:
            reasons.append("Degree percentage supports placement chance.")
        if row["mba_p"] >= 60:
            reasons.append("MBA percentage supports placement chance.")
        if row["ssc_p"] >= 60 and row["hsc_p"] >= 60:
            reasons.append("Previous academic performance is satisfactory.")

        if not reasons:
            reasons.append("The overall profile is similar to placed students in the dataset.")

        skills_needed.append("Improve interview confidence and HR round preparation.")
        skills_needed.append("Build a strong CV, LinkedIn profile, and project portfolio.")
        skills_needed.append("Add certification or practical project work related to specialization.")
        skills_needed.append("Improve communication, presentation, and teamwork skills.")

        heading = "✅ Why Placed?"
        skill_heading = "🚀 Further Skills to Improve Career Chance"
        summary = f"The model predicted Placed because the profile has enough positive placement indicators. Placement probability is {probability * 100:.2f}%."

    # Remove duplicate suggestions while keeping order
    skills_needed = list(dict.fromkeys(skills_needed))
    reasons = list(dict.fromkeys(reasons))

    return summary, heading, reasons, skill_heading, skills_needed


def render_explanation(input_data: pd.DataFrame, prediction: int, probability: float):
    summary, heading, reasons, skill_heading, skills_needed = explain_prediction(
        input_data, prediction, probability
    )

    st.markdown(
        f"""
        <div class="ai-card">
            <h3>🤖 AI Explanation</h3>
            <p>{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"### {heading}")
    box_class = "positive-box" if prediction == 1 else "negative-box"
    for reason in reasons:
        st.markdown(f'<div class="{box_class}">• {reason}</div>', unsafe_allow_html=True)

    st.markdown(f"### {skill_heading}")
    for skill in skills_needed:
        st.markdown(f'<div class="skill-box">🎯 {skill}</div>', unsafe_allow_html=True)


def build_input_form():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        ssc_p = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
        ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
        hsc_p = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
        hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
        hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])

    with col2:
        degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
        degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
        workex = st.selectbox("Work Experience", ["Yes", "No"])
        etest_p = st.number_input("Employability Test Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
        specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])
        mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.5)

    st.markdown('</div>', unsafe_allow_html=True)

    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "ssc_p": [ssc_p],
            "ssc_b": [ssc_b],
            "hsc_p": [hsc_p],
            "hsc_b": [hsc_b],
            "hsc_s": [hsc_s],
            "degree_p": [degree_p],
            "degree_t": [degree_t],
            "workex": [workex],
            "etest_p": [etest_p],
            "specialisation": [specialisation],
            "mba_p": [mba_p],
        }
    )

    return input_data


def main():
    st.sidebar.image("assets/logo.png", width=180)
    st.sidebar.title("PlacePredictor")
    st.sidebar.write("Naive Bayes Placement Prediction")
    st.markdown('<div class="main-title">🎓 PlacePredictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-powered Student Placement Prediction System using Naive Bayes</div>',
        unsafe_allow_html=True,
    )
    st.image("assets/banner.png", width="stretch")
    st.markdown(
        """
        <div class="hero-card">
            <h3>Predict placement status and get clear improvement guidance</h3>
            <p>The system predicts Placed / Not Placed, shows probability, explains the reason, and suggests skills needed for placement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    df = load_dataset()

    tab1, tab2 = st.tabs(["Student Prediction", "Examiner Dashboard"])

    with tab1:
        input_data = build_input_form()

        if st.button("Predict Placement"):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-placed">
                        <h2>✅ Prediction: Placed</h2>
                        <h3>Placement Probability: {probability * 100:.2f}%</h3>
                        <p>The student is likely to be placed.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-notplaced">
                        <h2>❌ Prediction: Not Placed</h2>
                        <h3>Placement Probability: {probability * 100:.2f}%</h3>
                        <p>The student is likely not to be placed.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.metric("Placement Probability", f"{probability * 100:.2f}%")
            draw_probability_chart(probability)
            render_explanation(input_data, prediction, probability)

            st.subheader("Input Summary")
            st.dataframe(input_data, width="stretch")

    with tab2:
        st.header("Examiner Dashboard")

        st.subheader("Model Comparison Table")
        try:
            results = pd.read_csv(RESULTS_PATH)
            st.dataframe(results, width="stretch")
            st.caption("All models were trained independently using the same preprocessing and evaluation method. The final selected model is Naive Bayes.")
        except Exception:
            st.warning("model_results.csv not found. Run train_model.py first.")

        st.subheader("Exploratory Data Analysis")
        try:
            col1, col2 = st.columns(2)

            with col1:
                st.write("Placement Status Count")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="status", ax=ax)
                st.pyplot(fig)
                st.caption("This chart shows the number of Placed and Not Placed students in the dataset. It helps identify class imbalance.")

            with col2:
                st.write("Work Experience vs Placement")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="workex", hue="status", ax=ax)
                st.pyplot(fig)
                st.caption("This chart compares placement status based on work experience. It helps show whether work experience affects placement chances.")

            st.write("Employability Test Score by Placement Status")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="status", y="etest_p", ax=ax)
            st.pyplot(fig)
            st.caption("This boxplot shows employability test score distribution for Placed and Not Placed students. Higher scores may support placement chances.")

            st.write("Degree Percentage by Placement Status")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="status", y="degree_p", ax=ax)
            st.pyplot(fig)
            st.caption("This boxplot compares degree percentage between Placed and Not Placed students. It helps understand the role of academic performance.")

            st.write("Correlation Heatmap")
            numeric_df = df.drop(columns=["sl_no", "salary"], errors="ignore").select_dtypes(include=["int64", "float64"])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            st.caption("This heatmap shows relationships between numerical features such as SSC, HSC, Degree, Employability Test, MBA percentage, and placement status.")
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), width="stretch")

        except Exception as error:
            st.error("Dataset loading or EDA error.")
            st.write(error)


if __name__ == "__main__":
    main()
