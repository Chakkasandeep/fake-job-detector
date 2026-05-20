"""
app.py  ─  Fake Job Posting Detector
======================================
Interactive Streamlit application that allows users to:
  - Paste any job description to get instant Real vs. Fake prediction.
  - View fraud probability gauge.
  - See word-level feature contributions (SHAP / feature importances).
  - Inspect highlighted suspicious words in the input text.
  - View the MLflow experiment results summary.

To run:
    streamlit run app.py
"""

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# SHAP for explainable AI
import shap

# MLflow tracking client
import mlflow
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PREMIUM CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stTextArea textarea { font-size: 14px; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 18px 24px; border-radius: 12px;
        color: white; text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .fake-card {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        padding: 20px 28px; border-radius: 14px;
        color: white; text-align: center; margin: 12px 0;
        box-shadow: 0 6px 20px rgba(220,38,38,0.35);
    }
    .real-card {
        background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
        padding: 20px 28px; border-radius: 14px;
        color: white; text-align: center; margin: 12px 0;
        box-shadow: 0 6px 20px rgba(22,163,74,0.35);
    }
    .result-title { font-size: 1.6rem; font-weight: 800; letter-spacing: 1px; }
    .result-sub   { font-size: 1.1rem; margin-top: 6px; opacity: 0.9; }
    .feature-positive { color: #dc2626; font-weight: 600; }
    .feature-negative { color: #16a34a; font-weight: 600; }
    .highlight-fake { background-color: rgba(239, 68, 68, 0.2); border-radius: 4px;
                      border: 1px solid rgba(239, 68, 68, 0.45); padding: 2px 5px; color: #fca5a5; font-weight: 600; }
    .highlight-real { background-color: rgba(34, 197, 94, 0.2); border-radius: 4px;
                      border: 1px solid rgba(34, 197, 94, 0.45); padding: 2px 5px; color: #86efac; font-weight: 600; }
    .section-title { font-size: 1.15rem; font-weight: 700;
                     color: #1e40af; margin: 16px 0 8px; }
    div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TEXT PREPROCESSING (MUST MATCH TRAINING EXACTLY)
# ─────────────────────────────────────────────────────────────
for pkg in ["stopwords", "punkt", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """Standardizes and cleans text to prepare it for vectorization."""
    if pd.isnull(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(clean_tokens)

# ─────────────────────────────────────────────────────────────
# CACHED ARTIFACT & SHAP LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Loads pre-trained best model and TF-IDF vectorizer."""
    model_path = "best_model.pkl"
    tfidf_path = "tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        return None, None, "Trained models not found. Please run 'python mlflow_experiments.py' first!"
        
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf, None


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model, _X_bg):
    """Pre-builds the appropriate SHAP explainer depending on model type."""
    try:
        # Works perfectly for tree-based models like XGBoost
        return shap.TreeExplainer(_model)
    except Exception:
        try:
            # Fallback explainer for linear models like Logistic Regression
            return shap.LinearExplainer(
                _model, _X_bg,
                feature_perturbation="correlation_dependent"
            )
        except Exception:
            return None


@st.cache_data(ttl=60, show_spinner=False)
def load_mlflow_runs():
    """Loads all completed experiment runs from local MLflow tracking database."""
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = MlflowClient()
        exp = client.get_experiment_by_name("Fake-Job-Detection")
        if exp is None:
            return pd.DataFrame()
            
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.f1_score DESC"]
        )
        
        records = []
        for run in runs:
            metrics = run.data.metrics
            records.append({
                "Run Name":  run.info.run_name or run.info.run_id[:8],
                "Status":    run.info.status,
                "Accuracy":  round(metrics.get("accuracy",  0), 4),
                "Precision": round(metrics.get("precision", 0), 4),
                "Recall":    round(metrics.get("recall",    0), 4),
                "F1 Score":  round(metrics.get("f1_score",  0), 4),
                "ROC-AUC":   round(metrics.get("roc_auc",   0), 4),
            })
        return pd.DataFrame(records)
    except Exception as e:
        # If MLflow tracking isn't running or empty, return empty df
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# PLOTTING AND RENDERING HELPERS
# ─────────────────────────────────────────────────────────────
def shap_bar_chart(contrib_vals, contrib_names, top_n=15):
    """Creates a clean horizontal bar chart displaying SHAP feature contributions for words present in text."""
    if len(contrib_vals) == 0:
        return None
        
    vals = contrib_vals[:top_n]
    names = contrib_names[:top_n]
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel("Feature Contribution (← Pushes Real | Pushes Fake →)", fontsize=10)
    ax.set_title("Which words influenced this prediction?", fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    fake_patch = mpatches.Patch(color="#dc2626", label="Pushes FAKE")
    real_patch = mpatches.Patch(color="#16a34a", label="Pushes REAL")
    ax.legend(handles=[fake_patch, real_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig


def highlight_words(text: str, fake_words: list, real_words: list) -> str:
    """Highlights suspicious terms in red and safe terms in green in the raw text block safely."""
    fake_set = {w.lower() for w in fake_words}
    real_set = {w.lower() for w in real_words}
    all_terms = list(fake_set.union(real_set))
    if not all_terms:
        return text
        
    # Sort by length descending to match multi-word phrases first
    all_terms.sort(key=len, reverse=True)
    pattern_str = r"\b(" + "|".join(re.escape(w) for w in all_terms) + r")\b"
    pattern = re.compile(pattern_str, re.IGNORECASE)
    
    def replace_match(match):
        matched_text = match.group(1)
        lower_matched = matched_text.lower()
        if lower_matched in fake_set:
            return f'<span class="highlight-fake">{matched_text}</span>'
        elif lower_matched in real_set:
            return f'<span class="highlight-real">{matched_text}</span>'
        return matched_text
        
    return pattern.sub(replace_match, text)


def prob_gauge(prob: float):
    """Generates a modern visual gauge bar showing the fraud percentage."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background track
    bar_h = 0.35
    bar_y = 0.3
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, bar_y), 0.96, bar_h,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#e5e7eb"
    ))
    
    # Filled progress
    fill_color = "#dc2626" if prob > 0.5 else "#16a34a"
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, bar_y), max(0.96 * prob, 0.01), bar_h,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=fill_color
    ))
    
    label = f"Fraud Risk: {prob*100:.1f}%"
    ax.text(0.5, 0.82, label, ha="center", va="center",
            fontsize=13, fontweight="bold", color="#111827")
    ax.text(0.02, 0.1, "0% (Safe)", ha="left", fontsize=9, color="#16a34a")
    ax.text(0.98, 0.1, "100% (Suspicious)", ha="right", fontsize=9, color="#dc2626")
    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────────────────────
# APP SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/search--v1.png", width=60)
    st.title("Fake Job Detector")
    st.caption("AI-powered fake job posting analysis and explanation tool.")

    st.divider()
    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Predict Job Posting", "📊 MLflow Experiments", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**UI Settings**")
    top_n_features = st.slider("SHAP features to show", 5, 25, 12)
    show_highlight  = st.checkbox("Highlight suspicious terms", value=True)
    show_gauge      = st.checkbox("Show probability bar", value=True)


# Load artifacts (Model & Vectorizer)
model, tfidf, load_error = load_artifacts()


# ═══════════════════════════════════════════════════════════════
# PAGE 1: PREDICT JOB POSTING
# ═══════════════════════════════════════════════════════════════
if page == "🔍 Predict Job Posting":
    st.markdown("## 🔍 Detect Fraudulent Job Postings")
    st.markdown(
        "Paste any job description below. The AI model will calculate a fraud risk score "
        "and explain **exactly which words** drove the decision."
    )

    if load_error:
        st.error(f"⚠️ {load_error}")
        st.code("python mlflow_experiments.py", language="bash")
        st.stop()

    # Initialize session state for text input value if not already present
    if "jd_text_val" not in st.session_state:
        st.session_state.jd_text_val = ""

    # Preloaded Examples
    col_s1, col_s2, _ = st.columns([1, 1, 3])
    use_fake = col_s1.button("📋 Load Fake Example")
    use_real = col_s2.button("📋 Load Real Example")

    FAKE_EXAMPLE = """URGENT HIRING! Work from home — earn $5,000 WEEKLY GUARANTEED!
No experience needed. No skills required. Apply immediately — limited positions!
Just send your personal details and bank info to claim this exclusive offer NOW.
100% legitimate. Easy money. Data entry work from home. $500 daily guaranteed income.
We are hiring urgently. Don't miss this once-in-a-lifetime opportunity!"""

    REAL_EXAMPLE = """Software Engineer — Python Backend (San Francisco, CA)
We are a Series B fintech startup looking for a passionate Python developer.
Requirements: 3+ years Python, REST APIs, PostgreSQL, Docker, AWS.
Responsibilities: Design and develop scalable backend microservices,
collaborate with cross-functional teams, write unit and integration tests.
Compensation: $130,000–$155,000/year + equity + health/dental/vision.
Apply through our careers page at company.com/careers."""

    if use_fake:
        st.session_state.jd_text_val = FAKE_EXAMPLE
    elif use_real:
        st.session_state.jd_text_val = REAL_EXAMPLE

    jd_text = st.text_area(
        "Paste Job Description Details",
        key="jd_text_val",
        height=200,
        placeholder="Paste job title, company information, requirements, or benefits text...",
    )

    predict_btn = st.button("🚀 Analyze Job Posting", type="primary", width="stretch")

    if predict_btn:
        if not jd_text.strip():
            st.warning("Please paste some job text first.")
        else:
            with st.spinner("Analyzing text..."):
                clean = preprocess_text(jd_text)
                vectorized = tfidf.transform([clean])
                prediction = model.predict(vectorized)[0]
                
                # Fetch probability risk
                has_proba = hasattr(model, "predict_proba")
                prob = model.predict_proba(vectorized)[0][1] if has_proba else 0.5
                feature_names = tfidf.get_feature_names_out()

                # Get the indices of words actually present in this job description
                present_indices = vectorized.nonzero()[1]
                
                # Compute SHAP values for the specific prediction
                explainer = get_shap_explainer(model, vectorized)
                shap_vals_1d = None

                if explainer is not None:
                    try:
                        sv = explainer.shap_values(vectorized)
                        if isinstance(sv, list):
                            shap_vals_1d = np.array(sv[1]).flatten()
                        else:
                            shap_vals_1d = np.array(sv).flatten()
                    except Exception:
                        pass

                # Fallback if SHAP fails or is not applicable
                if shap_vals_1d is None:
                    if hasattr(model, "coef_"):
                        shap_vals_1d = model.coef_.flatten() * vectorized.toarray().flatten()
                    elif hasattr(model, "feature_importances_"):
                        shap_vals_1d = model.feature_importances_ * vectorized.toarray().flatten()
                    else:
                        shap_vals_1d = np.zeros(len(feature_names))

                # Filter and sort by absolute contribution to keep only words present in the text
                if len(present_indices) > 0:
                    present_shap = shap_vals_1d[present_indices]
                    present_names = [feature_names[i] for i in present_indices]
                    sort_idx = np.argsort(np.abs(present_shap))[::-1]
                    contrib_vals = present_shap[sort_idx]
                    contrib_names = [present_names[i] for i in sort_idx]
                else:
                    contrib_vals = np.array([])
                    contrib_names = []

            # Prediction Display Card
            if prediction == 1:
                st.markdown(f"""
                <div class="fake-card">
                    <div class="result-title">🚨 FRAUDULENT JOB POSTING DETECTED</div>
                    <div class="result-sub">Estimated Fraud Risk: {prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="real-card">
                    <div class="result-title">✅ LOOKS LIKE A REAL JOB</div>
                    <div class="result-sub">Estimated Fraud Risk: {prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Risk Gauge
            if show_gauge and has_proba:
                st.pyplot(prob_gauge(prob), width="content")

            st.divider()

            # Explainability / Contributing Words
            col_shap, col_info = st.columns([3, 2])

            with col_shap:
                st.markdown('<div class="section-title">🧠 Explainable AI — Word Feature Impacts</div>', unsafe_allow_html=True)
                st.caption("Visualizing the words in the description that contributed most to this prediction.")

                if len(contrib_vals) > 0:
                    fig_shap = shap_bar_chart(contrib_vals, contrib_names, top_n=top_n_features)
                    if fig_shap is not None:
                        st.pyplot(fig_shap, width="stretch")
                else:
                    st.info("No matching vocabulary words found in the text to display in the chart.")

            with col_info:
                st.markdown('<div class="section-title">📋 suspicious Words Breakdown</div>', unsafe_allow_html=True)

                if len(contrib_vals) > 0:
                    fake_contribs = [(name, val) for name, val in zip(contrib_names, contrib_vals) if val > 0]
                    real_contribs = [(name, val) for name, val in zip(contrib_names, contrib_vals) if val < 0]

                    st.markdown("**Words signaling FAKE** 🔴")
                    if fake_contribs:
                        for name, val in fake_contribs[:12]:
                            bar = "█" * min(int(val * 80), 12)
                            st.markdown(
                                f'<span class="feature-positive">▲ {name}</span>'
                                f' `{val:+.4f}` {bar}',
                                unsafe_allow_html=True
                            )
                    else:
                        st.write("None of the words in the description signaled FAKE.")

                    st.markdown("**Words signaling REAL** 🟢")
                    if real_contribs:
                        for name, val in real_contribs[:12]:
                            bar = "█" * min(int(abs(val) * 80), 12)
                            st.markdown(
                                f'<span class="feature-negative">▼ {name}</span>'
                                f' `{val:+.4f}` {bar}',
                                unsafe_allow_html=True
                            )
                    else:
                        st.write("None of the words in the description signaled REAL.")
                else:
                    st.write("No vocabulary words available.")

                st.divider()
                st.markdown("**Analysis Summary**")
                st.metric("Fraud Risk Probability", f"{prob*100:.1f}%")
                st.metric("Prediction Output", "FAKE 🚨" if prediction == 1 else "REAL ✅")
                st.metric("Decision Confidence", 
                          "High" if abs(prob - 0.5) > 0.35 else
                          "Medium" if abs(prob - 0.5) > 0.15 else "Low")

            # Inline Text Highlighting
            if show_highlight and len(contrib_vals) > 0:
                st.divider()
                st.markdown('<div class="section-title">📝 Highlighted Job Text</div>', unsafe_allow_html=True)
                st.caption("Red highlights show terms indicating fraud risk, while green highlights show authentic terms indicating a real job.")

                fake_terms = [name for name, val in zip(contrib_names, contrib_vals) if val > 0]
                real_terms = [name for name, val in zip(contrib_names, contrib_vals) if val < 0]
                
                highlighted_text = highlight_words(jd_text, fake_terms, real_terms)
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-radius:10px;padding:20px;font-size:15px;'
                    f'line-height:1.8;white-space:pre-wrap;color:#f1f5f9;'
                    f'box-shadow:inset 0 2px 4px 0 rgba(0,0,0,0.06);">{highlighted_text}</div>',
                    unsafe_allow_html=True
                )


# ═══════════════════════════════════════════════════════════════
# PAGE 2: MLFLOW EXPERIMENT DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "📊 MLflow Experiments":
    st.markdown("## 📊 MLflow Experiment Dashboard")
    st.caption("All training runs logged on your system — sorted by F1-Score (best first)")

    runs_df = load_mlflow_runs()

    if runs_df.empty or "Error" in runs_df.columns:
        st.warning(
            "No active training runs logged in MLflow. Run 'python mlflow_experiments.py' first!"
        )
        st.code("python mlflow_experiments.py", language="bash")
    else:
        # Summary metrics from best run
        best_run = runs_df.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best_run['F1 Score']:.3f}</div>
            <div class="metric-label">Best F1 Score</div></div>""",
            unsafe_allow_html=True)
        col2.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best_run['Accuracy']:.3f}</div>
            <div class="metric-label">Best Accuracy</div></div>""",
            unsafe_allow_html=True)
        col3.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best_run['ROC-AUC']:.3f}</div>
            <div class="metric-label">Best ROC-AUC</div></div>""",
            unsafe_allow_html=True)
        col4.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(runs_df)}</div>
            <div class="metric-label">Total Logs Found</div></div>""",
            unsafe_allow_html=True)

        st.divider()

        # Detailed data table
        st.markdown("### Experiment Metrics Summary Table")
        st.dataframe(
            runs_df.style
            .highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                color="#bbf7d0"
            )
            .format({
                "Accuracy":  "{:.4f}",
                "Precision": "{:.4f}",
                "Recall":    "{:.4f}",
                "F1 Score":  "{:.4f}",
                "ROC-AUC":   "{:.4f}",
            }),
            width="stretch",
            height=250,
        )

        # Bar chart comparison
        st.markdown("### Model Comparison Chart")
        metric_choice = st.selectbox(
            "Select Metric to compare:",
            ["F1 Score", "Accuracy", "Precision", "Recall", "ROC-AUC"],
            index=0
        )
        chart_data = runs_df.set_index("Run Name")[metric_choice].sort_values(ascending=False)
        colors = ["#dc2626" if idx == 0 else "#2563eb" for idx in range(len(chart_data))]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(chart_data.index, chart_data.values, color=colors, edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylabel(metric_choice)
        ax.set_title(f"Comparison of {metric_choice} Across Models", fontweight="bold")
        ax.set_ylim(max(0, chart_data.min() - 0.05), min(1.05, chart_data.max() + 0.06))
        plt.xticks(rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")

        st.info(
            "🔗 Want to drill down further? Run 'mlflow ui' in your project directory and head to http://localhost:5000"
        )


# ═══════════════════════════════════════════════════════════════
# PAGE 3: ABOUT / DOCUMENTATION
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
### 🎯 Purpose
- Detects **fake and fraudulent job postings** before job seekers fall victim to scams.
- Trains and compares 3 prominent machine learning classifiers: **Logistic Regression**, **Naive Bayes**, and **XGBoost**.
- Uses **SHAP (SHapley Additive exPlanations)** to ensure AI transparency.
- Tracks hyperparameters, files, and metrics dynamically using **MLflow**.

### 📦 Dataset Info
**Real or Fake — Fake Job Posting Prediction**
- Comprises roughly 18,000 detailed job postings (~800 fake listings).
- Shared by the University of the Aegean.
- Kaggle Dataset: `shivamb/real-or-fake-fake-jobposting-prediction`
        """)

    with col_b:
        st.markdown("""
### 🛠️ Technology Stack

| Stage | Tools Used |
|---|---|
| Language | Python 3.9+ |
| Text Processing | NLTK (Tokenizer, Lemmatizer, Stopwords) |
| Representation | TF-IDF (10,000 maximum features with bigrams) |
| Balance Handling | SMOTE (Synthetic Minority Over-sampling) |
| Classifiers | Scikit-learn, XGBoost |
| AI Explainability | SHAP |
| Tracking Backend | MLflow |
| Visual Front-end | Streamlit |

### 🚀 Commands to Run
```bash
# 1. Train the models and save the best classifier
python mlflow_experiments.py

# 2. Start the Streamlit interactive dashboard
streamlit run app.py

# 3. View the MLflow Experiment Runs UI (optional)
mlflow ui
```
        """)

    st.divider()
    st.markdown("### 🔍 Workflow Behind Predictions")
    st.markdown("""
1. **Text Cleansing**: Text fields (title, profile, requirements, description, benefits) are merged and processed to keep only meaningful lowercase, lemmatized root words.
2. **Feature Extraction**: TF-IDF transforms the words into a sparse matrix representing 10,000 most predictive unigrams and bigrams.
3. **Classification**: The best-performing model (usually XGBoost or tuned XGBoost) predicts a fraud probability risk.
4. **SHAP Decomposition**: Computes the exact weight contribution of each word toward the final classification choice.
5. **UI Rendering**: Pushes word-level labels with red markers (suggests scam) or green markers (suggests legitimate) to help verify job posting details manually.
    """)
