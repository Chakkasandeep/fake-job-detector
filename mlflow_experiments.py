"""
mlflow_experiments.py
=====================
This script trains three ML models (Naive Bayes, Logistic Regression, and XGBoost),
logs the experiments to MLflow, performs simple hyperparameter tuning on the best
model, and saves the final best model and TF-IDF vectorizer to disk.

To run:
    python mlflow_experiments.py
"""

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Natural Language Processing (NLP)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Quietly download necessary NLTK packages
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Scikit-Learn & XGBoost imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Evaluation and Sampling
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# MLflow Tracking
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """Cleans raw text by removing URLs, HTML tags, punctuation, and stopwords, then lemmatizing."""
    if pd.isnull(text) or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"<.*?>", "", text)                    # Remove HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)               # Remove HTML entities
    text = re.sub(r"[^a-z\s]", "", text)                 # Remove punctuation & numbers
    
    tokens = word_tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(clean_tokens)


# ─────────────────────────────────────────────────────────────
# 2. LOAD AND PREPARE DATASET
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Fake Job Posting Detection — Training & MLflow Tracker")
print("=" * 60)

CSV_PATH = "fake_job_postings.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {CSV_PATH}\n"
        "Please download the dataset from Kaggle and place it in this folder."
    )

# Read the CSV file
df = pd.read_csv(CSV_PATH)
print(f"\n[+] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"    - Real Jobs (0): {(df['fraudulent'] == 0).sum()}")
print(f"    - Fake Jobs (1): {(df['fraudulent'] == 1).sum()}")

# Combine relevant text features for a holistic representation
TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
df["combined_text"] = df[TEXT_COLS].fillna("").apply(
    lambda row: " ".join(row.values.astype(str)), axis=1
)

print("\n[+] Preprocessing text data (this may take a minute)...")
df["clean_text"] = df["combined_text"].apply(preprocess_text)

X = df["clean_text"]
y = df["fraudulent"]

# Stratified split to maintain class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text into TF-IDF Features
print("[+] Extracting TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Balance the highly imbalanced dataset using SMOTE
print("[+] Balancing class distribution with SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)
print(f"    - Balanced training set size: {X_train_bal.shape}")

# Save TF-IDF Vectorizer and test set (used by Streamlit app later)
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump((X_test_tfidf, y_test), "test_data.pkl")
print("\n[+] TF-IDF Vectorizer and test data successfully saved!")


# ─────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, name):
    """Evaluates model performance and plots a confusion matrix."""
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate probability scores if supported
    has_proba = hasattr(model, "predict_proba")
    y_prob = model.predict_proba(X_test_tfidf)[:, 1] if has_proba else None
    
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else 0.0,
    }
    
    # Generate and save a clean Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    
    cm_path = f"cm_{name.replace(' ', '_')}.png"
    fig.savefig(cm_path, dpi=100)
    plt.close(fig)
    
    return metrics, y_pred, cm_path


# ─────────────────────────────────────────────────────────────
# 4. TRAINING AND EXPERIMENT TRACKING WITH MLFLOW
# ─────────────────────────────────────────────────────────────
# Explicitly use SQLite database backend to avoid deprecated FileStore bugs and missing meta.yaml errors on Windows
mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "Fake-Job-Detection"
mlflow.set_experiment(EXPERIMENT_NAME)

# Keep ONLY the 3 requested models: Naive Bayes, Logistic Regression, and XGBoost
MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42
    ),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),
}

all_results = []
best_f1     = 0.0
best_model  = None
best_model_name = ""

print("\n" + "─" * 60)
print("  Training models and logging runs to MLflow...")
print("─" * 60)

for name, model in MODELS.items():
    print(f"\n▶ Running: {name}")
    
    # Train the model
    if name == "Naive Bayes":
        # MultinomialNB expects non-negative feature counts.
        # Since SMOTE might produce very tiny negative floats due to precision, we ensure positive values.
        X_nb = X_train_bal.copy()
        X_nb.data = np.abs(X_nb.data)
        model.fit(X_nb, y_train_bal)
    else:
        model.fit(X_train_bal, y_train_bal)
    
    # Evaluate
    metrics, y_pred, cm_path = evaluate_model(model, name)
    print(f"   Metrics: Accuracy={metrics['accuracy']} | Precision={metrics['precision']} | Recall={metrics['recall']} | F1={metrics['f1_score']} | ROC-AUC={metrics['roc_auc']}")
    
    # Log details to MLflow
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({
            "model_type": name,
            "dataset": "fake_job_postings",
            "vectorizer": "TF-IDF (10k, bigrams)",
            "balancing": "SMOTE"
        })
        
        # Log parameters
        params = {"model": name, "tfidf_max_features": 10000, "tfidf_ngram": "(1,2)", "smote": True}
        if hasattr(model, "get_params"):
            params.update({k: str(v) for k, v in model.get_params().items()})
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log confusion matrix plot
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
        
        # Log model artifact
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            pass
            
    all_results.append({"Model": name, **metrics})
    
    # Track the best model based on F1 Score
    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_model = model
        best_model_name = name

# ─────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING ON THE BEST MODEL (XGBOOST)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Tuning Hyperparameters for XGBoost (Best Model Candidate)...")
print("  [Optimized for Speed: Running sequentially to avoid CPU Thrashing]")
print("─" * 60)

# Simplified param grid to ensure super fast search
param_grid = {
    "n_estimators":  [100, 150],
    "max_depth":     [3, 5],
    "learning_rate": [0.1, 0.2],
    "subsample":     [0.8, 1.0],
}

xgb_base = XGBClassifier(
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, n_jobs=-1
)

# Run sequentially (n_jobs=1) to prevent thread/process collision on Windows,
# while allowing XGBoost to utilize multithreading internally. Reduced to 3 iterations.
search = RandomizedSearchCV(
    xgb_base, param_grid, n_iter=3, cv=3,
    scoring="f1", n_jobs=1, random_state=42, verbose=1
)
search.fit(X_train_bal, y_train_bal)

tuned_xgb = search.best_estimator_
metrics_tuned, _, cm_path_tuned = evaluate_model(tuned_xgb, "XGBoost Tuned")

print(f"\n[+] Best Hyperparameters found: {search.best_params_}")
print(f"    - Tuned F1 Score: {metrics_tuned['f1_score']} | ROC-AUC: {metrics_tuned['roc_auc']}")

# Log tuned run to MLflow
with mlflow.start_run(run_name="XGBoost Tuned"):
    mlflow.set_tags({
        "model_type": "XGBoost_Tuned",
        "tuning": "RandomizedSearchCV (3 iter, cv=3)",
        "best": "true"
    })
    mlflow.log_params(search.best_params_)
    mlflow.log_metrics(metrics_tuned)
    mlflow.log_artifact(cm_path_tuned, artifact_path="confusion_matrices")
    
    try:
        mlflow.sklearn.log_model(tuned_xgb, artifact_path="best_model")
    except Exception:
        pass

# Update best model if tuned XGBoost performs better
if metrics_tuned["f1_score"] > best_f1:
    best_f1 = metrics_tuned["f1_score"]
    best_model = tuned_xgb
    best_model_name = "XGBoost Tuned"

all_results.append({"Model": "XGBoost Tuned", **metrics_tuned})

# ─────────────────────────────────────────────────────────────
# 6. SAVE THE BEST MODEL
# ─────────────────────────────────────────────────────────────
joblib.dump(best_model, "best_model.pkl")
print(f"\n[+] Final Best Model Selected: {best_model_name} (F1 Score = {best_f1})")
print("    - Successfully saved to 'best_model.pkl'!")

# ─────────────────────────────────────────────────────────────
# 7. SUMMARY AND VISUALIZATION
# ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results).sort_values("f1_score", ascending=False)
print("\n" + "=" * 60)
print("  SUMMARY OF EXPERIMENT RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

# Draw and save comparison chart
metrics_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
x = np.arange(len(results_df))
width = 0.14
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

fig, ax = plt.subplots(figsize=(12, 6))
for i, (metric, color) in enumerate(zip(metrics_cols, colors)):
    ax.bar(x + i * width, results_df[metric], width, label=metric.replace("_", " ").title(),
           color=color, alpha=0.85, edgecolor="white")

ax.set_xlabel("Machine Learning Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Experiment Comparisons — Fake Job Detection", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
ax.legend(loc="lower right")
ax.set_ylim(0.6, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("all_experiments_comparison.png", dpi=120)
print("\n[+] Summary comparison chart saved to 'all_experiments_comparison.png'!")
print("\n[+] To view full details, start MLflow UI by running: mlflow ui")
