"""Train, compare, and save heart-disease classification models."""
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.data import load_dataset

RANDOM_STATE = 42

def build_preprocessor(columns):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("numeric", numeric, columns)])

def candidates(columns):
    return {
        "logistic_regression": Pipeline([("preprocess", build_preprocessor(columns)), ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "decision_tree": Pipeline([("preprocess", build_preprocessor(columns)), ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=RANDOM_STATE))]),
    }

def evaluate(model, x_test, y_test):
    predicted = model.predict(x_test)
    probability = np.asarray(model.predict_proba(x_test))[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
    }

def visualizations(model, x_test, y_test, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(x_test), cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix: Test Set")
    fig.tight_layout(); fig.savefig(report_dir / "confusion_matrix.png", dpi=180); plt.close(fig)
    probability = np.asarray(model.predict_proba(x_test))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#0f766e", linewidth=2.5, label=f"ROC curve (AUC = {roc_auc_score(y_test, probability):.3f})")
    ax.plot([0,1], [0,1], color="#64748b", linestyle="--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC Curve: Test Set")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(report_dir / "roc_curve.png", dpi=180); plt.close(fig)

def train(data_path="data/heart_disease_uci.csv", model_path="models/heart_disease_model.joblib", report_dir="reports"):
    frame = load_dataset(data_path).dropna(subset=["target"])
    x, y = frame.drop(columns=["target"]), frame["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, stratify=y, random_state=RANDOM_STATE)
    models, scores = candidates(x.columns.tolist()), {}
    for name, model in models.items():
        model.fit(x_train, y_train); scores[name] = evaluate(model, x_test, y_test)
    best = max(scores, key=lambda name: scores[name]["roc_auc"])
    joblib.dump(models[best], model_path)
    visualizations(models[best], x_test, y_test, Path(report_dir))
    result = {"dataset_rows": len(frame), "test_rows": len(x_test), "best_model": best, "metrics": scores}
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir, "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart_disease_uci.csv")
    parser.add_argument("--model", default="models/heart_disease_model.joblib")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model, args.reports), indent=2))
