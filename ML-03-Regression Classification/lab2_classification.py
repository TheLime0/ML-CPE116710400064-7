"""
lab2_classification.py
-----------------------
LAB 2: Classification
  - Preparing classification data (encode Gender as 0/1, encode categoricals)
  - Decision Boundary Visualization (on 2 PCA components of the encoded features)
  - Logistic Regression -> Gender Prediction
  - Confusion Matrix + Accuracy/Precision/Recall/F1/ROC/AUC
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "/home/claude/lab_project/data/dogs_dataset.csv"
OUT_DIR = "/home/claude/lab_project/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Age (Years)": "Age", "Weight (kg)": "Weight"})
    return df


def main():
    df = load_data()

    # ---- Preparing classification data ----
    le = LabelEncoder()
    y = le.fit_transform(df["Gender"])  # Female/Male -> 0/1
    features = ["Breed", "Age", "Weight", "Color"]
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Breed", "Color"]),
            ("num", StandardScaler(), ["Age", "Weight"]),
        ]
    )

    clf_pipeline = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    clf_pipeline.fit(X_train, y_train)

    y_pred = clf_pipeline.predict(X_test)
    y_proba = clf_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1_score": round(f1_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        "classes": list(le.classes_),
    }

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    plt.title("Confusion Matrix: Gender Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=130)
    plt.close()

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#2b6cb0", linewidth=2, label=f"AUC = {metrics['roc_auc']}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Gender Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=130)
    plt.close()

    # ---- Decision Boundary Visualization on 2 PCA components ----
    X_train_encoded = preprocess.fit_transform(X_train)
    if hasattr(X_train_encoded, "toarray"):
        X_train_encoded = X_train_encoded.toarray()
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_2d = pca2.fit_transform(X_train_encoded)

    boundary_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    boundary_clf.fit(X_train_2d, y_train)

    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = boundary_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    scatter = plt.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="coolwarm",
        edgecolor="k", s=18, alpha=0.8
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Logistic Regression Decision Boundary (2D PCA projection)")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, le.classes_, title="Gender")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "decision_boundary.png"), dpi=130)
    plt.close()

    metrics["decision_boundary_pca_explained_variance"] = [
        round(float(v), 3) for v in pca2.explained_variance_ratio_
    ]

    with open(os.path.join(OUT_DIR, "lab2_classification_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    np.savez(
        os.path.join(OUT_DIR, "lab2_predictions.npz"),
        y_test=y_test, y_pred=y_pred, y_proba=y_proba,
    )


if __name__ == "__main__":
    main()
