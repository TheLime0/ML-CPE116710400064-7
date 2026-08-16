"""
lab1_knn.py
-----------------------
LEB 1: KNN on a Dataset of Your Choice
  - Dataset: dogs_dataset.csv (Gender classification from Breed, Age, Weight, Color)
  - Explore and preprocess the dataset
  - Standardize the input features before training
  - Train KNN models using k = 3, 5, 7
  - Evaluate each model using accuracy
  - Report the best k value and a brief discussion
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "dogs_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
K_VALUES = [3, 5, 7]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Age (Years)": "Age", "Weight (kg)": "Weight"})
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(df.describe(include="all").T[["count", "unique", "top", "freq"]].fillna(""))

    # ---- Explore / preprocess ----
    le = LabelEncoder()
    y = le.fit_transform(df["Gender"])  # Female/Male -> 0/1
    features = ["Breed", "Age", "Weight", "Color"]
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Standardize input features (numeric scaled, categorical one-hot) ----
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Breed", "Color"]),
            ("num", StandardScaler(), ["Age", "Weight"]),
        ]
    )

    # ---- Train KNN for each k, evaluate accuracy ----
    results = {}
    for k in K_VALUES:
        pipe = Pipeline([
            ("prep", preprocess),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[k] = {
            "accuracy": round(acc, 4),
            "y_pred": y_pred,
        }
        print(f"k={k}: accuracy = {acc:.4f}")

    best_k = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest k value: {best_k} (accuracy = {results[best_k]['accuracy']})")

    # ---- Accuracy vs k plot ----
    plt.figure(figsize=(6, 5))
    ks = list(results.keys())
    accs = [results[k]["accuracy"] for k in ks]
    plt.plot(ks, accs, marker="o", color="#2b6cb0", linewidth=2)
    plt.scatter([best_k], [results[best_k]["accuracy"]], color="red", zorder=5,
                label=f"best k = {best_k}")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Test Accuracy")
    plt.title("KNN Accuracy vs k (Gender Prediction)")
    plt.xticks(ks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_k.png"), dpi=130)
    plt.close()

    # ---- Confusion matrix for the best k ----
    cm = confusion_matrix(y_test, results[best_k]["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    plt.title(f"Confusion Matrix: KNN (k={best_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best_k.png"), dpi=130)
    plt.close()

    # ---- Save results ----
    summary = {
        "k_values_tested": K_VALUES,
        "accuracy_by_k": {str(k): results[k]["accuracy"] for k in K_VALUES},
        "best_k": best_k,
        "best_accuracy": results[best_k]["accuracy"],
        "classes": list(le.classes_),
        "features_used": features,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(OUT_DIR, "lab1_knn_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
