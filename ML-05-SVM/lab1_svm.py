"""
lab1_svm.py
-----------------------
LEB: SVM on a Dataset of Your Choice
  - Dataset: dogs_dataset.csv (Gender classification from Breed, Age, Weight, Color)
  - Explore and preprocess the dataset
  - Standardize the input features before training
  - Train SVM models using kernels: linear, polynomial, RBF
  - Evaluate each model using accuracy
  - Report the best kernel and a brief discussion
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
KERNELS = ["linear", "poly", "rbf"]


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

    # ---- Train SVM for each kernel, evaluate accuracy ----
    results = {}
    for kernel in KERNELS:
        pipe = Pipeline([
            ("prep", preprocess),
            ("svm", SVC(kernel=kernel, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[kernel] = {
            "accuracy": round(acc, 4),
            "y_pred": y_pred,
        }
        print(f"kernel={kernel}: accuracy = {acc:.4f}")

    best_kernel = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest kernel: {best_kernel} (accuracy = {results[best_kernel]['accuracy']})")

    # ---- Accuracy vs kernel plot ----
    plt.figure(figsize=(6, 5))
    kernels = list(results.keys())
    accs = [results[k]["accuracy"] for k in kernels]
    colors = ["#2b6cb0" if k != best_kernel else "red" for k in kernels]
    plt.bar(kernels, accs, color=colors)
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.xlabel("SVM kernel")
    plt.ylabel("Test Accuracy")
    plt.title("SVM Accuracy by Kernel (Gender Prediction)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_by_kernel.png"), dpi=130)
    plt.close()

    # ---- Confusion matrix for the best kernel ----
    cm = confusion_matrix(y_test, results[best_kernel]["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    plt.title(f"Confusion Matrix: SVM ({best_kernel} kernel)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best_kernel.png"), dpi=130)
    plt.close()

    # ---- Save results ----
    summary = {
        "kernels_tested": KERNELS,
        "accuracy_by_kernel": {k: results[k]["accuracy"] for k in KERNELS},
        "best_kernel": best_kernel,
        "best_accuracy": results[best_kernel]["accuracy"],
        "classes": list(le.classes_),
        "features_used": features,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(OUT_DIR, "lab1_svm_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
