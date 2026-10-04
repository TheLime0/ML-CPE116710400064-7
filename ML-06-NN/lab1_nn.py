"""
lab1_nn.py
-----------------------
LAB 6: Neural Network on a Dataset of Your Choice
  - Dataset: dogs_dataset.csv (Gender classification from Breed, Age, Weight, Color)
  - Explore and preprocess the dataset
  - Standardize the input features before training
  - Build a Neural Network (NN) model
  - Train the NN using different numbers of epochs
  - Compare different NN configurations (hidden layers / neurons)
  - Evaluate each model using accuracy
  - Report the best configuration and a brief discussion
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

EPOCH_OPTIONS = [10, 50, 100, 200, 400]
CONFIGURATIONS = {
    "1 layer, 8 neurons": (8,),
    "1 layer, 32 neurons": (32,),
    "2 layers, 32-16 neurons": (32, 16),
    "3 layers, 64-32-16 neurons": (64, 32, 16),
}
FIXED_EPOCHS_FOR_CONFIG_COMPARISON = 200


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

    # ---- Train NN for each number of epochs (fixed architecture) ----
    epoch_results = {}
    for n_epochs in EPOCH_OPTIONS:
        pipe = Pipeline([
            ("prep", preprocess),
            ("nn", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                max_iter=n_epochs,
                random_state=RANDOM_STATE,
            )),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        epoch_results[n_epochs] = round(acc, 4)
        print(f"epochs={n_epochs:>4}: accuracy = {acc:.4f}")

    # ---- Accuracy vs epochs plot ----
    plt.figure(figsize=(6, 5))
    epochs_list = list(epoch_results.keys())
    epoch_accs = list(epoch_results.values())
    plt.plot(epochs_list, epoch_accs, marker="o", color="#2b6cb0")
    for x, acc in zip(epochs_list, epoch_accs):
        plt.text(x, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.xlabel("Epochs (max_iter)")
    plt.ylabel("Test Accuracy")
    plt.title("NN Accuracy vs Number of Epochs (Gender Prediction)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_epochs.png"), dpi=130)
    plt.close()

    # ---- Train NN for each configuration (fixed epochs) ----
    config_results = {}
    config_preds = {}
    for name, layers in CONFIGURATIONS.items():
        pipe = Pipeline([
            ("prep", preprocess),
            ("nn", MLPClassifier(
                hidden_layer_sizes=layers,
                activation="relu",
                solver="adam",
                max_iter=FIXED_EPOCHS_FOR_CONFIG_COMPARISON,
                random_state=RANDOM_STATE,
            )),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        config_results[name] = round(acc, 4)
        config_preds[name] = y_pred
        print(f"config={name}: accuracy = {acc:.4f}")

    best_config = max(config_results, key=lambda k: config_results[k])
    print(f"\nBest configuration: {best_config} (accuracy = {config_results[best_config]})")

    # ---- Accuracy by configuration plot ----
    plt.figure(figsize=(7, 5))
    names = list(config_results.keys())
    accs = [config_results[n] for n in names]
    colors = ["#2b6cb0" if n != best_config else "red" for n in names]
    plt.bar(names, accs, color=colors)
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("NN configuration")
    plt.ylabel("Test Accuracy")
    plt.title("NN Accuracy by Configuration (Gender Prediction)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_by_configuration.png"), dpi=130)
    plt.close()

    # ---- Confusion matrix for the best configuration ----
    cm = confusion_matrix(y_test, config_preds[best_config])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    plt.title(f"Confusion Matrix: NN ({best_config})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best_config.png"), dpi=130)
    plt.close()

    # ---- Save results ----
    summary = {
        "epochs_tested": EPOCH_OPTIONS,
        "accuracy_by_epochs": epoch_results,
        "configurations_tested": {k: list(v) for k, v in CONFIGURATIONS.items()},
        "accuracy_by_configuration": config_results,
        "best_configuration": best_config,
        "best_accuracy": config_results[best_config],
        "classes": list(le.classes_),
        "features_used": features,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(OUT_DIR, "lab1_nn_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
