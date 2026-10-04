"""
lab1_cnn.py
-----------------------
LAB 7: CNN on a Dataset of Your Choice
  - Dataset: dogs_dataset.csv (Gender classification from Breed, Age, Weight, Color)
  - Split the dataset into training and testing sets
  - Standardize the input features before training
  - Build a CNN model (1D convolution over the feature vector)
  - Train the CNN using different numbers of epochs
  - Compare different CNN configurations (convolutional layers / filters / neurons)
  - Evaluate each model using accuracy
  - Show training/validation accuracy-loss and predictions
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = os.path.join(os.path.dirname(__file__), "dogs_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
BATCH_SIZE = 32

EPOCH_OPTIONS = [10, 50, 100, 200]
# name -> (filters per Conv1D layer, dense neurons)
CONFIGURATIONS = {
    "1 conv (16), dense 32": ([16], 32),
    "1 conv (32), dense 64": ([32], 64),
    "2 conv (16-32), dense 32": ([16, 32], 32),
    "2 conv (32-64), dense 64": ([32, 64], 64),
}
BASE_CONFIG = "2 conv (16-32), dense 32"
FIXED_EPOCHS_FOR_CONFIG_COMPARISON = 50


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Age (Years)": "Age", "Weight (kg)": "Weight"})
    return df


def build_cnn(n_features, conv_filters, dense_units):
    model = keras.Sequential([layers.Input(shape=(n_features, 1))])
    for f in conv_filters:
        model.add(layers.Conv1D(f, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(n_features, filters, dense, X_train, y_train, epochs):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = build_cnn(n_features, filters, dense)
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                     validation_split=0.1, verbose=0)
    return model, hist.history


def predict_labels(model, X):
    return (model.predict(X, verbose=0).ravel() >= 0.5).astype(int)


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # ---- Explore / preprocess ----
    le = LabelEncoder()
    y = le.fit_transform(df["Gender"])  # Female/Male -> 0/1
    features = ["Breed", "Age", "Weight", "Color"]
    X = df[features]

    # ---- Split ----
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Standardize input features (fit on train only) ----
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Breed", "Color"]),
            ("num", StandardScaler(), ["Age", "Weight"]),
        ]
    )
    X_train = preprocess.fit_transform(X_train_df).astype("float32")
    X_test = preprocess.transform(X_test_df).astype("float32")
    n_features = X_train.shape[1]
    # Conv1D expects (samples, steps, channels)
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    print(f"Feature vector length = {n_features}")

    # ---- Train CNN for each number of epochs (fixed architecture) ----
    base_filters, base_dense = CONFIGURATIONS[BASE_CONFIG]
    epoch_results = {}
    for n_epochs in EPOCH_OPTIONS:
        model, _ = train(n_features, base_filters, base_dense, X_train, y_train, n_epochs)
        acc = accuracy_score(y_test, predict_labels(model, X_test))
        epoch_results[n_epochs] = round(float(acc), 4)
        print(f"epochs={n_epochs:>4}: accuracy = {acc:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(list(epoch_results.keys()), list(epoch_results.values()), marker="o", color="#2b6cb0")
    for x, acc in epoch_results.items():
        plt.text(x, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("CNN Accuracy vs Number of Epochs (Gender Prediction)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_epochs.png"), dpi=130)
    plt.close()

    # ---- Train CNN for each configuration (fixed epochs) ----
    config_results, histories, models, config_params = {}, {}, {}, {}
    for name, (filters, dense) in CONFIGURATIONS.items():
        model, hist = train(n_features, filters, dense, X_train, y_train,
                            FIXED_EPOCHS_FOR_CONFIG_COMPARISON)
        acc = accuracy_score(y_test, predict_labels(model, X_test))
        config_results[name] = round(float(acc), 4)
        histories[name], models[name] = hist, model
        config_params[name] = int(model.count_params())
        print(f"config={name}: accuracy = {acc:.4f} (params={config_params[name]})")

    best_config = max(config_results, key=lambda k: config_results[k])
    print(f"\nBest configuration: {best_config} (accuracy = {config_results[best_config]})")

    plt.figure(figsize=(8, 5))
    names = list(config_results.keys())
    accs = [config_results[n] for n in names]
    colors = ["#2b6cb0" if n != best_config else "red" for n in names]
    plt.bar(names, accs, color=colors)
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("CNN configuration")
    plt.ylabel("Test Accuracy")
    plt.title("CNN Accuracy by Configuration (Gender Prediction)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_by_configuration.png"), dpi=130)
    plt.close()

    # ---- Training / validation accuracy & loss (best configuration) ----
    h = histories[best_config]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="validation")
    ax[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax[1].legend()
    fig.suptitle(f"Training vs Validation ({best_config})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves_best_config.png"), dpi=130)
    plt.close()

    # ---- Confusion matrix + predictions (best configuration) ----
    best_model = models[best_config]
    y_pred = predict_labels(best_model, X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    plt.title(f"Confusion Matrix: CNN ({best_config})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best_config.png"), dpi=130)
    plt.close()

    prob_male = best_model.predict(X_test, verbose=0).ravel()
    pred_df = X_test_df.reset_index(drop=True).copy()
    pred_df["Actual"] = le.inverse_transform(y_test)
    pred_df["Predicted"] = le.inverse_transform(y_pred)
    pred_df["Prob_Male"] = prob_male.round(4)
    pred_df.to_csv(os.path.join(OUT_DIR, "predictions_best_config.csv"), index=False)
    print(pred_df.head(10).to_string(index=False))

    # ---- Save results ----
    summary = {
        "epochs_tested": EPOCH_OPTIONS,
        "accuracy_by_epochs": epoch_results,
        "configurations_tested": {k: {"conv_filters": v[0], "dense_neurons": v[1]}
                                  for k, v in CONFIGURATIONS.items()},
        "parameters_by_configuration": config_params,
        "accuracy_by_configuration": config_results,
        "best_configuration": best_config,
        "best_accuracy": config_results[best_config],
        "final_train_accuracy_best": round(h["accuracy"][-1], 4),
        "final_val_accuracy_best": round(h["val_accuracy"][-1], 4),
        "final_train_loss_best": round(h["loss"][-1], 4),
        "final_val_loss_best": round(h["val_loss"][-1], 4),
        "classes": list(le.classes_),
        "features_used": features,
        "n_features_after_encoding": int(n_features),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(OUT_DIR, "lab1_cnn_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
