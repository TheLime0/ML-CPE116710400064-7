"""
lab1_cnn.py
-----------------------
LAB 7: CNN on a Dataset of Your Choice
  - Dataset: Digits (scikit-learn, 1797 images of handwritten digits 0-9, 8x8 pixels)
  - Split the dataset into training and testing sets
  - Standardize the input features before training
  - Build a CNN model
  - Train the CNN using different numbers of epochs
  - Compare different CNN configurations (convolutional layers / neurons)
  - Evaluate each model using accuracy
  - Show training/validation accuracy-loss and predictions
"""
import os
import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
tf.keras.utils.set_random_seed(RANDOM_STATE)

EPOCH_OPTIONS = [5, 10, 20, 50]
# (conv filters per layer, dense neurons)
CONFIGURATIONS = {
    "1 conv (16), dense 32": ([16], 32),
    "2 conv (16-32), dense 32": ([16, 32], 32),
    "2 conv (16-32), dense 64": ([16, 32], 64),
    "3 conv (16-32-64), dense 64": ([16, 32, 64], 64),
}
FIXED_EPOCHS_FOR_CONFIG_COMPARISON = 20
BATCH_SIZE = 32


def build_cnn(conv_filters, dense_units):
    model = keras.Sequential([layers.Input(shape=(8, 8, 1))])
    for i, f in enumerate(conv_filters):
        model.add(layers.Conv2D(f, (3, 3), padding="same", activation="relu"))
        if i < 2:  # 8x8 -> 4x4 -> 2x2 (avoid pooling below 2x2)
            model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class TestAccAtEpochs(keras.callbacks.Callback):
    """Record test accuracy when training reaches each epoch count in EPOCH_OPTIONS."""
    def __init__(self, X_test, y_test, checkpoints):
        super().__init__()
        self.X_test, self.y_test, self.checkpoints = X_test, y_test, set(checkpoints)
        self.results = {}

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 in self.checkpoints:
            _, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            self.results[epoch + 1] = round(float(acc), 4)


def main():
    digits = load_digits()
    X, y = digits.images.astype("float32"), digits.target
    print(f"Loaded {len(X)} images of shape {X.shape[1:]}, {len(np.unique(y))} classes")

    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, X, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Standardize (fit on train only) ----
    mean, std = X_train.mean(), X_train.std()
    X_train = ((X_train - mean) / std)[..., None]
    X_test = ((X_test - mean) / std)[..., None]

    # ---- Train for each number of epochs (fixed architecture = 2 conv, dense 32) ----
    # one run of max epochs; test accuracy is recorded at each checkpoint
    base_filters, base_dense = CONFIGURATIONS["2 conv (16-32), dense 32"]
    model = build_cnn(base_filters, base_dense)
    cb = TestAccAtEpochs(X_test, y_test, EPOCH_OPTIONS)
    model.fit(X_train, y_train, epochs=max(EPOCH_OPTIONS), batch_size=BATCH_SIZE,
              validation_split=0.1, callbacks=[cb], verbose=0)
    epoch_results = {e: cb.results[e] for e in EPOCH_OPTIONS}
    for e, a in epoch_results.items():
        print(f"epochs={e:>3}: accuracy = {a:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(EPOCH_OPTIONS, list(epoch_results.values()), marker="o", color="#2b6cb0")
    for x, acc in epoch_results.items():
        plt.text(x, acc + 0.005, f"{acc:.3f}", ha="center")
    plt.ylim(0.8, 1.02)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("CNN Accuracy vs Number of Epochs (Digits)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_epochs.png"), dpi=130)
    plt.close()

    # ---- Compare configurations (fixed epochs) ----
    config_results, histories, models = {}, {}, {}
    for name, (filters, dense) in CONFIGURATIONS.items():
        m = build_cnn(filters, dense)
        h = m.fit(X_train, y_train, epochs=FIXED_EPOCHS_FOR_CONFIG_COMPARISON,
                  batch_size=BATCH_SIZE, validation_split=0.1, verbose=0)
        acc = accuracy_score(y_test, m.predict(X_test, verbose=0).argmax(1))
        config_results[name] = round(float(acc), 4)
        histories[name], models[name] = h.history, m
        print(f"config={name}: accuracy = {acc:.4f} (params={m.count_params()})")

    best_config = max(config_results, key=lambda k: config_results[k])
    print(f"\nBest configuration: {best_config} (accuracy = {config_results[best_config]})")

    plt.figure(figsize=(8, 5))
    names = list(config_results.keys())
    accs = [config_results[n] for n in names]
    colors = ["#2b6cb0" if n != best_config else "red" for n in names]
    plt.bar(names, accs, color=colors)
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.003, f"{acc:.3f}", ha="center")
    plt.ylim(0.8, 1.02)
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("CNN configuration")
    plt.ylabel("Test Accuracy")
    plt.title("CNN Accuracy by Configuration (Digits)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_by_configuration.png"), dpi=130)
    plt.close()

    # ---- Training / validation accuracy & loss (best configuration) ----
    h = histories[best_config]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy"); ax[0].legend()
    ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="validation")
    ax[1].set(title="Loss", xlabel="Epoch", ylabel="Loss"); ax[1].legend()
    fig.suptitle(f"Training vs Validation ({best_config})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves_best_config.png"), dpi=130)
    plt.close()

    # ---- Predictions + confusion matrix (best configuration) ----
    y_pred = models[best_config].predict(X_test, verbose=0).argmax(1)
    fig, axes = plt.subplots(3, 6, figsize=(10, 5.5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(img_test[i], cmap="gray_r")
        ok = y_pred[i] == y_test[i]
        ax.set_title(f"pred {y_pred[i]} / true {y_test[i]}", color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    plt.suptitle(f"Predictions on test images ({best_config})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "predictions_best_config.png"), dpi=130)
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix: CNN ({best_config})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best_config.png"), dpi=130)
    plt.close()

    summary = {
        "epochs_tested": EPOCH_OPTIONS,
        "accuracy_by_epochs": epoch_results,
        "configurations_tested": {k: {"conv_filters": v[0], "dense_neurons": v[1]}
                                  for k, v in CONFIGURATIONS.items()},
        "accuracy_by_configuration": config_results,
        "best_configuration": best_config,
        "best_accuracy": config_results[best_config],
        "final_train_accuracy_best": round(h["accuracy"][-1], 4),
        "final_val_accuracy_best": round(h["val_accuracy"][-1], 4),
        "final_train_loss_best": round(h["loss"][-1], 4),
        "final_val_loss_best": round(h["val_loss"][-1], 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(OUT_DIR, "lab1_cnn_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
