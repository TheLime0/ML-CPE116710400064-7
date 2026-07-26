"""
lab3_comparison.py
-------------------
LAB 3: Model Comparison
  - Simple vs Multiple Linear Regression
  - Training vs Testing Performance (overfitting check)
  - Regression vs Classification (task-level comparison)
  - Consolidated Model Performance Metrics table
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/claude/lab_project/outputs"

with open(os.path.join(OUT_DIR, "lab1_regression_results.json")) as f:
    reg = json.load(f)
with open(os.path.join(OUT_DIR, "lab2_classification_results.json")) as f:
    clf = json.load(f)


def main():
    # ---- Simple vs Multiple Linear Regression (test set) ----
    simple_test = reg["simple_linear_regression"]["test"]
    multi_test = reg["multiple_linear_regression"]["test"]

    metrics_names = ["MAE", "MSE", "RMSE", "R2"]
    simple_vals = [simple_test[m] for m in metrics_names]
    multi_vals = [multi_test[m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, simple_vals, width, label="Simple LR", color="#63b3ed")
    plt.bar(x + width / 2, multi_vals, width, label="Multiple LR", color="#2b6cb0")
    plt.xticks(x, metrics_names)
    plt.title("Simple vs Multiple Linear Regression (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_simple_vs_multiple.png"), dpi=130)
    plt.close()

    # ---- Training vs Testing performance (overfitting check) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, title in zip(
        axes,
        ["simple_linear_regression", "multiple_linear_regression"],
        ["Simple Linear Regression", "Multiple Linear Regression"],
    ):
        train_m = reg[key]["train"]
        test_m = reg[key]["test"]
        vals_train = [train_m[m] for m in metrics_names]
        vals_test = [test_m[m] for m in metrics_names]
        xx = np.arange(len(metrics_names))
        ax.bar(xx - width / 2, vals_train, width, label="Train", color="#9ae6b4")
        ax.bar(xx + width / 2, vals_test, width, label="Test", color="#f6ad55")
        ax.set_xticks(xx)
        ax.set_xticklabels(metrics_names)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Training vs Testing Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_train_vs_test.png"), dpi=130)
    plt.close()

    # ---- Regression vs Classification (conceptual + metric-family comparison) ----
    comparison_table = {
        "Regression (Age Prediction)": {
            "output_type": "Continuous value (years)",
            "example_metrics": {k: multi_test[k] for k in metrics_names},
            "algorithm": "Linear Regression",
        },
        "Classification (Gender Prediction)": {
            "output_type": "Discrete class (Female/Male)",
            "example_metrics": {
                "Accuracy": clf["accuracy"],
                "Precision": clf["precision"],
                "Recall": clf["recall"],
                "F1_score": clf["f1_score"],
                "ROC_AUC": clf["roc_auc"],
            },
            "algorithm": "Logistic Regression",
        },
    }

    # ---- Consolidated metrics summary table (saved as JSON + printed) ----
    summary = {
        "simple_linear_regression_test": simple_test,
        "multiple_linear_regression_test": multi_test,
        "logistic_regression_classification_test": {
            "accuracy": clf["accuracy"],
            "precision": clf["precision"],
            "recall": clf["recall"],
            "f1_score": clf["f1_score"],
            "roc_auc": clf["roc_auc"],
        },
        "regression_vs_classification": comparison_table,
    }

    with open(os.path.join(OUT_DIR, "lab3_comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
