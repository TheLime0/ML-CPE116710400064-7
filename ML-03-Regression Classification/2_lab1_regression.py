"""
lab1_regression.py
-------------------
LAB 1: Regression
  - Simple Linear Regression   (Weight -> Age)
  - Multiple Linear Regression (Breed + Weight + Color + Gender -> Age)
  - Age Prediction evaluation (MAE, MSE, RMSE, R^2) on train & test sets

Data: data/dogs_dataset.csv (real, labelled tabular data)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def evaluate(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "MSE": round(mean_squared_error(y_true, y_pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(r2_score(y_true, y_pred), 3),
    }


def main():
    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    results = {}

    # ---------- Simple Linear Regression: Weight -> Age ----------
    X_train_s = train_df[["Weight"]].values
    X_test_s = test_df[["Weight"]].values
    y_train = train_df["Age"].values
    y_test = test_df["Age"].values

    simple_model = LinearRegression()
    simple_model.fit(X_train_s, y_train)
    pred_train_s = simple_model.predict(X_train_s)
    pred_test_s = simple_model.predict(X_test_s)

    results["simple_linear_regression"] = {
        "feature": "Weight (kg)",
        "coefficient": round(float(simple_model.coef_[0]), 4),
        "intercept": round(float(simple_model.intercept_), 4),
        "train": evaluate(y_train, pred_train_s),
        "test": evaluate(y_test, pred_test_s),
    }

    # scatter + fit line
    plt.figure(figsize=(7, 5))
    plt.scatter(test_df["Weight"], y_test, alpha=0.4, s=15, label="Actual (test)")
    order = np.argsort(X_test_s[:, 0])
    plt.plot(X_test_s[order, 0], pred_test_s[order], color="red", linewidth=2, label="Prediction")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Age (Years)")
    plt.title("Simple Linear Regression: Weight -> Age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "simple_regression.png"), dpi=130)
    plt.close()

    # ---------- Multiple Linear Regression: Breed + Weight + Color + Gender -> Age ----------
    features = ["Breed", "Weight", "Color", "Gender"]
    X_train_m = train_df[features]
    X_test_m = test_df[features]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Breed", "Color", "Gender"]),
        ],
        remainder="passthrough",
    )
    multi_pipeline = Pipeline([
        ("prep", preprocess),
        ("reg", LinearRegression()),
    ])
    multi_pipeline.fit(X_train_m, y_train)
    pred_train_m = multi_pipeline.predict(X_train_m)
    pred_test_m = multi_pipeline.predict(X_test_m)

    results["multiple_linear_regression"] = {
        "features": features,
        "train": evaluate(y_train, pred_train_m),
        "test": evaluate(y_test, pred_test_m),
    }

    # predicted vs actual plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_test_m, alpha=0.4, s=15)
    lims = [min(y_test.min(), pred_test_m.min()), max(y_test.max(), pred_test_m.max())]
    plt.plot(lims, lims, color="red", linewidth=2, linestyle="--")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Multiple Linear Regression: Predicted vs Actual Age")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "multiple_regression.png"), dpi=130)
    plt.close()

    with open(os.path.join(OUT_DIR, "lab1_regression_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    # persist predictions for later comparison in lab3
    np.savez(
        os.path.join(OUT_DIR, "lab1_predictions.npz"),
        y_test=y_test,
        pred_test_simple=pred_test_s,
        pred_test_multi=pred_test_m,
        y_train=y_train,
        pred_train_simple=pred_train_s,
        pred_train_multi=pred_train_m,
    )


if __name__ == "__main__":
    main()
