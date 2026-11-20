
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "Dataset" / "student.csv"
MODEL_PATH = BASE_DIR / "Models" / "model.pkl"


def Train():
    """Train Random Forest models, track experiments and save the best model."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH).dropna(subset=["result"])

    X = data.drop(columns="result").select_dtypes(include="number")
    X = X.fillna(X.median())

    y = data["result"].astype(str)

    if X.empty:
        raise ValueError("No numerical training features found.")

    configurations = [
        (0.5, 10),
        (0.8, 50),
        (0.3, 100)
    ]

    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("students")

    best_acc = -1.0
    best_model = None
    results = []

    for i, (train_size, n_estimators) in enumerate(configurations, start=1):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=42
        )

        with mlflow.start_run(
            run_name=f"train={train_size}_trees={n_estimators}"
        ):

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42
            )

            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)

            mlflow.log_params({
                "train_size": train_size,
                "n_estimators": n_estimators
            })

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(
                model,
                name=f"model_v{i}"
            )

        results.append({
            "train_size": train_size,
            "n_estimators": n_estimators,
            "accuracy": accuracy
        })

        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    return results
