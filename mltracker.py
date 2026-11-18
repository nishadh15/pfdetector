from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).parent / "Dataset" / "student.csv"
MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"


def Train():
    """Train several models, log them to MLflow, and save the best one."""
    data = pd.read_csv(DATA_PATH).dropna(subset=["result"])
    X = data.drop(columns="result").select_dtypes("number")
    X = X.fillna(X.median())
    y = data["result"].astype(str)

    configurations = [(0.5, 10), (0.8, 50), (0.3, 100)]
    mlflow.set_experiment("students")
    best_acc, best_model = -1.0, None
    results = []

    for i, (train_size, n_estimators) in enumerate(configurations, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42
        )

        with mlflow.start_run(run_name=f"train={train_size}_trees={n_estimators}"):
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42
            ).fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)

            mlflow.log_params({"train_size": train_size, "n_estimators": n_estimators})
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, name=f"v{i}", serialization_format="cloudpickle")

        results.append(
            {
                "train_size": train_size,
                "n_estimators": n_estimators,
                "accuracy": accuracy,
            }
        )
        if accuracy > best_acc:
            best_acc, best_model = accuracy, model

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    return results
