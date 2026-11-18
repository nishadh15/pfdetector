from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train():
    project_dir = Path(__file__).resolve().parent
    data = pd.read_csv(project_dir / "Dataset" / "student.csv").dropna(subset=["result"])

    X = data.drop(columns="result").select_dtypes("number")
    X = X.fillna(X.median())
    y = data["result"].astype(str)

    run_configs = [(0.5, 10), (0.8, 50), (0.3, 100)]
    mlflow.set_experiment("students")
    best_acc, best_model = 0, None
    results = []

    for i, (train_size, n_estimators) in enumerate(run_configs, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42
        )

        with mlflow.start_run(run_name=f"train={train_size}_trees={n_estimators}"):
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42
            ).fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            mlflow.log_params({"train_size": train_size, "n_estimators": n_estimators})
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(
                model, name=f"v{i}", serialization_format="cloudpickle"
            )

        print(f"train={train_size}  trees={n_estimators}  accuracy={acc * 100:.2f}%")
        results.append(
            {"train_size": train_size, "n_estimators": n_estimators, "accuracy": acc}
        )
        if acc > best_acc:
            best_acc, best_model = acc, model

    model_dir = project_dir / "Models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, model_dir / "model.pkl")
    return results


if __name__ == "__main__":
    train()