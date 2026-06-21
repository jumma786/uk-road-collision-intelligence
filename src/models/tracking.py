"""MLflow experiment tracking module."""

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

logger = logging.getLogger(__name__)

TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "uk_road_collision_severity"


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking: {TRACKING_URI}, experiment: {EXPERIMENT_NAME}")


def log_model_run(model, model_name: str, metrics: dict, params: dict,
                  feature_importance: pd.DataFrame = None,
                  X_train=None, y_train=None):
    setup_mlflow()

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(model, artifact_path="model")

        if feature_importance is not None:
            fi_path = Path("data/processed/ml") / f"fi_{model_name}.csv"
            fi_path.parent.mkdir(parents=True, exist_ok=True)
            feature_importance.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))

        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("dataset", "dft_road_casualty_2024")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Logged {model_name} to MLflow (run_id: {run_id})")
        return run_id


def log_tuning_results(study_results: dict, model_name: str = "lgbm_tuned"):
    setup_mlflow()

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({k: str(v)[:250] for k, v in study_results["best_params"].items()})
        mlflow.log_metric("best_f1_macro", study_results["best_score"])
        mlflow.log_metric("n_trials", study_results["n_trials"])
        mlflow.sklearn.log_model(study_results["model"], artifact_path="model")
        mlflow.set_tag("tuning", "optuna")
        mlflow.set_tag("model_type", "LGBMClassifier")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Logged tuned model to MLflow (run_id: {run_id})")
        return run_id


def get_best_run() -> dict:
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_weighted DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        return {}

    best = runs.iloc[0]
    return {
        "run_id": best["run_id"],
        "f1_weighted": best.get("metrics.f1_weighted"),
        "accuracy": best.get("metrics.accuracy"),
        "model_name": best.get("tags.mlflow.runName"),
    }
