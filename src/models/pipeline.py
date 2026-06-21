"""ML pipeline - orchestrates preprocessing, training, evaluation, and model selection."""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from src.models.preprocessing import select_features, split_data
from src.models.training import (
    train_baseline,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    cross_validate_model,
    get_feature_importance,
    save_model,
    evaluate_model,
)
from src.models.tuning import run_optuna_tuning
from src.models.tracking import log_model_run, log_tuning_results, setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ml_pipeline(config_path: str = "configs/config.yaml") -> dict:
    start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 3: ML MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["paths"]["processed_data_dir"])
    model_config = config["model"]

    # Load feature matrix with geo features from Phase 2
    logger.info("[1/6] Loading feature matrix...")
    geo_path = processed_dir / "feature_matrix_geo.parquet"
    base_path = processed_dir / "feature_matrix.parquet"
    feature_path = geo_path if geo_path.exists() else base_path

    if not feature_path.exists():
        logger.error("No feature matrix found. Run Phase 1/2 first.")
        sys.exit(1)

    df = pd.read_parquet(feature_path)
    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {feature_path.name}")

    # Preprocessing
    logger.info("[2/6] Preprocessing...")
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=model_config["test_size"],
        random_state=model_config["random_state"],
    )

    # Train models
    logger.info("[3/6] Training models...")
    all_results = {}

    all_results["logistic_regression"] = train_baseline(X_train, y_train, X_test, y_test)
    all_results["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test)
    all_results["xgboost"] = train_xgboost(X_train, y_train, X_test, y_test)
    all_results["lightgbm"] = train_lightgbm(X_train, y_train, X_test, y_test)

    # Compare models
    logger.info("[4/6] Comparing models...")
    comparison = []
    for name, res in all_results.items():
        m = res["metrics"]
        comparison.append({
            "model": m["model_name"],
            "accuracy": round(m["accuracy"], 4),
            "f1_weighted": round(m["f1_weighted"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "precision_wt": round(m["precision_weighted"], 4),
            "recall_wt": round(m["recall_weighted"], 4),
            "roc_auc": round(m["roc_auc_ovr"], 4) if m.get("roc_auc_ovr") else None,
            "train_time_s": round(res["train_time"], 1),
        })

    comparison_df = pd.DataFrame(comparison).sort_values("f1_weighted", ascending=False)
    logger.info("\nMODEL COMPARISON:")
    logger.info(comparison_df.to_string(index=False))

    # Select best model
    best_name = comparison_df.iloc[0]["model"]
    best_key = [k for k, v in all_results.items() if v["metrics"]["model_name"] == best_name][0]
    best_result = all_results[best_key]
    best_model = best_result["model"]

    logger.info(f"\nBest model: {best_name} (F1={comparison_df.iloc[0]['f1_weighted']:.4f})")

    # Optuna tuning
    logger.info("[5/8] Optuna hyperparameter tuning (LightGBM)...")
    tuning_results = run_optuna_tuning(X_train, y_train, n_trials=30, cv_folds=3)
    tuned_eval = evaluate_model(
        tuning_results["model"], X_test, y_test, "LightGBM_Tuned"
    )
    all_results["lightgbm_tuned"] = {
        **tuned_eval, "model": tuning_results["model"],
        "train_time": 0, "metrics": tuned_eval["metrics"],
    }

    # Re-compare with tuned model
    m = tuned_eval["metrics"]
    comparison.append({
        "model": m["model_name"], "accuracy": round(m["accuracy"], 4),
        "f1_weighted": round(m["f1_weighted"], 4),
        "f1_macro": round(m["f1_macro"], 4),
        "precision_wt": round(m["precision_weighted"], 4),
        "recall_wt": round(m["recall_weighted"], 4),
        "roc_auc": round(m["roc_auc_ovr"], 4) if m.get("roc_auc_ovr") else None,
        "train_time_s": 0,
    })
    comparison_df = pd.DataFrame(comparison).sort_values("f1_weighted", ascending=False)

    # Cross-validation on best model
    logger.info("[6/8] Cross-validating best model...")
    cv_results = cross_validate_model(
        best_model, X, y, cv_folds=model_config["cv_folds"]
    )

    # Feature importance
    fi = get_feature_importance(best_model, X.columns.tolist())

    # MLflow tracking
    logger.info("[7/8] Logging to MLflow...")
    try:
        setup_mlflow()
        for name, res in all_results.items():
            params = {}
            if hasattr(res["model"], "get_params"):
                params = {k: str(v)[:250] for k, v in res["model"].get_params().items()}
            log_model_run(res["model"], name, res["metrics"], params)
        log_tuning_results(tuning_results)
        logger.info("MLflow tracking complete")
    except Exception as e:
        logger.warning(f"MLflow tracking failed (non-critical): {e}")

    # Save everything
    logger.info("[8/8] Saving artifacts...")
    models_dir = "mlruns/models"
    for name, res in all_results.items():
        save_model(res["model"], models_dir, name)

    output_dir = processed_dir / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    if len(fi) > 0:
        fi.to_csv(output_dir / "feature_importance.csv", index=False)

    feature_list = pd.DataFrame({"feature": X.columns.tolist()})
    feature_list.to_csv(output_dir / "feature_list.csv", index=False)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Best model: {best_name}")
    logger.info(f"F1 (weighted): {comparison_df.iloc[0]['f1_weighted']:.4f}")
    logger.info(f"CV F1 mean: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
    logger.info(f"Models saved to: {models_dir}/")

    if len(fi) > 0:
        logger.info(f"\nTop 15 Features:")
        for _, row in fi.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance_pct']:.1f}%")

    logger.info("=" * 60)

    return {
        "all_results": all_results,
        "comparison": comparison_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "cv_results": cv_results,
        "feature_importance": fi,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    run_ml_pipeline()
