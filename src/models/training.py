"""Model training module - baseline, XGBoost, LightGBM with experiment tracking."""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

    severity_labels = {0: "Fatal", 1: "Serious", 2: "Slight"}
    target_names = [severity_labels.get(c, str(c)) for c in sorted(y_test.unique())]

    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"MODEL: {model_name}")
    logger.info(f"{'=' * 50}")
    logger.info(f"Accuracy:         {metrics['accuracy']:.4f}")
    logger.info(f"F1 (weighted):    {metrics['f1_weighted']:.4f}")
    logger.info(f"F1 (macro):       {metrics['f1_macro']:.4f}")
    logger.info(f"Precision (wt):   {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (wt):      {metrics['recall_weighted']:.4f}")
    if metrics.get("roc_auc_ovr"):
        logger.info(f"ROC-AUC (OVR):    {metrics['roc_auc_ovr']:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=target_names))
    logger.info(f"Confusion Matrix (Fatal/Serious/Slight):\n{cm}")

    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    logger.info("Training Logistic Regression baseline...")
    start = time.time()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start

    results = evaluate_model(model, X_test, y_test, "Logistic_Regression")
    results["model"] = model
    results["train_time"] = train_time
    logger.info(f"Training time: {train_time:.1f}s")
    return results


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    logger.info("Training Random Forest...")
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start

    results = evaluate_model(model, X_test, y_test, "Random_Forest")
    results["model"] = model
    results["train_time"] = train_time
    logger.info(f"Training time: {train_time:.1f}s")
    return results


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    from xgboost import XGBClassifier

    logger.info("Training XGBoost...")
    start = time.time()

    class_counts = y_train.value_counts()
    total = len(y_train)
    sample_weights = y_train.map(lambda x: total / (len(class_counts) * class_counts[x]))

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - start

    results = evaluate_model(model, X_test, y_test, "XGBoost")
    results["model"] = model
    results["train_time"] = train_time
    logger.info(f"Training time: {train_time:.1f}s")
    return results


def train_lightgbm(X_train, y_train, X_test, y_test) -> dict:
    from lightgbm import LGBMClassifier

    logger.info("Training LightGBM...")
    start = time.time()

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start

    results = evaluate_model(model, X_test, y_test, "LightGBM")
    results["model"] = model
    results["train_time"] = train_time
    logger.info(f"Training time: {train_time:.1f}s")
    return results


def cross_validate_model(model, X, y, cv_folds: int = 5) -> dict:
    logger.info(f"Running {cv_folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)

    logger.info(f"CV F1 scores: {scores.round(4)}")
    logger.info(f"CV F1 mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return {"cv_scores": scores, "cv_mean": scores.mean(), "cv_std": scores.std()}


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
    fi["cumulative_pct"] = fi["importance_pct"].cumsum().round(2)

    return fi


def save_model(model, path: str, model_name: str) -> str:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{model_name}.joblib"
    joblib.dump(model, filepath)
    logger.info(f"Model saved: {filepath}")
    return str(filepath)
