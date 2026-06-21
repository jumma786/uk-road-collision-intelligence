"""Hyperparameter tuning with Optuna for LightGBM severity prediction."""

import logging

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y, cv_folds=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


def run_optuna_tuning(X, y, n_trials: int = 50, cv_folds: int = 5) -> dict:
    logger.info(f"Starting Optuna tuning: {n_trials} trials, {cv_folds}-fold CV")

    study = optuna.create_study(direction="maximize", study_name="lgbm_severity")
    study.optimize(lambda trial: objective(trial, X, y, cv_folds), n_trials=n_trials)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best F1 macro: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "random_state": 42,
                        "n_jobs": -1, "verbose": -1})

    best_model = LGBMClassifier(**best_params)
    best_model.fit(X, y)

    return {
        "model": best_model,
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
        "n_trials": n_trials,
    }
