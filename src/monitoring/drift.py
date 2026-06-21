"""Model monitoring module - data drift detection and performance tracking."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference = reference_data
        self.threshold = threshold
        self.ref_stats = self._compute_stats(reference_data)

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include=[np.number])
        return {
            col: {"mean": numeric[col].mean(), "std": numeric[col].std(),
                   "median": numeric[col].median(), "q25": numeric[col].quantile(0.25),
                   "q75": numeric[col].quantile(0.75)}
            for col in numeric.columns
        }

    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        results = {"timestamp": datetime.now().isoformat(), "drifted_features": [],
                    "drift_scores": {}, "overall_drift": False}

        numeric_cols = set(self.reference.select_dtypes(include=[np.number]).columns)
        numeric_cols &= set(current_data.select_dtypes(include=[np.number]).columns)

        for col in numeric_cols:
            ref_values = self.reference[col].dropna()
            cur_values = current_data[col].dropna()
            if len(ref_values) < 10 or len(cur_values) < 10:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
            results["drift_scores"][col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 6),
                "drifted": p_value < self.threshold,
            }
            if p_value < self.threshold:
                results["drifted_features"].append(col)

        drift_pct = len(results["drifted_features"]) / max(len(numeric_cols), 1)
        results["overall_drift"] = drift_pct > 0.3
        results["drift_percentage"] = round(drift_pct * 100, 1)

        return results


class PerformanceMonitor:
    def __init__(self, log_dir: str = "data/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_log: list[dict] = []

    def log_prediction(self, features: dict, prediction: int,
                       probability: dict, actual: int = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "actual": actual,
            **{f"prob_{k}": v for k, v in probability.items()},
            **{f"feat_{k}": v for k, v in features.items()
               if isinstance(v, (int, float, str))},
        }
        self.predictions_log.append(entry)

    def get_performance_summary(self) -> dict:
        if not self.predictions_log:
            return {"status": "no_predictions_logged"}

        df = pd.DataFrame(self.predictions_log)
        summary = {
            "total_predictions": len(df),
            "prediction_distribution": df["prediction"].value_counts().to_dict(),
            "first_prediction": df["timestamp"].iloc[0],
            "last_prediction": df["timestamp"].iloc[-1],
        }

        if "actual" in df.columns and df["actual"].notna().any():
            matched = df[df["actual"].notna()]
            correct = (matched["prediction"] == matched["actual"]).sum()
            summary["accuracy"] = round(correct / len(matched), 4)
            summary["samples_with_actual"] = len(matched)

        return summary

    def save_log(self):
        if not self.predictions_log:
            return

        df = pd.DataFrame(self.predictions_log)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"prediction_log_{timestamp}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} prediction logs to {path}")
        return str(path)

    def check_alerts(self, window_size: int = 100) -> list[str]:
        alerts = []
        if len(self.predictions_log) < window_size:
            return alerts

        recent = pd.DataFrame(self.predictions_log[-window_size:])
        pred_dist = recent["prediction"].value_counts(normalize=True)

        if pred_dist.get(0, 0) > 0.1:
            alerts.append(
                f"High fatal prediction rate: {pred_dist.get(0, 0)*100:.1f}% "
                f"in last {window_size} predictions"
            )

        if len(pred_dist) == 1:
            alerts.append("Model predicting single class only - possible degradation")

        return alerts
