"""Tests for model training and prediction."""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.models.preprocessing import select_features, split_data
from src.models.training import evaluate_model, get_feature_importance


@pytest.fixture
def sample_feature_matrix():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "collision_index": [f"C{i}" for i in range(n)],
        "collision_severity": np.random.choice([1, 2, 3], n, p=[0.02, 0.23, 0.75]),
        "speed_limit": np.random.choice([20, 30, 40, 60, 70], n),
        "number_of_vehicles": np.random.randint(1, 5, n),
        "number_of_casualties": np.random.randint(1, 4, n),
        "is_dark": np.random.randint(0, 2, n),
        "is_rural": np.random.randint(0, 2, n),
        "is_wet_surface": np.random.randint(0, 2, n),
        "danger_score": np.random.randint(0, 6, n),
        "latitude": np.random.uniform(50, 56, n),
        "longitude": np.random.uniform(-5, 1, n),
        "hour": np.random.randint(0, 24, n),
    })


class TestPreprocessing:
    def test_select_features_returns_xy(self, sample_feature_matrix):
        X, y = select_features(sample_feature_matrix)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_target_removed_from_features(self, sample_feature_matrix):
        X, y = select_features(sample_feature_matrix)
        assert "collision_severity" not in X.columns

    def test_id_columns_removed(self, sample_feature_matrix):
        X, y = select_features(sample_feature_matrix)
        assert "collision_index" not in X.columns

    def test_target_values(self, sample_feature_matrix):
        X, y = select_features(sample_feature_matrix)
        assert set(y.unique()).issubset({0, 1, 2})

    def test_split_preserves_size(self, sample_feature_matrix):
        X, y = select_features(sample_feature_matrix)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_split_stratified(self, sample_feature_matrix):
        X, y = select_features(sample_feature_matrix)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        for cls in train_dist.index:
            assert abs(train_dist[cls] - test_dist[cls]) < 0.05


class TestModelArtifacts:
    def test_models_exist(self):
        models_dir = Path("mlruns/models")
        assert models_dir.exists()
        model_files = list(models_dir.glob("*.joblib"))
        assert len(model_files) >= 4

    def test_lightgbm_loads(self):
        model = joblib.load("mlruns/models/lightgbm.joblib")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_feature_list_exists(self):
        path = Path("data/processed/ml/feature_list.csv")
        assert path.exists()
        features = pd.read_csv(path)
        assert len(features) > 50

    def test_model_comparison_exists(self):
        path = Path("data/processed/ml/model_comparison.csv")
        assert path.exists()
        comp = pd.read_csv(path)
        assert len(comp) >= 4
        assert "f1_weighted" in comp.columns


class TestFeatureImportance:
    def test_feature_importance_file(self):
        path = Path("data/processed/ml/feature_importance.csv")
        assert path.exists()
        fi = pd.read_csv(path)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert fi["importance"].sum() > 0
