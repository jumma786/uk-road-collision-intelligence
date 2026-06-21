"""Tests for geospatial clustering module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.geo.clustering import run_dbscan, compute_cluster_stats, haversine_distance


class TestHaversineDistance:
    def test_same_point_is_zero(self):
        dist = haversine_distance(51.5, -0.1, 51.5, -0.1)
        assert dist == pytest.approx(0, abs=0.001)

    def test_known_distance(self):
        # London to Manchester ~262km
        dist = haversine_distance(51.5074, -0.1278, 53.4808, -2.2426)
        assert 250 < dist < 280

    def test_symmetric(self):
        d1 = haversine_distance(51.5, -0.1, 53.5, -2.0)
        d2 = haversine_distance(53.5, -2.0, 51.5, -0.1)
        assert d1 == pytest.approx(d2, abs=0.001)


class TestDBSCAN:
    @pytest.fixture
    def clustered_df(self):
        np.random.seed(42)
        # 3 tight clusters + noise
        c1 = np.random.normal(loc=[51.5, -0.1], scale=0.001, size=(20, 2))
        c2 = np.random.normal(loc=[53.5, -2.0], scale=0.001, size=(20, 2))
        c3 = np.random.normal(loc=[52.0, -1.0], scale=0.001, size=(20, 2))
        noise = np.random.uniform(low=[49, -6], high=[56, 2], size=(10, 2))
        all_points = np.vstack([c1, c2, c3, noise])

        df = pd.DataFrame({
            "collision_index": [f"C{i}" for i in range(70)],
            "latitude": all_points[:, 0],
            "longitude": all_points[:, 1],
            "collision_severity": np.random.choice([1, 2, 3], 70),
            "number_of_casualties": np.random.randint(1, 3, 70),
            "is_dark": np.random.randint(0, 2, 70),
            "is_wet_surface": np.random.randint(0, 2, 70),
            "is_rural": np.random.randint(0, 2, 70),
            "danger_score": np.random.randint(0, 5, 70),
            "speed_limit": np.random.choice([30, 60], 70),
        })
        return df

    def test_finds_clusters(self, clustered_df):
        result = run_dbscan(clustered_df, eps_km=0.5, min_samples=5)
        assert "hotspot_cluster" in result.columns
        n_clusters = result["hotspot_cluster"].nunique() - (1 if -1 in result["hotspot_cluster"].values else 0)
        assert n_clusters >= 2

    def test_preserves_rows(self, clustered_df):
        result = run_dbscan(clustered_df, eps_km=0.5, min_samples=5)
        assert len(result) == len(clustered_df)


class TestClusterArtifacts:
    def test_cluster_stats_exist(self):
        path = Path("data/processed/geo/cluster_stats.csv")
        assert path.exists()
        stats = pd.read_csv(path)
        assert len(stats) > 100
        assert "risk_score" in stats.columns

    def test_top_hotspots_exist(self):
        path = Path("data/processed/geo/top_hotspots.csv")
        assert path.exists()
        hotspots = pd.read_csv(path)
        assert len(hotspots) == 20

    def test_geo_feature_matrix_exists(self):
        path = Path("data/processed/feature_matrix_geo.parquet")
        assert path.exists()
        df = pd.read_parquet(path)
        assert "hotspot_cluster" in df.columns
        assert "is_in_hotspot" in df.columns
