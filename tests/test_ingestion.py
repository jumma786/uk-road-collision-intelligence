"""Tests for data ingestion module."""

import pytest
import pandas as pd
from pathlib import Path

from src.data.ingestion import load_config, load_collision_data, load_casualty_data, load_vehicle_data


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


class TestConfig:
    def test_config_loads(self, config):
        assert "paths" in config
        assert "etl" in config
        assert "model" in config

    def test_config_has_required_paths(self, config):
        paths = config["paths"]
        assert "raw_collision" in paths
        assert "raw_casualty" in paths
        assert "raw_vehicle" in paths

    def test_config_has_model_params(self, config):
        model = config["model"]
        assert model["target"] == "collision_severity"
        assert 0 < model["test_size"] < 1


class TestCollisionIngestion:
    @pytest.fixture
    def collision_df(self, config):
        return load_collision_data(config["paths"]["raw_collision"])

    def test_loads_dataframe(self, collision_df):
        assert isinstance(collision_df, pd.DataFrame)
        assert len(collision_df) > 0

    def test_has_required_columns(self, collision_df):
        required = ["collision_index", "collision_severity", "latitude",
                     "longitude", "date", "speed_limit"]
        for col in required:
            assert col in collision_df.columns, f"Missing column: {col}"

    def test_row_count(self, collision_df):
        assert len(collision_df) > 100000

    def test_collision_index_unique(self, collision_df):
        assert collision_df["collision_index"].is_unique


class TestCasualtyIngestion:
    @pytest.fixture
    def casualty_df(self, config):
        return load_casualty_data(config["paths"]["raw_casualty"])

    def test_loads_dataframe(self, casualty_df):
        assert isinstance(casualty_df, pd.DataFrame)
        assert len(casualty_df) > 0

    def test_has_collision_index(self, casualty_df):
        assert "collision_index" in casualty_df.columns

    def test_more_casualties_than_collisions(self, casualty_df):
        assert len(casualty_df) > 100000


class TestVehicleIngestion:
    @pytest.fixture
    def vehicle_df(self, config):
        return load_vehicle_data(config["paths"]["raw_vehicle"])

    def test_loads_dataframe(self, vehicle_df):
        assert isinstance(vehicle_df, pd.DataFrame)
        assert len(vehicle_df) > 0

    def test_has_vehicle_type(self, vehicle_df):
        assert "vehicle_type" in vehicle_df.columns

    def test_more_vehicles_than_collisions(self, vehicle_df):
        assert len(vehicle_df) > 100000
