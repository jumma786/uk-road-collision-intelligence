"""Tests for feature engineering module."""

import pytest
import numpy as np
import pandas as pd

from src.features.engineering import (
    add_temporal_features, add_road_risk_features,
    add_condition_features, add_interaction_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "collision_index": ["A1", "A2", "A3", "A4"],
        "hour": [8, 14, 23, 3],
        "day_of_week": [2, 5, 7, 1],
        "month": [1, 6, 12, 3],
        "speed_limit": [30, 60, 70, 20],
        "urban_or_rural_area": [1, 2, 2, 1],
        "road_type": [6, 3, 6, 1],
        "junction_detail": [0, 3, 0, 13],
        "first_road_class": [6, 1, 3, 6],
        "light_conditions": [1, 4, 6, 1],
        "weather_conditions": [1, 2, 5, 1],
        "road_surface_conditions": [1, 2, 4, 1],
    })


class TestTemporalFeatures:
    def test_time_of_day(self, sample_df):
        result = add_temporal_features(sample_df.copy())
        assert "time_of_day" in result.columns

    def test_is_rush_hour(self, sample_df):
        result = add_temporal_features(sample_df.copy())
        assert result["is_rush_hour"].iloc[0] == 1  # 8am
        assert result["is_rush_hour"].iloc[1] == 0  # 2pm

    def test_is_night(self, sample_df):
        result = add_temporal_features(sample_df.copy())
        assert result["is_night"].iloc[2] == 1  # 23:00
        assert result["is_night"].iloc[3] == 1  # 3:00
        assert result["is_night"].iloc[1] == 0  # 14:00

    def test_is_weekend(self, sample_df):
        result = add_temporal_features(sample_df.copy())
        assert result["is_weekend"].iloc[2] == 1  # Saturday (7)
        assert result["is_weekend"].iloc[3] == 1  # Sunday (1)
        assert result["is_weekend"].iloc[0] == 0  # Monday (2)

    def test_cyclical_encoding(self, sample_df):
        result = add_temporal_features(sample_df.copy())
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert all(-1 <= result["hour_sin"]) and all(result["hour_sin"] <= 1)


class TestRoadRiskFeatures:
    def test_is_high_speed(self, sample_df):
        result = add_road_risk_features(sample_df.copy())
        assert result["is_high_speed"].tolist() == [0, 1, 1, 0]

    def test_is_rural(self, sample_df):
        result = add_road_risk_features(sample_df.copy())
        assert result["is_rural"].tolist() == [0, 1, 1, 0]

    def test_is_at_junction(self, sample_df):
        result = add_road_risk_features(sample_df.copy())
        assert result["is_at_junction"].iloc[0] == 0
        assert result["is_at_junction"].iloc[1] == 1

    def test_is_major_road(self, sample_df):
        result = add_road_risk_features(sample_df.copy())
        assert result["is_major_road"].iloc[1] == 1  # Motorway
        assert result["is_major_road"].iloc[0] == 0  # Unclassified


class TestConditionFeatures:
    def test_is_dark(self, sample_df):
        result = add_condition_features(sample_df.copy())
        assert result["is_dark"].iloc[0] == 0  # daylight
        assert result["is_dark"].iloc[1] == 1  # dark lit
        assert result["is_dark"].iloc[2] == 1  # dark no lighting

    def test_is_raining(self, sample_df):
        result = add_condition_features(sample_df.copy())
        assert result["is_raining"].iloc[0] == 0  # fine
        assert result["is_raining"].iloc[1] == 1  # raining
        assert result["is_raining"].iloc[2] == 1  # raining+wind

    def test_danger_score_range(self, sample_df):
        result = add_road_risk_features(sample_df.copy())
        result = add_condition_features(result)
        assert result["danger_score"].min() >= 0
        assert result["danger_score"].max() <= 5


class TestInteractionFeatures:
    def test_dark_rural(self, sample_df):
        df = add_road_risk_features(sample_df.copy())
        df = add_condition_features(df)
        result = add_interaction_features(df)
        assert "dark_rural" in result.columns
        # index 2: dark=1, rural=1 -> dark_rural=1
        assert result["dark_rural"].iloc[2] == 1
        # index 0: dark=0, rural=0 -> dark_rural=0
        assert result["dark_rural"].iloc[0] == 0

    def test_high_speed_wet(self, sample_df):
        df = add_road_risk_features(sample_df.copy())
        df = add_condition_features(df)
        result = add_interaction_features(df)
        assert "high_speed_wet" in result.columns
