"""Tests for data validation module."""

import pytest
import pandas as pd
import numpy as np

from src.data.validation import (
    validate_collision, validate_casualty, validate_vehicle,
    validate_join_integrity, generate_profile,
)


@pytest.fixture
def valid_collision():
    return pd.DataFrame({
        "collision_index": ["A1", "A2", "A3"],
        "collision_severity": [1, 2, 3],
        "latitude": [51.5, 52.0, 53.5],
        "longitude": [-0.1, -1.0, -2.0],
        "speed_limit": [30, 60, 70],
        "number_of_casualties": [1, 2, 1],
    })


@pytest.fixture
def valid_casualty():
    return pd.DataFrame({
        "collision_index": ["A1", "A2", "A2", "A3"],
        "casualty_severity": [1, 2, 3, 3],
        "age_of_casualty": [25, 40, 60, 30],
    })


@pytest.fixture
def valid_vehicle():
    return pd.DataFrame({
        "collision_index": ["A1", "A1", "A2", "A3"],
        "vehicle_type": [9, 9, 11, 1],
        "age_of_driver": [30, 45, 55, 25],
        "engine_capacity_cc": [1600, 2000, 5000, 125],
    })


class TestCollisionValidation:
    def test_passes_valid_data(self, valid_collision):
        result = validate_collision(valid_collision)
        assert result.is_valid

    def test_fails_on_empty(self):
        result = validate_collision(pd.DataFrame())
        assert not result.is_valid

    def test_fails_on_duplicate_index(self, valid_collision):
        df = valid_collision.copy()
        df.loc[2, "collision_index"] = "A1"
        result = validate_collision(df)
        assert not result.is_valid

    def test_fails_on_invalid_severity(self, valid_collision):
        df = valid_collision.copy()
        df.loc[0, "collision_severity"] = 5
        result = validate_collision(df)
        assert not result.is_valid


class TestCasualtyValidation:
    def test_passes_valid_data(self, valid_casualty):
        result = validate_casualty(valid_casualty)
        assert result.is_valid

    def test_fails_on_empty(self):
        result = validate_casualty(pd.DataFrame())
        assert not result.is_valid


class TestVehicleValidation:
    def test_passes_valid_data(self, valid_vehicle):
        result = validate_vehicle(valid_vehicle)
        assert result.is_valid


class TestJoinIntegrity:
    def test_passes_matching_keys(self, valid_collision, valid_casualty, valid_vehicle):
        tables = {"collision": valid_collision, "casualty": valid_casualty,
                  "vehicle": valid_vehicle}
        result = validate_join_integrity(tables)
        assert result.is_valid

    def test_warns_on_orphan_casualties(self, valid_collision, valid_vehicle):
        orphan_casualty = pd.DataFrame({
            "collision_index": ["A1", "ORPHAN"],
            "casualty_severity": [2, 3],
            "age_of_casualty": [25, 30],
        })
        tables = {"collision": valid_collision, "casualty": orphan_casualty,
                  "vehicle": valid_vehicle}
        result = validate_join_integrity(tables)
        assert len(result.warnings) > 0


class TestProfile:
    def test_generates_profile(self, valid_collision):
        profile = generate_profile(valid_collision, "test")
        assert "dtype" in profile.columns
        assert "null_count" in profile.columns
        assert "unique" in profile.columns
        assert len(profile) == len(valid_collision.columns)
