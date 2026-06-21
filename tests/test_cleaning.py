"""Tests for data cleaning module."""

import pytest
import numpy as np
import pandas as pd

from src.data.cleaning import (
    replace_missing_codes, parse_dates, apply_label_mappings,
    drop_constant_columns, clean_collision, clean_casualty, clean_vehicle,
)
from src.data.mappings import SEVERITY, COLLISION_TABLE_MAPPINGS


@pytest.fixture
def sample_collision():
    return pd.DataFrame({
        "collision_index": ["A1", "A2", "A3"],
        "collision_severity": [1, 2, 3],
        "speed_limit": [30, -1, 60],
        "light_conditions": [1, 4, -1],
        "date": ["01/01/2024", "15/06/2024", "31/12/2024"],
        "time": ["08:30", "14:00", "23:45"],
        "constant_col": [1, 1, 1],
        "latitude": [51.5, 52.0, 53.5],
        "longitude": [-0.1, -1.0, -2.0],
        "collision_year": [2024, 2024, 2024],
        "local_authority_district": [-1, -1, -1],
    })


class TestReplaceMissingCodes:
    def test_replaces_minus_one(self, sample_collision):
        result = replace_missing_codes(sample_collision.copy())
        assert result["speed_limit"].isna().sum() == 1
        assert result["light_conditions"].isna().sum() == 1

    def test_preserves_valid_values(self, sample_collision):
        result = replace_missing_codes(sample_collision.copy())
        assert result["collision_severity"].tolist() == [1, 2, 3]

    def test_does_not_affect_strings(self, sample_collision):
        result = replace_missing_codes(sample_collision.copy())
        assert result["collision_index"].tolist() == ["A1", "A2", "A3"]


class TestParseDates:
    def test_parses_date_column(self, sample_collision):
        result = parse_dates(sample_collision.copy())
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_creates_datetime(self, sample_collision):
        result = parse_dates(sample_collision.copy())
        assert "datetime" in result.columns

    def test_extracts_hour(self, sample_collision):
        result = parse_dates(sample_collision.copy())
        assert "hour" in result.columns
        assert result["hour"].tolist() == [8, 14, 23]

    def test_extracts_month(self, sample_collision):
        result = parse_dates(sample_collision.copy())
        assert "month" in result.columns
        assert result["month"].tolist() == [1, 6, 12]


class TestApplyLabelMappings:
    def test_creates_label_columns(self):
        df = pd.DataFrame({"collision_severity": [1, 2, 3]})
        result = apply_label_mappings(df, {"collision_severity": SEVERITY})
        assert "collision_severity_label" in result.columns
        assert result["collision_severity_label"].tolist() == ["Fatal", "Serious", "Slight"]

    def test_handles_unknown_codes(self):
        df = pd.DataFrame({"collision_severity": [1, 99]})
        result = apply_label_mappings(df, {"collision_severity": SEVERITY})
        assert result["collision_severity_label"].iloc[1] == "Unknown"


class TestDropConstantColumns:
    def test_drops_constant(self, sample_collision):
        result = drop_constant_columns(sample_collision.copy())
        assert "constant_col" not in result.columns

    def test_keeps_varying(self, sample_collision):
        result = drop_constant_columns(sample_collision.copy())
        assert "collision_severity" in result.columns


class TestCleanCollision:
    def test_adds_label_columns(self, sample_collision):
        result = clean_collision(sample_collision.copy())
        assert "collision_severity_label" in result.columns

    def test_drops_constant_year(self, sample_collision):
        result = clean_collision(sample_collision.copy())
        assert "collision_year" not in result.columns
