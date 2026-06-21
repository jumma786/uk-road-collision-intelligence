"""Tests for FastAPI serving module."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app, CollisionInput, engineer_input_features


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        payload = {
            "speed_limit": 30, "number_of_vehicles": 2,
            "number_of_casualties": 1, "road_type": 6,
            "light_conditions": 1, "weather_conditions": 1,
            "road_surface_conditions": 1, "urban_or_rural_area": 1,
            "day_of_week": 3, "hour": 14, "first_road_class": 3,
            "junction_detail": 0, "latitude": 51.5, "longitude": -0.1,
            "police_force": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "severity" in data
        assert data["severity"] in ["Fatal", "Serious", "Slight"]
        assert "probabilities" in data

    def test_predict_invalid_speed(self, client):
        payload = {
            "speed_limit": 999, "number_of_vehicles": 2,
            "number_of_casualties": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestInputValidation:
    def test_valid_input(self):
        inp = CollisionInput(
            speed_limit=30, number_of_vehicles=2,
            number_of_casualties=1,
        )
        assert inp.speed_limit == 30

    def test_feature_engineering(self):
        inp = CollisionInput(
            speed_limit=60, number_of_vehicles=2,
            number_of_casualties=1, light_conditions=6,
            weather_conditions=5, road_surface_conditions=2,
            urban_or_rural_area=2, hour=23,
        )
        features = engineer_input_features(inp)
        assert features["is_dark"].iloc[0] == 1
        assert features["is_rural"].iloc[0] == 1
        assert features["is_high_speed"].iloc[0] == 1
        assert features["danger_score"].iloc[0] == 5
