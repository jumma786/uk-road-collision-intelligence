"""FastAPI serving module - REST API for collision severity prediction."""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="UK Road Collision Severity Predictor",
    description="Predict collision severity (Fatal/Serious/Slight) from road and condition features",
    version="1.0.0",
)

MODEL_PATH = "mlruns/models/lightgbm.joblib"
# LightGBM chosen over Random Forest: 4x better Fatal recall (33% vs 8%)
FEATURE_LIST_PATH = "data/processed/ml/feature_list.csv"

model = None
feature_names = None


class CollisionInput(BaseModel):
    speed_limit: int = Field(..., ge=20, le=70, description="Speed limit in mph")
    number_of_vehicles: int = Field(..., ge=1, le=30)
    number_of_casualties: int = Field(..., ge=1, le=30)
    road_type: int = Field(6, description="1=Roundabout, 3=Dual, 6=Single, 7=Slip")
    light_conditions: int = Field(1, description="1=Daylight, 4=Dark-lit, 5=Dark-unlit")
    weather_conditions: int = Field(1, description="1=Fine, 2=Rain, 7=Fog")
    road_surface_conditions: int = Field(1, description="1=Dry, 2=Wet, 4=Frost")
    urban_or_rural_area: int = Field(1, description="1=Urban, 2=Rural")
    day_of_week: int = Field(2, ge=1, le=7, description="1=Sun, 7=Sat")
    hour: int = Field(12, ge=0, le=23)
    first_road_class: int = Field(6, description="1=Motorway, 3=A, 6=Unclassified")
    junction_detail: int = Field(0, description="0=Not at junction")
    latitude: float = Field(51.5, description="UK latitude")
    longitude: float = Field(-0.1, description="UK longitude")
    police_force: int = Field(1)

    model_config = {"json_schema_extra": {
        "examples": [{
            "speed_limit": 30, "number_of_vehicles": 2,
            "number_of_casualties": 1, "road_type": 6,
            "light_conditions": 1, "weather_conditions": 1,
            "road_surface_conditions": 1, "urban_or_rural_area": 1,
            "day_of_week": 3, "hour": 14, "first_road_class": 3,
            "junction_detail": 0, "latitude": 51.5, "longitude": -0.1,
            "police_force": 1,
        }]
    }}


class PredictionOutput(BaseModel):
    severity: str
    severity_code: int
    probabilities: dict[str, float]
    risk_factors: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    feature_count: int


def engineer_input_features(data: CollisionInput) -> pd.DataFrame:
    d = data.model_dump()

    d["is_rush_hour"] = int(d["hour"] in [7, 8, 9, 16, 17, 18])
    d["is_night"] = int(d["hour"] >= 22 or d["hour"] <= 5)
    d["is_weekend"] = int(d["day_of_week"] in [1, 7])
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["day_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["day_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)

    d["is_high_speed"] = int(d["speed_limit"] >= 60)
    d["is_rural"] = int(d["urban_or_rural_area"] == 2)
    d["is_single_carriageway"] = int(d["road_type"] == 6)
    d["is_at_junction"] = int(d["junction_detail"] != 0)
    d["is_major_road"] = int(d["first_road_class"] in [1, 2, 3])

    d["is_dark"] = int(d["light_conditions"] in [4, 5, 6, 7])
    d["is_adverse_weather"] = int(d["weather_conditions"] in [2, 3, 5, 6, 7])
    d["is_raining"] = int(d["weather_conditions"] in [2, 5])
    d["is_wet_surface"] = int(d["road_surface_conditions"] in [2, 3, 4, 5])

    d["danger_score"] = (
        d["is_dark"] + d["is_adverse_weather"] + d["is_wet_surface"]
        + d["is_high_speed"] + d["is_rural"]
    )

    d["dark_rural"] = d["is_dark"] * d["is_rural"]
    d["high_speed_wet"] = d["is_high_speed"] * d["is_wet_surface"]
    d["weekend_night"] = d["is_night"] * d["is_weekend"]
    d["rural_high_speed"] = d["is_rural"] * d["is_high_speed"]

    uk_center_lat, uk_center_lon = 53.5, -1.5
    d["dist_from_center_km"] = np.sqrt(
        ((d["latitude"] - uk_center_lat) * 111) ** 2
        + ((d["longitude"] - uk_center_lon) * 111 * np.cos(np.radians(53.5))) ** 2
    )

    row = pd.DataFrame([d])

    if feature_names is not None:
        for col in feature_names:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_names]

    return row


def identify_risk_factors(data: CollisionInput) -> list[str]:
    factors = []
    if data.speed_limit >= 60:
        factors.append("High speed zone (60+ mph)")
    if data.light_conditions in [4, 5, 6, 7]:
        factors.append("Dark conditions")
    if data.weather_conditions in [2, 3, 5, 6, 7]:
        factors.append("Adverse weather")
    if data.road_surface_conditions in [2, 3, 4, 5]:
        factors.append("Poor road surface")
    if data.urban_or_rural_area == 2:
        factors.append("Rural area")
    if data.number_of_vehicles >= 3:
        factors.append("Multi-vehicle collision")
    if data.hour >= 22 or data.hour <= 5:
        factors.append("Night time")
    return factors


def _load_model_and_features():
    global model, feature_names

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        for alt in ["xgboost", "random_forest", "logistic_regression"]:
            alt_path = model_path.parent / f"{alt}.joblib"
            if alt_path.exists():
                model_path = alt_path
                break

    if model_path.exists():
        model = joblib.load(model_path)
        logger.info(f"Model loaded: {model_path}")
    else:
        logger.warning(f"No model found at {MODEL_PATH}")

    fl_path = Path(FEATURE_LIST_PATH)
    if fl_path.exists():
        feature_names = pd.read_csv(fl_path)["feature"].tolist()
        logger.info(f"Feature list loaded: {len(feature_names)} features")


@app.on_event("startup")
def load_model():
    _load_model_and_features()


_load_model_and_features()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model is not None else "no_model",
        model_loaded=model is not None,
        model_name=type(model).__name__ if model else "none",
        feature_count=len(feature_names) if feature_names else 0,
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(data: CollisionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = engineer_input_features(data)
    prediction = model.predict(features)[0]

    severity_map = {0: "Fatal", 1: "Serious", 2: "Slight"}
    severity_label = severity_map.get(prediction, "Unknown")

    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        probabilities = {
            severity_map[i]: round(float(p), 4) for i, p in enumerate(proba)
        }

    risk_factors = identify_risk_factors(data)

    return PredictionOutput(
        severity=severity_label,
        severity_code=int(prediction),
        probabilities=probabilities,
        risk_factors=risk_factors,
    )


@app.post("/predict/batch")
def predict_batch(inputs: list[CollisionInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for data in inputs:
        features = engineer_input_features(data)
        prediction = model.predict(features)[0]
        severity_map = {0: "Fatal", 1: "Serious", 2: "Slight"}
        results.append({
            "severity": severity_map.get(prediction, "Unknown"),
            "severity_code": int(prediction),
        })
    return results
