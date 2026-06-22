---
title: UK Road Collision Intelligence
emoji: 🚗
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.28.0"
app_file: src/dashboard/app.py
pinned: false
---

# UK Road Collision Intelligence Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://uk-road-collision-intelligence-imutd2qu9ybzujv9appkmd.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/JUMMAMOHAMMAD477/uk-road-collision-intelligence)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)](https://uk-road-collision-intelligence.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end MLOps platform for analyzing and predicting UK road collision severity using the Department for Transport (DfT) 2024 road casualty statistics.

## Live Demo

| Platform | Link |
|----------|------|
| **Streamlit Cloud** | [uk-road-collision-intelligence.streamlit.app](https://uk-road-collision-intelligence-imutd2qu9ybzujv9appkmd.streamlit.app/) |
| **Hugging Face Spaces** | [huggingface.co/spaces/JUMMAMOHAMMAD477](https://huggingface.co/spaces/JUMMAMOHAMMAD477/uk-road-collision-intelligence) |
| **Render** | [uk-road-collision-intelligence.onrender.com](https://uk-road-collision-intelligence.onrender.com) |

## Overview

This project processes **100,927 collisions**, **128,272 casualties**, and **183,514 vehicles** from the official DfT dataset to build a complete machine learning pipeline — from raw data ingestion to a deployed prediction API and interactive dashboard.

## Key Findings

| Finding | Value |
|---------|-------|
| Night driving fatal rate | **2.1x** higher than daytime |
| Rural vs urban fatal rate | **3.4x** higher in rural areas |
| 60mph vs 20mph fatal rate | **6.6x** higher at 60mph |
| HGV involvement fatal rate | **5.96%** (4x the average) |
| Elderly casualty (65+) fatal rate | **3.58%** (2.4x average) |
| Dangerous hotspot clusters | **2,335** identified (117 critical) |
| Peak collision hour | **5 PM** (8,784 collisions) |

## Architecture

```
Raw CSV (3 tables, 412K records)
    |
    v
[Phase 1] ETL Pipeline ──> Cleaned Parquet + 112 Engineered Features
    |
    v
[Phase 2] DBSCAN Clustering ──> 2,335 Hotspot Clusters + Risk Scores
    |
    v
[Phase 3] ML Training ──> 5 Models (LR, RF, XGBoost, LightGBM, Tuned)
    |
    v
[Phase 4] MLOps ──> MLflow Tracking + FastAPI Serving + Drift Monitoring
    |
    v
[Phase 5] Dashboard ──> 5-Page Streamlit App with Live Predictions
```

## Project Structure

```
uk-road-collision-intelligence/
├── configs/
│   └── config.yaml                 # All project configuration
├── data/
│   ├── processed/                  # Pipeline outputs (Parquet)
│   └── predictions/                # Prediction logs
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── src/
│   ├── data/                       # ETL: ingestion, cleaning, validation
│   │   ├── ingestion.py            # Load 3 DfT CSV tables
│   │   ├── cleaning.py             # Handle -1 codes, parse dates, map labels
│   │   ├── validation.py           # Schema checks, quality assertions
│   │   ├── mappings.py             # 18 DfT code-to-label dictionaries
│   │   └── pipeline.py             # ETL orchestrator
│   ├── features/
│   │   └── engineering.py          # 112 features: temporal, road, conditions, geo
│   ├── geo/                        # Geospatial analysis
│   │   ├── clustering.py           # DBSCAN with haversine distance
│   │   ├── analysis.py             # Hotspot summaries, police force breakdown
│   │   └── pipeline.py             # Geo pipeline orchestrator
│   ├── models/                     # ML training
│   │   ├── preprocessing.py        # Feature selection, encoding, split
│   │   ├── training.py             # LR, RF, XGBoost, LightGBM trainers
│   │   ├── tuning.py               # Optuna hyperparameter optimization
│   │   ├── tracking.py             # MLflow experiment tracking
│   │   └── pipeline.py             # ML pipeline orchestrator
│   ├── serving/
│   │   └── app.py                  # FastAPI prediction API
│   ├── monitoring/
│   │   └── drift.py                # KS-test drift detection, performance tracking
│   └── dashboard/
│       └── app.py                  # Streamlit 5-page dashboard
├── tests/                          # 74 unit tests (pytest)
│   ├── test_ingestion.py
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_validation.py
│   ├── test_model.py
│   ├── test_geo.py
│   └── test_serving.py
├── scripts/
│   └── run_all.py                  # Run all phases sequentially
├── mlruns/models/                  # Saved model artifacts (.joblib)
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── Dockerfile                      # Container deployment
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Data

Download the 3 DfT CSV files from [road casualty statistics](https://www.data.gov.uk/dataset/road-accidents-safety-data) and place them in the project root:
- `dft-road-casualty-statistics-collision-2024_raw_data.csv`
- `dft-road-casualty-statistics-casualty-2024.csv`
- `dft-road-casualty-statistics-vehicle-2024.csv`

### 3. Run the Full Pipeline

```bash
python scripts/run_all.py
```

This runs all 3 phases (ETL, Geospatial, ML) and takes ~15 minutes.

### 4. Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open http://localhost:8501

### 5. Start the API

```bash
uvicorn src.serving.app:app --reload
```

API docs at http://localhost:8000/docs

### 6. Run Tests

```bash
pytest tests/ -v
```

## Pipeline Phases

### Phase 1: ETL + Feature Engineering

- Ingests 412,713 records across 3 relational tables
- Replaces 1.28M missing codes (-1 values) with NaN
- Parses dates, maps all DfT coded values to human-readable labels
- Validates schema, primary keys, join integrity (100% coverage)
- Engineers 112 features: temporal, road risk, conditions, vehicle/casualty aggregates, interactions

### Phase 2: Geospatial DBSCAN Clustering

- Clusters 100K collision GPS coordinates using DBSCAN with haversine distance
- Identifies 2,335 hotspot clusters (61.5% of collisions fall in hotspots)
- Risk-tiers clusters: 117 Critical, 336 High, 706 Medium, 1,176 Low
- Critical clusters have 9.9% fatality rate (6.6x national average)

### Phase 3: ML Model Training

| Model | F1 Weighted | F1 Macro | ROC-AUC | Fatal Recall |
|-------|------------|----------|---------|-------------|
| Random Forest | 0.7291 | 0.4638 | 0.7366 | 8% |
| XGBoost | 0.7115 | 0.4647 | 0.7403 | 11% |
| **LightGBM Tuned** | **0.6946** | **0.4822** | **0.7489** | **35%** |
| LightGBM | 0.6909 | 0.4679 | 0.7446 | 33% |
| Logistic Regression | 0.4835 | 0.2869 | 0.5630 | 57% |

LightGBM Tuned is selected for production — best ROC-AUC and F1 macro, with 4x better fatal recall than Random Forest.

### Phase 4: MLOps

- **MLflow**: Experiment tracking, model registry, artifact logging
- **Optuna**: 30-trial Bayesian hyperparameter optimization
- **FastAPI**: `/predict`, `/health`, `/predict/batch` endpoints with Pydantic validation
- **Monitoring**: KS-test drift detection, prediction logging, alerting

### Phase 5: Dashboard

5-page Streamlit app:
1. **Overview** — KPIs, severity distribution, hourly/daily/monthly trends
2. **Hotspot Map** — Interactive collision map + top 100 cluster map by risk tier
3. **Conditions Analysis** — Fatal rates by light, weather, surface, danger score
4. **Model Performance** — Model comparison + feature importance
5. **Predict Severity** — Live prediction with probability breakdown

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "speed_limit": 60,
    "number_of_vehicles": 2,
    "number_of_casualties": 1,
    "light_conditions": 6,
    "weather_conditions": 5,
    "road_surface_conditions": 2,
    "urban_or_rural_area": 2,
    "hour": 23
  }'
```

Response:
```json
{
  "severity": "Serious",
  "severity_code": 1,
  "probabilities": {"Fatal": 0.03, "Serious": 0.78, "Slight": 0.19},
  "risk_factors": ["High speed zone (60+ mph)", "Dark conditions", "Adverse weather", "Rural area", "Night time"]
}
```

## Docker

```bash
docker build -t uk-collision-api .
docker run -p 8000:8000 uk-collision-api
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | pandas, numpy, PyArrow |
| ML | scikit-learn, XGBoost, LightGBM |
| Tuning | Optuna |
| Tracking | MLflow |
| API | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |
| Geospatial | DBSCAN (haversine), Plotly Mapbox |
| Monitoring | scipy (KS-test), custom drift detector |
| Testing | pytest (74 tests) |
| CI/CD | GitHub Actions |
| Container | Docker |

## Data Source

[Department for Transport — Road Casualty Statistics 2024](https://www.data.gov.uk/dataset/road-accidents-safety-data)

Official UK government data covering every reported road collision in Great Britain for the year 2024.

## License

MIT
