# Building a Full MLOps Pipeline to Predict UK Road Collision Severity

## How I turned 100,000+ government collision records into a real-time severity prediction system — from raw CSVs to a deployed dashboard

---

Every year, thousands of people die on UK roads. In 2024 alone, the Department for Transport recorded over 100,000 road collisions resulting in more than 128,000 casualties. Behind every number is a person — and behind every collision is a pattern waiting to be found.

I built the **UK Road Collision Intelligence Platform** — an end-to-end machine learning system that ingests raw government data, discovers geographic hotspots, trains severity prediction models, and serves predictions through both a live dashboard and a REST API. This article walks through the architecture, the engineering decisions, and what the data actually reveals about road safety in the UK.

---

## The Data: Three Tables, One Story

The UK Department for Transport publishes detailed collision data annually. The 2024 dataset comes in three separate CSV files:

- **Collisions** (~100,927 records) — location, time, road conditions, severity
- **Casualties** (~128,272 records) — age, type, severity of each person involved
- **Vehicles** (~188,000+ records) — vehicle type, age, engine capacity, driver age

Each collision can involve multiple vehicles and multiple casualties, creating a one-to-many relationship that needs careful aggregation during feature engineering.

The raw data uses numeric codes for everything — `1` means "Fatal," `6` means "Single carriageway," `-1` means "Missing." The first challenge was turning this into something meaningful.

---

## Phase 1: The ETL Pipeline

### Ingestion

Nothing fancy here — just `pandas.read_csv` with `low_memory=False` to let pandas infer types freely. All three tables load through a config-driven system (`configs/config.yaml`) so paths and parameters live outside the code.

### Cleaning

The cleaning pipeline handles several concerns:

**Missing codes → NaN.** The DfT encodes missing values as `-1` across all numeric columns. I scan every numeric column, replace `-1` with `NaN`, and log how many were found. This single step affected thousands of records.

**Date parsing.** Collision dates arrive as `DD/MM/YYYY` strings with separate time fields. These get parsed into proper datetimes, and I extract `hour`, `month`, and `week` for temporal analysis.

**Label mappings.** I maintain a comprehensive mapping dictionary that translates every numeric code into human-readable labels — severity, road type, weather conditions, light conditions, vehicle types. These label columns power the dashboard filters and visualizations.

**Outlier clipping.** Driver ages get clipped to 16–100, casualty ages to 0–110, engine capacity to 0–10,000cc, and vehicle age to 0–80 years. These bounds catch data entry errors without discarding valid records.

**Constant column removal.** Any column with zero variance gets dropped automatically.

The cleaned tables are saved as Parquet files — roughly 5x smaller than the original CSVs and significantly faster to read.

---

## Phase 2: Feature Engineering

This is where raw columns become predictive signals. I engineered features across five categories:

### Temporal Features

Raw hour and day-of-week values don't capture cyclical patterns well — hour 23 and hour 0 are numerically distant but temporally adjacent. I used **sine/cosine encoding** to fix this:

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

Plus binary flags: `is_rush_hour` (7–9am, 4–6pm), `is_night` (10pm–5am), `is_weekend`.

### Road Risk Features

- `is_high_speed` — speed limit ≥ 60mph
- `is_rural` — urban/rural area code = 2
- `is_single_carriageway` — the most common road type in fatal collisions
- `is_at_junction` — junction detail is non-zero
- `is_major_road` — motorway, A-road, or B-road

### Condition Features

- `is_dark` — any darkness condition (lit, unlit, no lighting)
- `is_adverse_weather` — rain, snow, fog, high winds
- `is_wet_surface` — wet, damp, frost, ice, snow

These roll up into a **composite danger score** (0–5) that sums the binary risk flags. Simple, interpretable, and surprisingly predictive.

### Interaction Features

Some risk factors compound. I explicitly modeled four key interactions:

- `dark_rural` — darkness × rural area
- `high_speed_wet` — high speed × wet surface
- `weekend_night` — weekend × nighttime
- `rural_high_speed` — rural × high speed zone

### Vehicle and Casualty Aggregations

Since multiple vehicles and casualties map to a single collision, I aggregated up:

- **Vehicle-level:** average driver age, max engine capacity, motorcycle/HGV involvement, skid events, e-scooter flags
- **Casualty-level:** average/minimum casualty age, pedestrian count, child and elderly casualty flags

The final feature matrix: **100,927 rows × 60+ features**.

---

## Phase 3: Geospatial Hotspot Detection

Not all locations are equally dangerous. I used **DBSCAN clustering** on collision coordinates to identify geographic hotspots — areas where collisions cluster abnormally.

### Why DBSCAN?

DBSCAN doesn't require specifying the number of clusters upfront (unlike K-Means), and it naturally handles noise — isolated collisions that don't belong to any cluster. Perfect for spatial data.

The key parameter is `eps` — the maximum distance between two points to be considered neighbors. I set this to **0.5 km** (converted to radians for the haversine metric) with a minimum of 5 collisions per cluster.

```python
dbscan = DBSCAN(
    eps=eps_rad,           # 0.5km in radians
    metric="haversine",    # proper Earth-surface distance
    algorithm="ball_tree", # efficient for haversine
    min_samples=5,
    n_jobs=-1,
)
```

### Cluster Risk Scoring

Each cluster gets profiled with a composite risk score:

```
risk_score = fatality_rate × 50
           + serious_injury_rate × 30
           + avg_danger_score / 5 × 10
           + relative_cluster_size × 10
```

Clusters are then binned into risk tiers — **Critical** (top 5%), **High** (top 20%), **Medium** (top 50%), and **Low**. The cluster features (`cluster_risk_score`, `is_in_hotspot`) feed back into the ML model as additional predictive signals.

---

## Phase 4: Model Training

### The Setup

Target variable: **collision severity** (Fatal / Serious / Slight).

This is a heavily imbalanced multi-class problem:
- Slight: ~83%
- Serious: ~15%
- Fatal: ~2%

Every model uses `class_weight="balanced"` or manual sample weighting to counteract this imbalance. Evaluation uses **F1-weighted** and **F1-macro** (which treats all classes equally regardless of size) alongside standard accuracy.

### Four Models, Head to Head

I trained four classifiers with 5-fold stratified cross-validation:

| Model | Key Config |
|-------|-----------|
| **Logistic Regression** | Baseline, balanced weights, max_iter=1000 |
| **Random Forest** | 200 trees, max_depth=20, balanced weights |
| **XGBoost** | 300 estimators, max_depth=8, manual sample weights |
| **LightGBM** | 300 estimators, max_depth=8, balanced weights |

### Hyperparameter Tuning

The best-performing model (LightGBM) gets a second pass through **Optuna** — a Bayesian hyperparameter optimization framework. I ran 50 trials optimizing F1-macro with 5-fold CV:

```python
params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 600),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    # ... plus regularization params
}
```

### Why LightGBM Won

LightGBM consistently outperformed the others, particularly on **Fatal recall** — the metric that matters most for a road safety application. A model that never predicts "Fatal" can still achieve 98% accuracy, but it's useless for identifying dangerous collisions before they happen.

The key advantage: LightGBM's leaf-wise tree growth strategy captures complex feature interactions (like `dark_rural` combined with `high_speed_wet`) more efficiently than level-wise approaches.

### Feature Importance

The top predictive features tell a clear story:

1. **Number of casualties** — more casualties correlate with more severe collisions
2. **Speed limit** — fatal collision rates rise sharply above 60mph
3. **Casualty age** — elderly casualties face disproportionately severe outcomes
4. **Engine capacity** — a proxy for vehicle size/type differences
5. **Danger score** — the composite condition risk metric
6. **is_dark** / **is_rural** — and their interaction `dark_rural`

### Target Leakage Prevention

One critical engineering decision: I explicitly dropped columns that encode or are derived from the target variable. Columns like `fatal_casualty_count`, `serious_casualty_count`, and `casualty_severity` would give the model perfect information about the outcome — making it useless for actual prediction. These are stripped out during preprocessing.

---

## Serving: Dashboard + API

### Streamlit Dashboard

The interactive dashboard has five pages:

1. **Overview** — KPI cards (total collisions, fatal count, casualty count), severity distribution pie chart, hourly/daily/monthly trends, speed limit vs. fatality rate
2. **Hotspot Map** — Interactive Mapbox scatter plot of 5,000 sampled collisions color-coded by severity, plus a cluster risk map showing the top 100 hotspots sized by collision count and colored by risk tier
3. **Conditions Analysis** — Fatal rate breakdowns by light conditions, weather, road surface, and danger score
4. **Model Performance** — Side-by-side model comparison chart and top 20 feature importance
5. **Predict Severity** — Interactive form where users input road conditions, driver details, and collision characteristics to get a real-time severity prediction with probability breakdown

The sidebar provides **live filtering** across all analytical pages — by severity, area, month range, hour range, road class, speed limit, and weather conditions.

### FastAPI REST API

For programmatic access, a FastAPI server exposes:

- `POST /predict` — single collision prediction with severity label, probability distribution, and identified risk factors
- `POST /predict/batch` — batch predictions
- `GET /health` — model status and feature count

The API replicates the full feature engineering pipeline at inference time — every binary flag, interaction term, and danger score calculation is reproduced from raw inputs to ensure consistency between training and serving.

---

## Monitoring: Drift Detection

Deployed models degrade. Road patterns change — new speed limits, infrastructure changes, policy shifts. The monitoring module implements:

**Statistical drift detection** using the Kolmogorov–Smirnov test across all numeric features. If more than 30% of features show statistically significant distribution shift (p < 0.05), the system flags overall drift.

**Performance monitoring** that logs every prediction with timestamps and features, enabling trend analysis on prediction distributions. An alert triggers if fatal prediction rates spike above 10% — a signal that either the model or the input data has shifted.

---

## CI/CD Pipeline

The GitHub Actions workflow runs the full pipeline on every push:

1. Install dependencies
2. Run ETL pipeline
3. Run geospatial pipeline
4. Run ML pipeline
5. Execute test suite (`pytest`)
6. Upload model artifacts

This ensures that code changes don't silently break the pipeline, and that model artifacts are always reproducible from source.

---

## What the Data Reveals

Some findings from the analysis:

**Time patterns.** Collisions peak during evening rush hour (4–6pm), but **fatal** collisions peak later — between 8pm and midnight. Weekend nights are particularly dangerous.

**Speed kills — literally.** The fatality rate jumps dramatically at 60mph and above. Rural single carriageways at national speed limit are the highest-risk road type.

**Darkness compounds risk.** Dark conditions without street lighting have the highest fatal rate of any light condition. Combined with rural locations (`dark_rural` interaction), this is the single strongest risk signal.

**Weather is less predictive than you'd think.** Fine weather actually sees more collisions (more traffic). The fatal *rate* increases in fog and heavy rain, but the absolute numbers are smaller.

**Vehicle type matters.** Motorcycle involvement significantly increases severity. HGV involvement increases casualty counts. E-scooter collisions are a growing category.

---

## Architecture Decisions Worth Noting

**Config-driven everything.** Paths, parameters, thresholds — all in `config.yaml`. No magic numbers buried in code.

**Modular pipeline phases.** ETL, Geo, and ML can run independently or sequentially via `scripts/run_all.py`. Each phase reads from and writes to `data/processed/`.

**Parquet over CSV.** Faster reads, smaller files, preserved dtypes. The feature matrix loads in under a second versus 10+ seconds from CSV.

**Leakage-aware preprocessing.** A dedicated `DROP_COLUMNS` list in the preprocessing module explicitly blocks target-derived columns. This is easy to audit and extend.

**Label mapping separation.** Human-readable labels live in a dedicated `mappings.py` module — a single source of truth used by cleaning, the dashboard, and the API.

---

## What I'd Build Next

1. **SHAP explanations** on each prediction — not just "Serious," but *why* the model thinks so.
2. **Time-series forecasting** — predicting collision counts by region and week, not just individual severity.
3. **Real-time ingestion** — connecting to live police incident feeds instead of annual CSV dumps.
4. **A/B testing framework** for model versions in production.
5. **Automated retraining** triggered by the drift detector.

---

## Try It Yourself

The entire platform is open source. Clone it, run `python scripts/run_all.py`, and the full pipeline executes in minutes. Then launch the dashboard with `streamlit run src/dashboard/app.py` to explore the data interactively.

The prediction API starts with `uvicorn src.serving.app:app --reload` and accepts JSON requests at `/predict`.

---

**The code doesn't just analyze collisions — it builds the infrastructure to keep analyzing them as new data arrives.** That's the difference between a notebook and a platform, and it's where real MLOps begins.

---

*Built with Python, scikit-learn, LightGBM, XGBoost, Optuna, DBSCAN, FastAPI, Streamlit, and Plotly. Data from the UK Department for Transport Road Casualty Statistics 2024.*
