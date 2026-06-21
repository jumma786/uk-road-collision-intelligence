"""Streamlit dashboard - UK Road Collision Intelligence Platform."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


@st.cache_data
def load_data():
    processed = ROOT / "data" / "processed"
    data = {}

    geo_path = processed / "feature_matrix_geo.parquet"
    base_path = processed / "feature_matrix.parquet"
    path = geo_path if geo_path.exists() else base_path
    data["collisions"] = pd.read_parquet(path)

    ml_dir = processed / "ml"
    if (ml_dir / "model_comparison.csv").exists():
        data["model_comparison"] = pd.read_csv(ml_dir / "model_comparison.csv")
    if (ml_dir / "feature_importance.csv").exists():
        data["feature_importance"] = pd.read_csv(ml_dir / "feature_importance.csv")

    geo_dir = processed / "geo"
    if (geo_dir / "cluster_stats.csv").exists():
        data["clusters"] = pd.read_csv(geo_dir / "cluster_stats.csv")
    if (geo_dir / "top_hotspots.csv").exists():
        data["hotspots"] = pd.read_csv(geo_dir / "top_hotspots.csv")

    return data


@st.cache_resource
def load_model():
    models_dir = ROOT / "mlruns" / "models"
    for name in ["lightgbm", "xgboost", "random_forest"]:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            return joblib.load(path), name
    return None, None


def render_kpi_bar(df):
    total = len(df)
    fatal = (df["collision_severity"] == 1).sum() if "collision_severity" in df.columns else 0
    serious = (df["collision_severity"] == 2).sum() if "collision_severity" in df.columns else 0
    casualties = df["number_of_casualties"].sum() if "number_of_casualties" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Collisions", f"{total:,}")
    c2.metric("Fatal", f"{fatal:,}", delta=f"{fatal/total*100:.1f}%", delta_color="inverse")
    c3.metric("Serious", f"{serious:,}", delta=f"{serious/total*100:.1f}%", delta_color="inverse")
    c4.metric("Slight", f"{total - fatal - serious:,}")
    c5.metric("Total Casualties", f"{casualties:,}")


def page_overview(data):
    st.header("Overview")
    df = data["collisions"]
    render_kpi_bar(df)

    col1, col2 = st.columns(2)

    with col1:
        if "collision_severity_label" in df.columns:
            sev_counts = df["collision_severity_label"].value_counts()
        else:
            sev_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
            sev_counts = df["collision_severity"].map(sev_map).value_counts()
        fig = px.pie(values=sev_counts.values, names=sev_counts.index,
                     title="Severity Distribution",
                     color_discrete_sequence=["#e74c3c", "#f39c12", "#27ae60"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "hour" in df.columns:
            hourly = df.groupby("hour")["collision_index"].count().reset_index()
            hourly.columns = ["Hour", "Collisions"]
            fig = px.bar(hourly, x="Hour", y="Collisions",
                         title="Collisions by Hour of Day",
                         color="Collisions", color_continuous_scale="YlOrRd")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "day_of_week" in df.columns:
            day_map = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed",
                       5: "Thu", 6: "Fri", 7: "Sat"}
            daily = df.groupby("day_of_week")["collision_index"].count().reset_index()
            daily.columns = ["Day", "Collisions"]
            daily["Day_Name"] = daily["Day"].map(day_map)
            fig = px.bar(daily, x="Day_Name", y="Collisions",
                         title="Collisions by Day of Week")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "speed_limit" in df.columns:
            speed = df.groupby("speed_limit").agg(
                count=("collision_index", "count"),
                fatal_pct=("collision_severity", lambda x: (x == 1).mean() * 100),
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=speed["speed_limit"], y=speed["count"],
                                 name="Collisions", yaxis="y"))
            fig.add_trace(go.Scatter(x=speed["speed_limit"], y=speed["fatal_pct"],
                                     name="Fatal %", yaxis="y2", mode="lines+markers",
                                     line=dict(color="red")))
            fig.update_layout(title="Speed Limit: Collision Count vs Fatal %",
                              yaxis=dict(title="Collisions"),
                              yaxis2=dict(title="Fatal %", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)

    if "month" in df.columns:
        monthly = df.groupby("month").agg(
            collisions=("collision_index", "count"),
            fatal=("collision_severity", lambda x: (x == 1).sum()),
        ).reset_index()
        fig = px.line(monthly, x="month", y=["collisions", "fatal"],
                      title="Monthly Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True)


def page_hotspots(data):
    st.header("Geospatial Hotspot Analysis")
    df = data["collisions"]

    if "hotspot_cluster" not in df.columns:
        st.warning("Run Phase 2 (Geospatial Pipeline) first.")
        return

    hotspot_count = (df["hotspot_cluster"] >= 0).sum()
    clusters = data.get("clusters", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hotspot Clusters", f"{len(clusters):,}" if len(clusters) > 0 else "0")
    c2.metric("Collisions in Hotspots", f"{hotspot_count:,}")
    c3.metric("Hotspot %", f"{hotspot_count/len(df)*100:.1f}%")
    if len(clusters) > 0 and "risk_tier" in clusters.columns:
        critical = (clusters["risk_tier"] == "Critical").sum()
        c4.metric("Critical Clusters", f"{critical}")

    sample = df.sample(n=min(5000, len(df)), random_state=42)
    sev_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    sample["severity_label"] = sample["collision_severity"].map(sev_map)
    color_map = {"Fatal": "#e74c3c", "Serious": "#f39c12", "Slight": "#27ae60"}

    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color="severity_label", color_discrete_map=color_map,
        zoom=5.5, center={"lat": 53.5, "lon": -1.5},
        title="Collision Map (5,000 sample)",
        opacity=0.6, size_max=8,
        mapbox_style="carto-positron",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    if len(clusters) > 0 and "risk_tier" in clusters.columns:
        tier_colors = {"Critical": "#e74c3c", "High": "#f39c12",
                       "Medium": "#3498db", "Low": "#27ae60"}
        top_clusters = clusters.head(100)
        fig2 = px.scatter_mapbox(
            top_clusters, lat="center_lat", lon="center_lon",
            color="risk_tier", color_discrete_map=tier_colors,
            size="collision_count",
            hover_data=["risk_score", "fatal_count", "collision_count"],
            zoom=5.5, center={"lat": 53.5, "lon": -1.5},
            title="Top 100 Hotspot Clusters by Risk",
            mapbox_style="carto-positron",
        )
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

    if "hotspots" in data and len(data["hotspots"]) > 0:
        st.subheader("Top 20 Most Dangerous Hotspots")
        st.dataframe(data["hotspots"], use_container_width=True)


def page_conditions(data):
    st.header("Conditions Analysis")
    df = data["collisions"]

    col1, col2 = st.columns(2)
    with col1:
        if "light_conditions_label" in df.columns:
            light_sev = df.groupby("light_conditions_label")["collision_severity"].apply(
                lambda x: (x == 1).mean() * 100
            ).reset_index()
            light_sev.columns = ["Light Condition", "Fatal %"]
            fig = px.bar(light_sev, x="Light Condition", y="Fatal %",
                         title="Fatal Rate by Light Conditions", color="Fatal %",
                         color_continuous_scale="YlOrRd")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "weather_conditions_label" in df.columns:
            weather_sev = df.groupby("weather_conditions_label")["collision_severity"].apply(
                lambda x: (x == 1).mean() * 100
            ).reset_index()
            weather_sev.columns = ["Weather", "Fatal %"]
            fig = px.bar(weather_sev, x="Weather", y="Fatal %",
                         title="Fatal Rate by Weather", color="Fatal %",
                         color_continuous_scale="YlOrRd")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "road_surface_conditions_label" in df.columns:
            surf_sev = df.groupby("road_surface_conditions_label")["collision_severity"].apply(
                lambda x: (x == 1).mean() * 100
            ).reset_index()
            surf_sev.columns = ["Surface", "Fatal %"]
            fig = px.bar(surf_sev, x="Surface", y="Fatal %",
                         title="Fatal Rate by Road Surface", color="Fatal %",
                         color_continuous_scale="YlOrRd")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "danger_score" in df.columns:
            danger = df.groupby("danger_score").agg(
                count=("collision_index", "count"),
                fatal_pct=("collision_severity", lambda x: (x == 1).mean() * 100),
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=danger["danger_score"], y=danger["count"],
                                 name="Collisions"))
            fig.add_trace(go.Scatter(x=danger["danger_score"], y=danger["fatal_pct"],
                                     name="Fatal %", yaxis="y2",
                                     mode="lines+markers", line=dict(color="red")))
            fig.update_layout(title="Danger Score vs Fatality Rate",
                              yaxis=dict(title="Collisions"),
                              yaxis2=dict(title="Fatal %", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)


def page_model(data):
    st.header("Model Performance")

    if "model_comparison" in data:
        st.subheader("Model Comparison")
        comp = data["model_comparison"]
        st.dataframe(comp, use_container_width=True)

        fig = px.bar(comp, x="model", y=["f1_weighted", "f1_macro", "accuracy"],
                     title="Model Performance Comparison", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    if "feature_importance" in data:
        st.subheader("Top 20 Feature Importance")
        fi = data["feature_importance"].head(20)
        fig = px.bar(fi, x="importance_pct", y="feature", orientation="h",
                     title="Feature Importance (%)",
                     color="importance_pct", color_continuous_scale="Viridis")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


def page_predict(data):
    st.header("Predict Collision Severity")
    model, model_name = load_model()

    if model is None:
        st.error("No trained model found. Run Phase 3 first.")
        return

    st.info(f"Using model: **{model_name}**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Road")
        speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
        road_type = st.selectbox("Road Type",
                                  options=[1, 2, 3, 6, 7],
                                  format_func=lambda x: {1: "Roundabout", 2: "One way",
                                    3: "Dual carriageway", 6: "Single carriageway",
                                    7: "Slip road"}[x], index=3)
        road_class = st.selectbox("Road Class",
                                   options=[1, 3, 4, 5, 6],
                                   format_func=lambda x: {1: "Motorway", 3: "A road",
                                     4: "B road", 5: "C road", 6: "Unclassified"}[x],
                                   index=1)
        area = st.selectbox("Area", options=[1, 2],
                             format_func=lambda x: {1: "Urban", 2: "Rural"}[x])

    with col2:
        st.subheader("Conditions")
        light = st.selectbox("Light Conditions",
                              options=[1, 4, 5, 6],
                              format_func=lambda x: {1: "Daylight", 4: "Dark - lights lit",
                                5: "Dark - lights unlit", 6: "Dark - no lighting"}[x])
        weather = st.selectbox("Weather",
                                options=[1, 2, 5, 7],
                                format_func=lambda x: {1: "Fine", 2: "Raining",
                                  5: "Raining + wind", 7: "Fog/mist"}[x])
        surface = st.selectbox("Road Surface",
                                options=[1, 2, 3, 4],
                                format_func=lambda x: {1: "Dry", 2: "Wet",
                                  3: "Snow", 4: "Frost/ice"}[x])
        day = st.selectbox("Day of Week",
                            options=list(range(1, 8)),
                            format_func=lambda x: ["Sun", "Mon", "Tue", "Wed",
                              "Thu", "Fri", "Sat"][x-1], index=2)
        hour = st.slider("Hour of Day", 0, 23, 14)

    with col3:
        st.subheader("Collision")
        num_vehicles = st.slider("Number of Vehicles", 1, 10, 2)
        num_casualties = st.slider("Number of Casualties", 1, 10, 1)
        has_motorcycle = st.checkbox("Motorcycle involved")
        has_hgv = st.checkbox("HGV/Lorry involved")
        has_pedestrian = st.checkbox("Pedestrian involved")
        police_attended = st.selectbox("Police Attended",
                                        options=[1, 2, 3],
                                        format_func=lambda x: {1: "Yes", 2: "No",
                                          3: "Yes (self-reported)"}[x], index=1)

    with col4:
        st.subheader("People")
        driver_age = st.slider("Driver Age", 17, 90, 40)
        casualty_age = st.slider("Casualty Age", 0, 99, 35)
        engine_cc = st.slider("Engine Capacity (cc)", 50, 6000, 1600, step=50)
        vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 8)

    if st.button("Predict Severity", type="primary", use_container_width=True):
        features = build_prediction_features(
            speed_limit, num_vehicles, num_casualties, road_type,
            light, weather, surface, area, day, hour, road_class,
            driver_age, casualty_age, engine_cc, vehicle_age,
            police_attended, has_motorcycle, has_hgv, has_pedestrian,
        )
        try:
            feature_list_path = ROOT / "data" / "processed" / "ml" / "feature_list.csv"
            if feature_list_path.exists():
                expected = pd.read_csv(feature_list_path)["feature"].tolist()
                for col in expected:
                    if col not in features.columns:
                        features[col] = 0
                features = features[expected]

            pred = model.predict(features)[0]
            sev_map = {0: "Fatal", 1: "Serious", 2: "Slight"}
            sev_colors = {0: "red", 1: "orange", 2: "green"}

            st.markdown(f"### Predicted Severity: :{sev_colors[pred]}[**{sev_map[pred]}**]")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                prob_col1.metric("Fatal Probability", f"{proba[0]*100:.1f}%")
                prob_col2.metric("Serious Probability", f"{proba[1]*100:.1f}%")
                prob_col3.metric("Slight Probability", f"{proba[2]*100:.1f}%")

                fig = px.bar(x=["Fatal", "Serious", "Slight"], y=proba * 100,
                             color=["Fatal", "Serious", "Slight"],
                             color_discrete_map={"Fatal": "#e74c3c",
                               "Serious": "#f39c12", "Slight": "#27ae60"},
                             title="Prediction Probabilities")
                fig.update_layout(yaxis_title="Probability %", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")


def build_prediction_features(speed, vehicles, casualties, road_type,
                               light, weather, surface, area, day, hour,
                               road_class, driver_age=40, casualty_age=35,
                               engine_cc=1600, vehicle_age=8,
                               police_attended=2, has_motorcycle=False,
                               has_hgv=False, has_pedestrian=False):
    d = {
        "speed_limit": speed, "number_of_vehicles": vehicles,
        "number_of_casualties": casualties, "road_type": road_type,
        "light_conditions": light, "weather_conditions": weather,
        "road_surface_conditions": surface, "urban_or_rural_area": area,
        "day_of_week": day, "hour": hour, "first_road_class": road_class,
        "is_rush_hour": int(hour in [7, 8, 9, 16, 17, 18]),
        "is_night": int(hour >= 22 or hour <= 5),
        "is_weekend": int(day in [1, 7]),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * day / 7),
        "day_cos": np.cos(2 * np.pi * day / 7),
        "is_high_speed": int(speed >= 60),
        "is_rural": int(area == 2),
        "is_single_carriageway": int(road_type == 6),
        "is_at_junction": 0,
        "is_major_road": int(road_class in [1, 2, 3]),
        "is_dark": int(light in [4, 5, 6, 7]),
        "is_adverse_weather": int(weather in [2, 3, 5, 6, 7]),
        "is_raining": int(weather in [2, 5]),
        "is_wet_surface": int(surface in [2, 3, 4, 5]),
        # Top predictive features — vehicle/casualty/driver
        "did_police_officer_attend_scene_of_accident": police_attended,
        "max_engine_cc": engine_cc,
        "avg_casualty_age": float(casualty_age),
        "min_casualty_age": float(casualty_age),
        "avg_driver_age": float(driver_age),
        "min_driver_age": float(driver_age),
        "avg_vehicle_age": float(vehicle_age),
        "has_motorcycle": int(has_motorcycle),
        "has_hgv": int(has_hgv),
        "has_pedestrian_cycle": int(False),
        "escooter_involved": 0,
        "skid_event": 0,
        "pedestrian_count": int(has_pedestrian),
        "driver_casualty_count": 1,
        "has_child_casualty": int(casualty_age < 16),
        "has_elderly_casualty": int(casualty_age >= 65),
    }

    d["danger_score"] = (d["is_dark"] + d["is_adverse_weather"] + d["is_wet_surface"]
                         + d["is_high_speed"] + d["is_rural"])
    d["dark_rural"] = d["is_dark"] * d["is_rural"]
    d["high_speed_wet"] = d["is_high_speed"] * d["is_wet_surface"]
    d["weekend_night"] = d["is_night"] * d["is_weekend"]
    d["rural_high_speed"] = d["is_rural"] * d["is_high_speed"]
    d["latitude"] = 53.5
    d["longitude"] = -1.5
    d["dist_from_center_km"] = 0
    d["cluster_risk_score"] = d["danger_score"] * 5
    d["is_in_hotspot"] = 0

    return pd.DataFrame([d])


def main():
    st.set_page_config(
        page_title="UK Road Collision Intelligence",
        page_icon="🚗",
        layout="wide",
    )

    st.title("UK Road Collision Intelligence Platform")
    st.caption("2024 DfT Road Casualty Statistics | 100,927 Collisions | 128,272 Casualties")

    data = load_data()

    pages = {
        "Overview": page_overview,
        "Hotspot Map": page_hotspots,
        "Conditions Analysis": page_conditions,
        "Model Performance": page_model,
        "Predict Severity": page_predict,
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page](data)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Source:** DfT Road Casualty Statistics 2024")
    st.sidebar.markdown(f"**Records:** {len(data['collisions']):,}")
    st.sidebar.markdown(f"**Features:** {len(data['collisions'].columns)}")


if __name__ == "__main__":
    main()
