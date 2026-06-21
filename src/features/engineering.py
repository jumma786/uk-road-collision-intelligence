"""Feature engineering module - creates ML-ready features from cleaned data."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df.columns and "time" in df.columns:
        df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour

    if "hour" in df.columns:
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
        )
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([1, 7]).astype(int)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    if "month" in df.columns:
        df["season"] = pd.cut(
            df["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Autumn"],
        )
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Added temporal features")
    return df


def add_road_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    if "speed_limit" in df.columns:
        df["speed_risk"] = pd.cut(
            df["speed_limit"],
            bins=[0, 20, 40, 60, 100],
            labels=["Low", "Medium", "High", "Very_High"],
        )
        df["is_high_speed"] = (df["speed_limit"] >= 60).astype(int)

    if "urban_or_rural_area" in df.columns:
        df["is_rural"] = (df["urban_or_rural_area"] == 2).astype(int)

    if "road_type" in df.columns:
        df["is_single_carriageway"] = (df["road_type"] == 6).astype(int)

    if "junction_detail" in df.columns:
        df["is_at_junction"] = (
            (df["junction_detail"] != 0) & df["junction_detail"].notna()
        ).astype(int)

    if "first_road_class" in df.columns:
        df["is_major_road"] = df["first_road_class"].isin([1, 2, 3]).astype(int)

    logger.info("Added road risk features")
    return df


def add_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    if "light_conditions" in df.columns:
        df["is_dark"] = df["light_conditions"].isin([4, 5, 6, 7]).astype(int)

    if "weather_conditions" in df.columns:
        df["is_adverse_weather"] = df["weather_conditions"].isin(
            [2, 3, 5, 6, 7]
        ).astype(int)
        df["is_raining"] = df["weather_conditions"].isin([2, 5]).astype(int)

    if "road_surface_conditions" in df.columns:
        df["is_wet_surface"] = df["road_surface_conditions"].isin([2, 3, 4, 5]).astype(int)

    danger_score = 0
    if "is_dark" in df.columns:
        danger_score = danger_score + df["is_dark"]
    if "is_adverse_weather" in df.columns:
        danger_score = danger_score + df["is_adverse_weather"]
    if "is_wet_surface" in df.columns:
        danger_score = danger_score + df["is_wet_surface"]
    if "is_high_speed" in df.columns:
        danger_score = danger_score + df["is_high_speed"]
    if "is_rural" in df.columns:
        danger_score = danger_score + df["is_rural"]
    if isinstance(danger_score, pd.Series):
        df["danger_score"] = danger_score

    logger.info("Added condition features")
    return df


def add_vehicle_agg_features(
    collision_df: pd.DataFrame, vehicle_df: pd.DataFrame
) -> pd.DataFrame:
    if vehicle_df is None or len(vehicle_df) == 0:
        return collision_df

    agg = vehicle_df.groupby("collision_index").agg(
        avg_driver_age=("age_of_driver", "mean"),
        min_driver_age=("age_of_driver", "min"),
        max_engine_cc=("engine_capacity_cc", "max"),
        avg_vehicle_age=("age_of_vehicle", "mean"),
        has_motorcycle=("vehicle_type", lambda x: int(x.isin([1, 2, 3, 4, 5, 23, 97]).any())),
        has_hgv=("vehicle_type", lambda x: int(x.isin([20, 21, 22]).any())),
        has_pedestrian_cycle=("vehicle_type", lambda x: int(x.isin([1]).any())),
        escooter_involved=("escooter_flag", "max"),
        skid_event=("skidding_and_overturning", lambda x: int((x > 0).any())),
    ).reset_index()

    collision_df = collision_df.merge(agg, on="collision_index", how="left")
    logger.info(f"Added {len(agg.columns) - 1} vehicle aggregate features")
    return collision_df


def add_casualty_agg_features(
    collision_df: pd.DataFrame, casualty_df: pd.DataFrame
) -> pd.DataFrame:
    if casualty_df is None or len(casualty_df) == 0:
        return collision_df

    agg = casualty_df.groupby("collision_index").agg(
        avg_casualty_age=("age_of_casualty", "mean"),
        min_casualty_age=("age_of_casualty", "min"),
        pedestrian_count=("casualty_class", lambda x: (x == 3).sum()),
        driver_casualty_count=("casualty_class", lambda x: (x == 1).sum()),
        has_child_casualty=("age_of_casualty", lambda x: int((x < 16).any())),
        has_elderly_casualty=("age_of_casualty", lambda x: int((x >= 65).any())),
        fatal_casualty_count=("casualty_severity", lambda x: (x == 1).sum()),
        serious_casualty_count=("casualty_severity", lambda x: (x == 2).sum()),
    ).reset_index()

    collision_df = collision_df.merge(agg, on="collision_index", how="left")
    logger.info(f"Added {len(agg.columns) - 1} casualty aggregate features")
    return collision_df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    uk_center_lat, uk_center_lon = 53.5, -1.5
    df["dist_from_center_km"] = np.sqrt(
        ((df["latitude"] - uk_center_lat) * 111) ** 2
        + ((df["longitude"] - uk_center_lon) * 111 * np.cos(np.radians(53.5))) ** 2
    )

    df["lat_bin"] = pd.cut(df["latitude"], bins=20, labels=False)
    df["lon_bin"] = pd.cut(df["longitude"], bins=20, labels=False)

    logger.info("Added geographic features")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "is_dark" in df.columns and "is_rural" in df.columns:
        df["dark_rural"] = df["is_dark"] * df["is_rural"]

    if "is_high_speed" in df.columns and "is_wet_surface" in df.columns:
        df["high_speed_wet"] = df["is_high_speed"] * df["is_wet_surface"]

    if "is_night" in df.columns and "is_weekend" in df.columns:
        df["weekend_night"] = df["is_night"] * df["is_weekend"]

    if "is_rural" in df.columns and "is_high_speed" in df.columns:
        df["rural_high_speed"] = df["is_rural"] * df["is_high_speed"]

    logger.info("Added interaction features")
    return df


def engineer_all_features(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    logger.info("Starting feature engineering pipeline...")
    collision = tables["collision"].copy()
    casualty = tables.get("casualty")
    vehicle = tables.get("vehicle")

    collision = add_temporal_features(collision)
    collision = add_road_risk_features(collision)
    collision = add_condition_features(collision)
    collision = add_geo_features(collision)
    collision = add_interaction_features(collision)

    if vehicle is not None:
        collision = add_vehicle_agg_features(collision, vehicle)
    if casualty is not None:
        collision = add_casualty_agg_features(collision, casualty)

    logger.info(
        f"Feature engineering complete: {len(collision.columns)} columns, "
        f"{len(collision):,} rows"
    )
    return collision
