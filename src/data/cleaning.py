"""Data cleaning module - handles missing values, type casting, and code resolution."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.mappings import (
    CASUALTY_TABLE_MAPPINGS,
    COLLISION_TABLE_MAPPINGS,
    VEHICLE_TABLE_MAPPINGS,
)

logger = logging.getLogger(__name__)

MISSING_CODE = -1


def replace_missing_codes(df: pd.DataFrame, code: int = MISSING_CODE) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    replaced = 0
    for col in numeric_cols:
        mask = df[col] == code
        count = mask.sum()
        if count > 0:
            df[col] = df[col].replace(code, np.nan)
            replaced += count
    logger.info(f"Replaced {replaced:,} missing codes ({code}) with NaN")
    return df


def parse_dates(df: pd.DataFrame, date_col: str = "date",
                time_col: str = "time", fmt: str = "%d/%m/%Y") -> pd.DataFrame:
    if date_col not in df.columns:
        return df

    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors="coerce")

    if time_col in df.columns:
        df["datetime"] = pd.to_datetime(
            df[date_col].dt.strftime("%Y-%m-%d") + " " + df[time_col].astype(str),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month
        df["week"] = df["datetime"].dt.isocalendar().week.astype(int)

    logger.info("Parsed date and time columns")
    return df


def apply_label_mappings(df: pd.DataFrame,
                         mappings: dict[str, dict],
                         suffix: str = "_label") -> pd.DataFrame:
    for col, mapping in mappings.items():
        if col in df.columns:
            label_col = col + suffix
            df[label_col] = df[col].map(mapping).fillna("Unknown")
            logger.debug(f"Mapped {col} -> {label_col}")
    return df


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        if col.endswith("_label"):
            continue
        if df[col].dropna().apply(lambda x: x == int(x)).all():
            df[col] = df[col].astype("Int64")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        logger.info(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    return df


def clean_collision(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning collision table...")
    initial_cols = len(df.columns)

    df = replace_missing_codes(df)
    df = parse_dates(df)
    df = apply_label_mappings(df, COLLISION_TABLE_MAPPINGS)
    df = drop_constant_columns(df)

    logger.info(f"Collision cleaning done: {initial_cols} -> {len(df.columns)} columns")
    return df


def clean_casualty(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning casualty table...")
    df = replace_missing_codes(df)
    df = apply_label_mappings(df, CASUALTY_TABLE_MAPPINGS)

    if "age_of_casualty" in df.columns:
        df["age_of_casualty"] = df["age_of_casualty"].clip(lower=0, upper=110)

    df = drop_constant_columns(df)
    logger.info(f"Casualty cleaning done: {len(df.columns)} columns")
    return df


def clean_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning vehicle table...")
    df = replace_missing_codes(df)
    df = apply_label_mappings(df, VEHICLE_TABLE_MAPPINGS)

    if "age_of_driver" in df.columns:
        df["age_of_driver"] = df["age_of_driver"].clip(lower=16, upper=100)

    if "age_of_vehicle" in df.columns:
        df["age_of_vehicle"] = df["age_of_vehicle"].clip(lower=0, upper=80)

    if "engine_capacity_cc" in df.columns:
        df["engine_capacity_cc"] = df["engine_capacity_cc"].clip(lower=0, upper=10000)

    df = drop_constant_columns(df)
    logger.info(f"Vehicle cleaning done: {len(df.columns)} columns")
    return df


def clean_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "collision": clean_collision(tables["collision"].copy()),
        "casualty": clean_casualty(tables["casualty"].copy()),
        "vehicle": clean_vehicle(tables["vehicle"].copy()),
    }
