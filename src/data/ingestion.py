"""Data ingestion module - loads raw DfT CSV files."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_collision_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading collision data from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df):,} collision records with {len(df.columns)} columns")
    return df


def load_casualty_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading casualty data from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df):,} casualty records with {len(df.columns)} columns")
    return df


def load_vehicle_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading vehicle data from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df):,} vehicle records with {len(df.columns)} columns")
    return df


def load_all_tables(config: dict) -> dict[str, pd.DataFrame]:
    paths = config["paths"]
    base = Path(paths["raw_data_dir"])

    tables = {
        "collision": load_collision_data(str(base / paths["raw_collision"])),
        "casualty": load_casualty_data(str(base / paths["raw_casualty"])),
        "vehicle": load_vehicle_data(str(base / paths["raw_vehicle"])),
    }

    logger.info(
        f"All tables loaded: "
        f"collision={len(tables['collision']):,}, "
        f"casualty={len(tables['casualty']):,}, "
        f"vehicle={len(tables['vehicle']):,}"
    )
    return tables
