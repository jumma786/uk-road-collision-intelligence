"""Main ETL pipeline - orchestrates ingestion, cleaning, validation, and feature engineering."""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.ingestion import load_all_tables, load_config
from src.data.cleaning import clean_all
from src.data.validation import validate_all, generate_profile
from src.features.engineering import engineer_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_processed(tables: dict[str, pd.DataFrame], output_dir: str,
                   fmt: str = "parquet") -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = {}

    for name, df in tables.items():
        if fmt == "parquet":
            path = output_path / f"{name}_cleaned.parquet"
            df.to_parquet(path, index=False)
        else:
            path = output_path / f"{name}_cleaned.csv"
            df.to_csv(path, index=False)
        saved[name] = str(path)
        logger.info(f"Saved {name}: {path} ({len(df):,} rows)")

    return saved


def save_feature_matrix(df: pd.DataFrame, output_dir: str,
                        fmt: str = "parquet") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        path = output_path / "feature_matrix.parquet"
        df.to_parquet(path, index=False)
    else:
        path = output_path / "feature_matrix.csv"
        df.to_csv(path, index=False)

    logger.info(f"Saved feature matrix: {path} ({len(df):,} rows, {len(df.columns)} cols)")
    return str(path)


def run_pipeline(config_path: str = "configs/config.yaml") -> dict:
    start = time.time()
    logger.info("=" * 60)
    logger.info("UK ROAD COLLISION PLATFORM - ETL PIPELINE")
    logger.info("=" * 60)

    config = load_config(config_path)
    output_dir = config["paths"]["processed_data_dir"]
    output_fmt = config["etl"]["output_format"]

    # Stage 1: Ingest
    logger.info("[1/5] INGESTION")
    tables = load_all_tables(config)

    # Stage 2: Validate raw data
    logger.info("[2/5] RAW DATA VALIDATION")
    raw_results = validate_all(tables)
    failures = [r for r in raw_results if not r.is_valid]
    if failures:
        for f in failures:
            logger.error(f.summary())
        logger.error("Raw data validation failed - aborting pipeline")
        sys.exit(1)

    # Stage 3: Clean
    logger.info("[3/5] CLEANING")
    cleaned = clean_all(tables)
    saved_cleaned = save_processed(cleaned, output_dir, output_fmt)

    # Stage 4: Validate cleaned data
    logger.info("[4/5] CLEANED DATA VALIDATION")
    clean_results = validate_all(cleaned)

    # Stage 5: Feature engineering
    logger.info("[5/5] FEATURE ENGINEERING")
    feature_matrix = engineer_all_features(cleaned)
    feature_path = save_feature_matrix(feature_matrix, output_dir, output_fmt)

    # Profile
    profile = generate_profile(feature_matrix, "feature_matrix")
    profile_path = Path(output_dir) / "feature_matrix_profile.csv"
    profile.to_csv(profile_path)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Raw tables: {sum(len(t) for t in tables.values()):,} total records")
    logger.info(f"Feature matrix: {len(feature_matrix):,} rows x {len(feature_matrix.columns)} cols")
    logger.info(f"Output: {output_dir}/")
    logger.info("=" * 60)

    return {
        "tables": cleaned,
        "feature_matrix": feature_matrix,
        "feature_path": feature_path,
        "cleaned_paths": saved_cleaned,
        "validation": clean_results,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    run_pipeline()
