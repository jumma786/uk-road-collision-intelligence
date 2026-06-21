"""ML preprocessing - feature selection, encoding, and train/test split."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

DROP_COLUMNS = [
    "collision_index", "collision_ref_no", "date", "time", "datetime",
    "lsoa_of_accident_location", "local_authority_ons_district",
    "local_authority_highway", "local_authority_highway_current",
    "collision_injury_based",
    "collision_adjusted_severity_serious", "collision_adjusted_severity_slight",
    "enhanced_severity_collision",
    # Leakage columns — derived from or directly encoding the target
    "fatal_casualty_count", "serious_casualty_count",
    "casualty_severity", "enhanced_casualty_severity",
    "collision_adjusted_severity",
    "cluster_fatality_rate", "cluster_serious_rate",
]

LABEL_COLUMNS = [c for c in [] if c.endswith("_label")]

TARGET = "collision_severity"


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    label_cols = [c for c in df.columns if c.endswith("_label")]
    drop_cols = [c for c in DROP_COLUMNS + label_cols if c in df.columns]
    drop_cols = [c for c in drop_cols if c != TARGET]

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")

    y = df[TARGET].copy()
    # XGBoost requires 0-indexed classes
    y = y - 1
    X = df.drop(columns=drop_cols + [TARGET], errors="ignore")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=[np.number])

    inf_cols = X.columns[X.isin([np.inf, -np.inf]).any()].tolist()
    if inf_cols:
        X[inf_cols] = X[inf_cols].replace([np.inf, -np.inf], np.nan)

    null_pct = X.isnull().sum() / len(X)
    high_null_cols = null_pct[null_pct > 0.6].index.tolist()
    if high_null_cols:
        logger.info(f"Dropping {len(high_null_cols)} high-null columns (>60%): {high_null_cols}")
        X = X.drop(columns=high_null_cols)

    X = X.fillna(X.median())

    logger.info(f"Feature selection: {len(X.columns)} features, {len(y):,} samples")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    logger.info(
        f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Train target dist: {y_train.value_counts(normalize=True).round(3).to_dict()}"
    )
    return X_train, X_test, y_train, y_test
