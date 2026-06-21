"""Data validation module - schema checks, quality assertions, and profiling."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    table_name: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        lines = [f"[{status}] {self.table_name}"]
        lines += [f"  + {p}" for p in self.passed]
        lines += [f"  X {f}" for f in self.failed]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_collision(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("collision")

    if len(df) > 0:
        result.passed.append(f"Non-empty: {len(df):,} rows")
    else:
        result.failed.append("Table is empty")

    if "collision_index" in df.columns:
        dupes = df["collision_index"].duplicated().sum()
        if dupes == 0:
            result.passed.append("collision_index is unique")
        else:
            result.failed.append(f"collision_index has {dupes:,} duplicates")

    if "collision_severity" in df.columns:
        valid_sevs = {1, 2, 3}
        actual = set(df["collision_severity"].dropna().unique())
        if actual.issubset(valid_sevs):
            result.passed.append(f"Severity values valid: {actual}")
        else:
            result.failed.append(f"Invalid severity values: {actual - valid_sevs}")

    if "latitude" in df.columns and "longitude" in df.columns:
        lat_ok = df["latitude"].between(49, 61).all()
        lon_ok = df["longitude"].between(-8, 2).all()
        if lat_ok and lon_ok:
            result.passed.append("GPS coordinates within UK bounds")
        else:
            result.warnings.append("Some GPS coordinates outside UK bounds")

    if "speed_limit" in df.columns:
        valid_speeds = {20, 30, 40, 50, 60, 70}
        actual = set(df["speed_limit"].dropna().unique())
        if actual.issubset(valid_speeds):
            result.passed.append("Speed limits valid")
        else:
            result.warnings.append(f"Unusual speed limits: {actual - valid_speeds}")

    null_pct = df.isnull().sum() / len(df) * 100
    high_null = null_pct[null_pct > 50]
    if len(high_null) == 0:
        result.passed.append("No columns with >50% nulls")
    else:
        for col, pct in high_null.items():
            result.warnings.append(f"{col}: {pct:.1f}% null")

    return result


def validate_casualty(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("casualty")

    if len(df) > 0:
        result.passed.append(f"Non-empty: {len(df):,} rows")
    else:
        result.failed.append("Table is empty")

    if "collision_index" in df.columns:
        result.passed.append(f"Covers {df['collision_index'].nunique():,} collisions")

    if "age_of_casualty" in df.columns:
        age_range = (df["age_of_casualty"].min(), df["age_of_casualty"].max())
        if 0 <= age_range[0] and age_range[1] <= 110:
            result.passed.append(f"Casualty ages in range: {age_range}")
        else:
            result.warnings.append(f"Unusual age range: {age_range}")

    if "casualty_severity" in df.columns:
        valid = {1, 2, 3}
        actual = set(df["casualty_severity"].dropna().unique())
        if actual.issubset(valid):
            result.passed.append("Casualty severity values valid")
        else:
            result.failed.append(f"Invalid casualty severity: {actual - valid}")

    return result


def validate_vehicle(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("vehicle")

    if len(df) > 0:
        result.passed.append(f"Non-empty: {len(df):,} rows")
    else:
        result.failed.append("Table is empty")

    if "collision_index" in df.columns:
        result.passed.append(f"Covers {df['collision_index'].nunique():,} collisions")

    if "age_of_driver" in df.columns:
        valid_age = df["age_of_driver"].dropna()
        if len(valid_age) > 0:
            age_range = (valid_age.min(), valid_age.max())
            result.passed.append(f"Driver ages in range: {age_range}")

    if "engine_capacity_cc" in df.columns:
        valid_cc = df["engine_capacity_cc"].dropna()
        if len(valid_cc) > 0 and valid_cc.max() <= 10000:
            result.passed.append(f"Engine capacity valid (max {valid_cc.max():.0f}cc)")

    return result


def validate_join_integrity(tables: dict[str, pd.DataFrame]) -> ValidationResult:
    result = ValidationResult("join_integrity")

    collision_ids = set(tables["collision"]["collision_index"].unique())
    casualty_ids = set(tables["casualty"]["collision_index"].unique())
    vehicle_ids = set(tables["vehicle"]["collision_index"].unique())

    cas_orphans = casualty_ids - collision_ids
    veh_orphans = vehicle_ids - collision_ids

    if len(cas_orphans) == 0:
        result.passed.append("All casualty records link to a collision")
    else:
        result.warnings.append(f"{len(cas_orphans)} casualty records have no matching collision")

    if len(veh_orphans) == 0:
        result.passed.append("All vehicle records link to a collision")
    else:
        result.warnings.append(f"{len(veh_orphans)} vehicle records have no matching collision")

    cas_coverage = len(casualty_ids & collision_ids) / len(collision_ids) * 100
    veh_coverage = len(vehicle_ids & collision_ids) / len(collision_ids) * 100
    result.passed.append(f"Casualty coverage: {cas_coverage:.1f}% of collisions")
    result.passed.append(f"Vehicle coverage: {veh_coverage:.1f}% of collisions")

    return result


def validate_all(tables: dict[str, pd.DataFrame]) -> list[ValidationResult]:
    results = [
        validate_collision(tables["collision"]),
        validate_casualty(tables["casualty"]),
        validate_vehicle(tables["vehicle"]),
        validate_join_integrity(tables),
    ]
    for r in results:
        logger.info(r.summary())
    return results


def generate_profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    profile = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.count(),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
        "unique_pct": (df.nunique() / len(df) * 100).round(2),
    })

    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        stats = numeric.describe().T[["mean", "std", "min", "max"]]
        profile = profile.join(stats)

    return profile
