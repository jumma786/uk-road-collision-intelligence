"""Geospatial clustering module - DBSCAN hotspot detection on collision coordinates."""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def run_dbscan(df: pd.DataFrame, eps_km: float = 0.5,
               min_samples: int = 5) -> pd.DataFrame:
    logger.info(f"Running DBSCAN: eps={eps_km}km, min_samples={min_samples}")

    coords = df[["latitude", "longitude"]].dropna()
    valid_idx = coords.index
    logger.info(f"Valid coordinates: {len(valid_idx):,} / {len(df):,}")

    coords_rad = np.radians(coords.values)
    eps_rad = eps_km / EARTH_RADIUS_KM

    dbscan = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    labels = dbscan.fit_predict(coords_rad)

    df = df.copy()
    df["hotspot_cluster"] = -1
    df.loc[valid_idx, "hotspot_cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    n_clustered = (labels != -1).sum()

    logger.info(
        f"DBSCAN results: {n_clusters} clusters, "
        f"{n_clustered:,} clustered ({n_clustered / len(valid_idx) * 100:.1f}%), "
        f"{n_noise:,} noise points"
    )
    return df


def compute_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df["hotspot_cluster"] >= 0].copy()
    if len(clustered) == 0:
        logger.warning("No clusters found")
        return pd.DataFrame()

    stats = clustered.groupby("hotspot_cluster").agg(
        collision_count=("collision_index", "count"),
        center_lat=("latitude", "mean"),
        center_lon=("longitude", "mean"),
        fatal_count=("collision_severity", lambda x: (x == 1).sum()),
        serious_count=("collision_severity", lambda x: (x == 2).sum()),
        slight_count=("collision_severity", lambda x: (x == 3).sum()),
        avg_casualties=("number_of_casualties", "mean"),
        total_casualties=("number_of_casualties", "sum"),
        avg_speed_limit=("speed_limit", "mean"),
        pct_dark=("is_dark", "mean"),
        pct_wet=("is_wet_surface", "mean"),
        pct_rural=("is_rural", "mean"),
        avg_danger_score=("danger_score", "mean"),
        lat_spread=("latitude", lambda x: x.max() - x.min()),
        lon_spread=("longitude", lambda x: x.max() - x.min()),
    ).reset_index()

    stats["fatality_rate"] = stats["fatal_count"] / stats["collision_count"]
    stats["serious_rate"] = (
        (stats["fatal_count"] + stats["serious_count"]) / stats["collision_count"]
    )

    stats["risk_score"] = (
        stats["fatality_rate"] * 50
        + stats["serious_rate"] * 30
        + stats["avg_danger_score"] / 5 * 10
        + np.minimum(stats["collision_count"] / stats["collision_count"].max(), 1) * 10
    ).round(2)

    stats = stats.sort_values("risk_score", ascending=False).reset_index(drop=True)
    stats["risk_rank"] = range(1, len(stats) + 1)

    logger.info(
        f"Cluster stats computed: {len(stats)} clusters, "
        f"top risk score: {stats['risk_score'].iloc[0]:.2f}"
    )
    return stats


def assign_risk_tier(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    stats["risk_tier"] = pd.qcut(
        stats["risk_score"],
        q=[0, 0.5, 0.8, 0.95, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )
    tier_counts = stats["risk_tier"].value_counts()
    for tier, count in tier_counts.items():
        logger.info(f"  {tier}: {count} clusters")
    return stats


def merge_cluster_features(df: pd.DataFrame,
                           cluster_stats: pd.DataFrame) -> pd.DataFrame:
    if len(cluster_stats) == 0:
        df["cluster_risk_score"] = 0
        df["cluster_risk_tier"] = "No_Cluster"
        df["cluster_collision_count"] = 0
        return df

    merge_cols = [
        "hotspot_cluster", "risk_score", "risk_tier",
        "collision_count", "fatality_rate", "serious_rate",
    ]
    available = [c for c in merge_cols if c in cluster_stats.columns]
    merge_df = cluster_stats[available].rename(columns={
        "risk_score": "cluster_risk_score",
        "risk_tier": "cluster_risk_tier",
        "collision_count": "cluster_collision_count",
        "fatality_rate": "cluster_fatality_rate",
        "serious_rate": "cluster_serious_rate",
    })

    df = df.merge(merge_df, on="hotspot_cluster", how="left")

    df["cluster_risk_score"] = df["cluster_risk_score"].fillna(0)
    if hasattr(df["cluster_risk_tier"].dtype, "categories"):
        df["cluster_risk_tier"] = df["cluster_risk_tier"].cat.add_categories("No_Cluster")
    df["cluster_risk_tier"] = df["cluster_risk_tier"].fillna("No_Cluster")
    df["cluster_collision_count"] = df["cluster_collision_count"].fillna(0)
    if "cluster_fatality_rate" in df.columns:
        df["cluster_fatality_rate"] = df["cluster_fatality_rate"].fillna(0)
    else:
        df["cluster_fatality_rate"] = 0
    if "cluster_serious_rate" in df.columns:
        df["cluster_serious_rate"] = df["cluster_serious_rate"].fillna(0)
    else:
        df["cluster_serious_rate"] = 0
    df["is_in_hotspot"] = (df["hotspot_cluster"] >= 0).astype(int)

    logger.info(f"Merged cluster features: {df['is_in_hotspot'].sum():,} collisions in hotspots")
    return df
