"""Geospatial analysis module - hotspot summaries and geographic insights."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def top_hotspots_summary(cluster_stats: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if len(cluster_stats) == 0:
        return pd.DataFrame()

    top = cluster_stats.head(top_n).copy()
    summary = top[[
        "risk_rank", "hotspot_cluster", "collision_count", "total_casualties",
        "fatal_count", "serious_count", "fatality_rate", "serious_rate",
        "risk_score", "risk_tier", "center_lat", "center_lon",
        "avg_speed_limit", "pct_dark", "pct_wet", "pct_rural",
    ]].copy()

    summary["fatality_rate"] = (summary["fatality_rate"] * 100).round(1)
    summary["serious_rate"] = (summary["serious_rate"] * 100).round(1)
    summary["pct_dark"] = (summary["pct_dark"] * 100).round(1)
    summary["pct_wet"] = (summary["pct_wet"] * 100).round(1)
    summary["pct_rural"] = (summary["pct_rural"] * 100).round(1)

    return summary


def hotspot_risk_profile(cluster_stats: pd.DataFrame) -> dict:
    if len(cluster_stats) == 0:
        return {}

    return {
        "total_clusters": len(cluster_stats),
        "total_collisions_in_clusters": int(cluster_stats["collision_count"].sum()),
        "total_fatalities_in_clusters": int(cluster_stats["fatal_count"].sum()),
        "avg_cluster_size": round(cluster_stats["collision_count"].mean(), 1),
        "max_cluster_size": int(cluster_stats["collision_count"].max()),
        "avg_risk_score": round(cluster_stats["risk_score"].mean(), 2),
        "max_risk_score": round(cluster_stats["risk_score"].max(), 2),
        "pct_clusters_rural": round(
            (cluster_stats["pct_rural"] > 0.5).mean() * 100, 1
        ),
        "pct_clusters_dark_heavy": round(
            (cluster_stats["pct_dark"] > 0.3).mean() * 100, 1
        ),
        "tier_distribution": cluster_stats["risk_tier"].value_counts().to_dict()
        if "risk_tier" in cluster_stats.columns
        else {},
    }


def police_force_hotspot_summary(
    df: pd.DataFrame, cluster_stats: pd.DataFrame
) -> pd.DataFrame:
    if "police_force" not in df.columns or len(cluster_stats) == 0:
        return pd.DataFrame()

    hotspot_df = df[df["hotspot_cluster"] >= 0].copy()
    if len(hotspot_df) == 0:
        return pd.DataFrame()

    pf_stats = hotspot_df.groupby("police_force").agg(
        hotspot_collisions=("collision_index", "count"),
        hotspot_clusters=("hotspot_cluster", "nunique"),
        hotspot_fatalities=("collision_severity", lambda x: (x == 1).sum()),
        avg_risk_score=("cluster_risk_score", "mean")
        if "cluster_risk_score" in hotspot_df.columns
        else ("collision_index", "count"),
    ).reset_index()

    total_by_force = df.groupby("police_force")["collision_index"].count().reset_index()
    total_by_force.columns = ["police_force", "total_collisions"]

    pf_stats = pf_stats.merge(total_by_force, on="police_force", how="left")
    pf_stats["hotspot_pct"] = (
        pf_stats["hotspot_collisions"] / pf_stats["total_collisions"] * 100
    ).round(1)

    return pf_stats.sort_values("hotspot_collisions", ascending=False)


def temporal_hotspot_pattern(df: pd.DataFrame) -> pd.DataFrame:
    hotspot_df = df[df.get("is_in_hotspot", 0) == 1].copy()
    non_hotspot = df[df.get("is_in_hotspot", 0) == 0].copy()

    if len(hotspot_df) == 0 or "hour" not in df.columns:
        return pd.DataFrame()

    hotspot_hourly = hotspot_df.groupby("hour")["collision_index"].count()
    non_hotspot_hourly = non_hotspot.groupby("hour")["collision_index"].count()

    hourly = pd.DataFrame({
        "hour": range(24),
        "hotspot_count": [hotspot_hourly.get(h, 0) for h in range(24)],
        "non_hotspot_count": [non_hotspot_hourly.get(h, 0) for h in range(24)],
    })

    hourly["hotspot_pct"] = (
        hourly["hotspot_count"]
        / (hourly["hotspot_count"] + hourly["non_hotspot_count"])
        * 100
    ).round(1)

    return hourly
