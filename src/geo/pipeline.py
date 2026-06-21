"""Geospatial pipeline - runs DBSCAN clustering and generates hotspot analysis."""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from src.geo.clustering import (
    run_dbscan,
    compute_cluster_stats,
    assign_risk_tier,
    merge_cluster_features,
)
from src.geo.analysis import (
    top_hotspots_summary,
    hotspot_risk_profile,
    police_force_hotspot_summary,
    temporal_hotspot_pattern,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_geo_pipeline(config_path: str = "configs/config.yaml") -> dict:
    start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 2: GEOSPATIAL HOTSPOT ANALYSIS")
    logger.info("=" * 60)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["paths"]["processed_data_dir"])
    geo_config = config["geo"]

    # Load feature matrix from Phase 1
    logger.info("[1/5] Loading feature matrix...")
    feature_path = processed_dir / "feature_matrix.parquet"
    if not feature_path.exists():
        logger.error(f"Feature matrix not found at {feature_path}. Run Phase 1 first.")
        sys.exit(1)

    df = pd.read_parquet(feature_path)
    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Run DBSCAN
    logger.info("[2/5] Running DBSCAN clustering...")
    df = run_dbscan(
        df,
        eps_km=geo_config["dbscan_eps_km"],
        min_samples=geo_config["dbscan_min_samples"],
    )

    # Compute cluster statistics
    logger.info("[3/5] Computing cluster statistics...")
    cluster_stats = compute_cluster_stats(df)
    cluster_stats = assign_risk_tier(cluster_stats)

    # Merge cluster features back to main dataframe
    logger.info("[4/5] Merging cluster features...")
    df = merge_cluster_features(df, cluster_stats)

    # Generate analysis reports
    logger.info("[5/5] Generating analysis reports...")
    top_hotspots = top_hotspots_summary(cluster_stats, top_n=20)
    risk_profile = hotspot_risk_profile(cluster_stats)
    pf_summary = police_force_hotspot_summary(df, cluster_stats)
    temporal = temporal_hotspot_pattern(df)

    # Save outputs
    output_dir = processed_dir / "geo"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(processed_dir / "feature_matrix_geo.parquet", index=False)
    cluster_stats.to_csv(output_dir / "cluster_stats.csv", index=False)
    top_hotspots.to_csv(output_dir / "top_hotspots.csv", index=False)
    if len(pf_summary) > 0:
        pf_summary.to_csv(output_dir / "police_force_hotspots.csv", index=False)
    if len(temporal) > 0:
        temporal.to_csv(output_dir / "temporal_hotspot_pattern.csv", index=False)

    logger.info(f"Saved feature matrix with geo features: {len(df.columns)} columns")
    logger.info(f"Saved analysis reports to {output_dir}/")

    # Print summary
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Clusters found: {len(cluster_stats)}")
    logger.info(f"Collisions in hotspots: {df['is_in_hotspot'].sum():,}")

    if risk_profile:
        logger.info(f"Total fatalities in clusters: {risk_profile['total_fatalities_in_clusters']}")
        logger.info(f"Avg cluster size: {risk_profile['avg_cluster_size']}")
        logger.info(f"Max risk score: {risk_profile['max_risk_score']}")
        if risk_profile.get("tier_distribution"):
            for tier, count in risk_profile["tier_distribution"].items():
                logger.info(f"  {tier}: {count} clusters")

    logger.info(f"Feature matrix: {len(df):,} rows x {len(df.columns)} cols")
    logger.info("=" * 60)

    if len(top_hotspots) > 0:
        logger.info("\nTOP 10 MOST DANGEROUS HOTSPOTS:")
        logger.info("-" * 40)
        for _, row in top_hotspots.head(10).iterrows():
            logger.info(
                f"  Rank {int(row['risk_rank'])}: "
                f"Cluster {int(row['hotspot_cluster'])} | "
                f"{int(row['collision_count'])} collisions | "
                f"{int(row['fatal_count'])} fatal | "
                f"Risk: {row['risk_score']:.1f} ({row['risk_tier']}) | "
                f"({row['center_lat']:.4f}, {row['center_lon']:.4f})"
            )

    return {
        "feature_matrix": df,
        "cluster_stats": cluster_stats,
        "top_hotspots": top_hotspots,
        "risk_profile": risk_profile,
        "police_force_summary": pf_summary,
        "temporal_pattern": temporal,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    run_geo_pipeline()
