"""Master pipeline - runs all phases sequentially."""

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("UK ROAD COLLISION INTELLIGENCE PLATFORM")
    logger.info("RUNNING ALL PHASES")
    logger.info("=" * 60)

    # Phase 1: ETL
    logger.info("\n>>> PHASE 1: ETL PIPELINE")
    from src.data.pipeline import run_pipeline
    etl_result = run_pipeline()

    # Phase 2: Geospatial
    logger.info("\n>>> PHASE 2: GEOSPATIAL ANALYSIS")
    from src.geo.pipeline import run_geo_pipeline
    geo_result = run_geo_pipeline()

    # Phase 3: ML
    logger.info("\n>>> PHASE 3: ML TRAINING")
    from src.models.pipeline import run_ml_pipeline
    ml_result = run_ml_pipeline()

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("ALL PHASES COMPLETE")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Feature matrix: {len(etl_result['feature_matrix']):,} rows")
    logger.info(f"Hotspot clusters: {len(geo_result['cluster_stats'])}")
    logger.info(f"Best model: {ml_result['best_model_name']} "
                f"(F1={ml_result['comparison'].iloc[0]['f1_weighted']:.4f})")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  Dashboard:  streamlit run src/dashboard/app.py")
    logger.info("  API:        uvicorn src.serving.app:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
