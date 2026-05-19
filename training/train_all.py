"""
Training Orchestrator

Single script that runs all training steps in order:
1. Check data / ingest
2. Build features
3. Build interaction matrix
4. Train ALS → build ALS FAISS index
5. Encode items with SBERT → build CBF FAISS index
6. Train Two-Tower → build Neural FAISS index
7. Train LightGBM ranker

Prints elapsed time for each step.
"""

import sys
import time
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, func, *args, **kwargs):
    """Run a training step with timing."""
    logger.info(f"{'='*60}")
    logger.info(f"STEP: {name}")
    logger.info(f"{'='*60}")
    start = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"✓ {name} completed in {elapsed:.1f}s")
        return elapsed
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"✗ {name} failed after {elapsed:.1f}s: {e}")
        raise


def step_check_data():
    """Step 1: Check if data exists, run ingestion if needed."""
    data_dir = PROJECT_ROOT / "data" / "processed"
    matrix_path = data_dir / "interaction_matrix.npz"

    if matrix_path.exists():
        logger.info("Interaction matrix already exists, skipping ingestion")
        return

    logger.info("No interaction matrix found. Running data pipeline...")

    # Check if DB has data
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/recsys")
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM interactions")).scalar()
            if count == 0:
                logger.info("DB is empty — running ingestion scripts...")
                subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "training" / "ingest_retailrocket.py")],
                    check=True,
                )
            else:
                logger.info(f"DB has {count} interactions")
    except Exception as e:
        logger.warning(f"Could not check DB: {e}")


def step_build_features():
    """Step 2: Build user and item features."""
    logger.info("Running user feature engineering...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "app" / "features" / "user_features.py")],
        check=True,
    )

    logger.info("Running item feature engineering...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "app" / "features" / "item_features.py")],
        check=True,
    )


def step_build_matrix():
    """Step 3: Build interaction matrix."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "training" / "build_matrix.py")],
        check=True,
    )


def step_train_als():
    """Step 4: Train ALS model."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.ml.als.train import train_als
    train_als()


def step_build_als_faiss():
    """Step 4b: Build ALS FAISS index."""
    from app.ml.retrieval.faiss_index import build_faiss_index
    import numpy as np

    model_dir = PROJECT_ROOT / "data" / "models"
    embeddings = np.load(model_dir / "item_embeddings_als.npy")
    build_faiss_index(embeddings, str(model_dir / "faiss_als.index"))


def step_encode_sbert():
    """Step 5: Encode items with SBERT + PCA."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.ml.cbf.encoder import encode_items
    encode_items()


def step_build_cbf_faiss():
    """Step 5b: Build CBF FAISS index."""
    from app.ml.retrieval.faiss_index import build_faiss_index
    import numpy as np

    model_dir = PROJECT_ROOT / "data" / "models"
    embeddings = np.load(model_dir / "item_pca_embeddings.npy")
    build_faiss_index(embeddings, str(model_dir / "faiss_cbf.index"))


def step_train_two_tower():
    """Step 6: Train Two-Tower neural network (also builds FAISS index)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.ml.neural.train import train_two_tower
    train_two_tower()


def step_train_sasrec():
    """Step 6b: Train SASRec model."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "app" / "ml" / "retrieval" / "train_sasrec.py")],
        check=True,
    )


def step_train_ranker():
    """Step 7: Train LightGBM ranker."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.ml.ranking.train import train_ranker
    train_ranker()


def main():
    """Run the full training pipeline."""
    total_start = time.time()
    timings = {}

    sys.path.insert(0, str(PROJECT_ROOT))

    steps = [
        ("1. Check/Ingest Data", step_check_data),
        ("2. Extract & Build Interaction Matrix", step_build_matrix),
        ("3. Build Features", step_build_features),
        ("4a. Train ALS", step_train_als),
        ("4b. Build ALS FAISS Index", step_build_als_faiss),
        ("5a. Encode Items (SBERT)", step_encode_sbert),
        ("5b. Build CBF FAISS Index", step_build_cbf_faiss),
        ("6a. Train Two-Tower + Neural FAISS", step_train_two_tower),
        ("6b. Train SASRec", step_train_sasrec),
        ("7. Train LightGBM Ranker", step_train_ranker),
    ]

    for name, func in steps:
        try:
            elapsed = run_step(name, func)
            timings[name] = elapsed
        except Exception as e:
            logger.error(f"Pipeline failed at step '{name}': {e}")
            break

    total_time = time.time() - total_start

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING PIPELINE SUMMARY")
    logger.info(f"{'='*60}")
    for step_name, elapsed in timings.items():
        logger.info(f"  {step_name}: {elapsed:.1f}s")
    logger.info(f"  {'─'*50}")
    logger.info(f"  Total: {total_time:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
