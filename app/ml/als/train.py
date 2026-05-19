"""
iALS Collaborative Filtering — Training Script

Trains an implicit ALS model on the user-item interaction matrix,
extracts user and item embeddings, and saves everything to data/models/.
"""

import sys
import time
import json
import pickle
import random
import logging
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)


def train_als():
    """Train iALS model and save embeddings."""
    import implicit

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Hyperparameters ──────────────────────────────────────────────
    hparams = {
        "factors": 128,
        "regularization": 0.01,
        "iterations": 20,
        "alpha": 40,
    }

    # Detect GPU availability
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    hparams["use_gpu"] = use_gpu
    logger.info(f"Hyperparameters: {hparams}")
    logger.info(f"GPU available: {use_gpu}")

    # ── Load Data ────────────────────────────────────────────────────
    logger.info("Loading interaction matrix...")
    interaction_matrix = load_npz(data_dir / "interaction_matrix.npz").tocsr()
    n_users, n_items = interaction_matrix.shape
    logger.info(f"Matrix shape: ({n_users}, {n_items}), NNZ: {interaction_matrix.nnz}")

    # Apply confidence weighting: C = 1 + alpha * R
    # implicit expects the raw confidence matrix; it applies alpha internally
    # when alpha is passed to the constructor

    # ── MLflow Setup (fail gracefully) ───────────────────────────────
    mlflow_available = False
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("als_training")
        mlflow_available = True
        logger.info("MLflow tracking enabled at localhost:5001")
    except Exception as e:
        logger.warning(f"MLflow not available, skipping tracking: {e}")

    # ── Train Model ──────────────────────────────────────────────────
    logger.info("Initializing ALS model...")
    try:
        model = implicit.als.AlternatingLeastSquares(
            factors=hparams["factors"],
            regularization=hparams["regularization"],
            iterations=hparams["iterations"],
            use_gpu=hparams["use_gpu"],
        )
    except ValueError as e:
        if "No CUDA extension" in str(e):
            logger.warning("GPU requested but CUDA extension not found. Falling back to CPU...")
            model = implicit.als.AlternatingLeastSquares(
                factors=hparams["factors"],
                regularization=hparams["regularization"],
                iterations=hparams["iterations"],
                use_gpu=False,
            )
        else:
            raise e

    logger.info("Training ALS model...")
    start_time = time.time()

    # implicit's fit method takes item-user matrix (transpose of user-item)
    # Actually in implicit >= 0.6, fit() takes user-item directly
    model.fit(interaction_matrix, show_progress=True)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")

    # ── Extract Embeddings ───────────────────────────────────────────
    logger.info("Extracting embeddings...")

    # In implicit >= 0.7, factors are accessed via model.user_factors / model.item_factors
    user_factors = np.array(model.user_factors)
    item_factors = np.array(model.item_factors)

    # If GPU was used, factors might be on GPU — move to CPU
    if hasattr(user_factors, 'to_numpy'):
        user_factors = user_factors.to_numpy()
    if hasattr(item_factors, 'to_numpy'):
        item_factors = item_factors.to_numpy()

    logger.info(f"User embeddings shape: {user_factors.shape}")
    logger.info(f"Item embeddings shape: {item_factors.shape}")

    assert user_factors.shape == (n_users, hparams["factors"]), \
        f"Expected ({n_users}, {hparams['factors']}), got {user_factors.shape}"
    assert item_factors.shape == (n_items, hparams["factors"]), \
        f"Expected ({n_items}, {hparams['factors']}), got {item_factors.shape}"

    # ── Save Artifacts ───────────────────────────────────────────────
    logger.info("Saving model and embeddings...")

    with open(model_dir / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    np.save(model_dir / "user_embeddings_als.npy", user_factors)
    np.save(model_dir / "item_embeddings_als.npy", item_factors)

    # Copy index maps to model directory for self-contained deployment
    import shutil
    shutil.copy2(data_dir / "user_idx_map.json", model_dir / "user_idx_map.json")
    shutil.copy2(data_dir / "item_idx_map.json", model_dir / "item_idx_map.json")

    logger.info(f"Saved: als_model.pkl, user_embeddings_als.npy, item_embeddings_als.npy")
    logger.info(f"Copied: user_idx_map.json, item_idx_map.json → data/models/")

    # ── Log to MLflow ────────────────────────────────────────────────
    if mlflow_available:
        try:
            with mlflow.start_run(run_name="als_training"):
                mlflow.log_params(hparams)
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("n_users", n_users)
                mlflow.log_metric("n_items", n_items)
                mlflow.log_metric("nnz", interaction_matrix.nnz)
                logger.info("Logged metrics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    logger.info("✓ ALS training complete.")
    return model, user_factors, item_factors


if __name__ == "__main__":
    train_als()
