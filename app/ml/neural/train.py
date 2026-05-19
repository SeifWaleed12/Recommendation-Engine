"""
Two-Tower Neural Network — Training Script

Trains the Two-Tower model with BPR loss, saves user/item tower weights,
generates item embeddings, and builds a FAISS index.
"""

import sys
import time
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.

    loss = -log(sigmoid(pos_score - neg_score)).mean()
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()


def train_two_tower():
    """Train the Two-Tower model and generate item embeddings."""
    from app.ml.neural.towers import TwoTowerModel
    from app.ml.neural.dataset import create_datasets
    from app.ml.retrieval.faiss_index import build_faiss_index

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Hyperparameters ──────────────────────────────────────────────
    hparams = {
        "batch_size": 512,
        "epochs": 10,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "n_negatives": 4,
        "feature_dim": 16,
    }
    logger.info(f"Hyperparameters: {hparams}")

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Create Datasets ──────────────────────────────────────────────
    logger.info("Creating datasets...")
    data_dir = str(project_root / "data" / "processed")
    train_dataset, val_dataset, n_users, n_items = create_datasets(
        data_dir=data_dir,
        n_negatives=hparams["n_negatives"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"n_users={n_users}, n_items={n_items}")

    # ── MLflow Setup ─────────────────────────────────────────────────
    mlflow_available = False
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("two_tower_training")
        mlflow_available = True
    except Exception:
        logger.warning("MLflow not available, skipping tracking")

    # ── Model ────────────────────────────────────────────────────────
    model = TwoTowerModel(n_users, n_items, feature_dim=hparams["feature_dim"])
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )

    # ── Training Loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    start_time = time.time()

    if mlflow_available:
        try:
            mlflow.start_run(run_name="two_tower_training")
            mlflow.log_params(hparams)
        except Exception:
            mlflow_available = False

    for epoch in range(hparams["epochs"]):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hparams['epochs']}")
        for batch in pbar:
            user_idx = batch["user_idx"].to(device)
            user_feats = batch["user_feats"].to(device)
            pos_item_idx = batch["pos_item_idx"].to(device)
            pos_item_feats = batch["pos_item_feats"].to(device)
            neg_item_idxs = batch["neg_item_idxs"].to(device)  # (batch, n_neg)
            neg_item_feats = batch["neg_item_feats"].to(device)  # (batch, n_neg, 16)

            # Positive scores
            pos_scores = model(user_idx, user_feats, pos_item_idx, pos_item_feats)

            # Negative scores — iterate over negatives
            batch_size = user_idx.size(0)
            n_neg = neg_item_idxs.size(1)

            total_loss = torch.tensor(0.0, device=device)
            for j in range(n_neg):
                neg_idx = neg_item_idxs[:, j]
                neg_feats = neg_item_feats[:, j, :]
                neg_scores = model(user_idx, user_feats, neg_idx, neg_feats)
                total_loss += bpr_loss(pos_scores, neg_scores)

            total_loss = total_loss / n_neg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_losses.append(total_loss.item())
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        avg_train_loss = np.mean(train_losses)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                user_idx = batch["user_idx"].to(device)
                user_feats = batch["user_feats"].to(device)
                pos_item_idx = batch["pos_item_idx"].to(device)
                pos_item_feats = batch["pos_item_feats"].to(device)
                neg_item_idxs = batch["neg_item_idxs"].to(device)
                neg_item_feats = batch["neg_item_feats"].to(device)

                pos_scores = model(user_idx, user_feats, pos_item_idx, pos_item_feats)

                total_loss = torch.tensor(0.0, device=device)
                n_neg = neg_item_idxs.size(1)
                for j in range(n_neg):
                    neg_idx = neg_item_idxs[:, j]
                    neg_feats = neg_item_feats[:, j, :]
                    neg_scores = model(user_idx, user_feats, neg_idx, neg_feats)
                    total_loss += bpr_loss(pos_scores, neg_scores)

                val_losses.append((total_loss / n_neg).item())

        avg_val_loss = np.mean(val_losses)

        logger.info(
            f"Epoch {epoch+1}/{hparams['epochs']} — "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if mlflow_available:
            try:
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            except Exception:
                pass

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.user_tower.state_dict(), model_dir / "two_tower_user.pt")
            torch.save(model.item_tower.state_dict(), model_dir / "two_tower_item.pt")
            logger.info(f"  → Saved best model (val_loss={best_val_loss:.4f})")

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")

    # ── Generate Item Embeddings ─────────────────────────────────────
    logger.info("Generating item embeddings from trained ItemTower...")

    # Load best model weights
    model.item_tower.load_state_dict(torch.load(model_dir / "two_tower_item.pt", weights_only=True))
    model.eval()

    all_item_embeddings = []
    batch_size = 256

    with torch.no_grad():
        for start in tqdm(range(0, n_items, batch_size), desc="Encoding items"):
            end = min(start + batch_size, n_items)
            item_idxs = torch.arange(start, end, dtype=torch.long, device=device)
            item_feats = torch.zeros(end - start, hparams["feature_dim"],
                                     dtype=torch.float32, device=device)

            embeds = model.item_tower(item_idxs, item_feats)
            all_item_embeddings.append(embeds.cpu().numpy())

    item_embeddings_neural = np.vstack(all_item_embeddings)
    logger.info(f"Neural item embeddings shape: {item_embeddings_neural.shape}")

    np.save(model_dir / "item_embeddings_neural.npy", item_embeddings_neural)
    logger.info("Saved: item_embeddings_neural.npy")

    # ── Build FAISS Index ────────────────────────────────────────────
    logger.info("Building FAISS index for neural embeddings...")
    build_faiss_index(
        item_embeddings_neural,
        str(model_dir / "faiss_neural.index"),
        normalize=True,
    )

    # ── Log final metrics ────────────────────────────────────────────
    if mlflow_available:
        try:
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("training_time_seconds", training_time)
            mlflow.log_metric("n_users", n_users)
            mlflow.log_metric("n_items", n_items)
            mlflow.end_run()
        except Exception:
            pass

    logger.info("✓ Two-Tower training complete.")
    return model


if __name__ == "__main__":
    train_two_tower()
