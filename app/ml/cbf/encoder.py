"""
SBERT Item Encoder

Encodes all items using SentenceTransformer('all-MiniLM-L6-v2'),
then reduces dimensionality with PCA to 128 dimensions.
"""

import sys
import time
import json
import pickle
import random
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"


def encode_items():
    """Encode all items with SBERT and apply PCA dimensionality reduction."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from sqlalchemy import create_engine

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load item index map ──────────────────────────────────────────
    with open(data_dir / "item_idx_map.json", "r") as f:
        item_idx_map = json.load(f)  # external_id → int index

    n_items = len(item_idx_map)
    # Reverse map: int index → external_id
    idx_to_ext = {v: k for k, v in item_idx_map.items()}
    logger.info(f"Loaded item_idx_map: {n_items} items")

    # ── Load items from DB ───────────────────────────────────────────
    logger.info("Fetching items from database...")
    engine = create_engine(DB_URL)

    import pandas as pd
    items_df = pd.read_sql(
        "SELECT external_id, title, description, category, brand, price FROM items",
        engine,
    )
    logger.info(f"Loaded {len(items_df)} items from DB")

    # Create a lookup by external_id
    items_lookup = items_df.set_index("external_id").to_dict("index")

    # ── Build text strings aligned to item indices ───────────────────
    logger.info("Building text representations...")
    texts = []
    for idx in range(n_items):
        ext_id = idx_to_ext.get(idx, "")
        item = items_lookup.get(ext_id, {})

        title = str(item.get("title", "")) if item.get("title") else ""
        description = str(item.get("description", "")) if item.get("description") else ""
        category = str(item.get("category", "")) if item.get("category") else ""
        brand = str(item.get("brand", "")) if item.get("brand") else ""
        price = str(item.get("price", "")) if item.get("price") else ""

        # Truncate description to 200 chars
        desc_short = description[:200]
        
        if not title and not description:
            # RetailRocket items usually only have category and price
            text = f"Retail product in category {category} priced at ${price}."
        else:
            text = f"{title}. Category: {category}. Brand: {brand}. Price: ${price}. {desc_short}".strip()

        # Fallback for items with no text at all
        if not text or text == ". Category: . Brand: . Price: $.":
            text = f"retail product {ext_id}"

        texts.append(text)

    logger.info(f"Built {len(texts)} text representations")

    # ── Encode with SBERT ────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SentenceTransformer('all-MiniLM-L6-v2') on {device}...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    logger.info(f"Encoding {n_items} items in batches of 64...")
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    encoding_time = time.time() - start_time
    logger.info(f"Encoding completed in {encoding_time:.1f}s")
    logger.info(f"Raw embeddings shape: {embeddings.shape}")  # (n_items, 384)

    # Save raw SBERT embeddings
    np.save(model_dir / "sbert_item_embeddings.npy", embeddings)
    logger.info("Saved: sbert_item_embeddings.npy")

    # ── Apply PCA to 128 dimensions ──────────────────────────────────
    logger.info("Applying PCA (384 → 128)...")
    pca = PCA(n_components=128, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings).astype(np.float32)

    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA explained variance ratio (128 components): {explained_variance:.4f}")
    logger.info(f"PCA embeddings shape: {embeddings_pca.shape}")  # (n_items, 128)

    # Save PCA embeddings
    np.save(model_dir / "item_pca_embeddings.npy", embeddings_pca)
    logger.info("Saved: item_pca_embeddings.npy")

    # Save PCA model
    with open(model_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    logger.info("Saved: pca_model.pkl")

    logger.info(f"✓ SBERT encoding complete. Time: {encoding_time:.1f}s, "
                f"PCA variance: {explained_variance:.4f}")

    return embeddings, embeddings_pca


if __name__ == "__main__":
    encode_items()
