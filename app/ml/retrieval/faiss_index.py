"""
FAISS Index Builder and Searcher

Builds FAISS IndexFlatIP indexes over L2-normalized embeddings
for fast nearest-neighbor retrieval. Inner product on normalized
vectors equals cosine similarity.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return vectors / norms


def build_faiss_index(
    embeddings: np.ndarray,
    save_path: str,
    normalize: bool = True,
) -> "FaissSearcher":
    """
    Build a FAISS IndexFlatIP index from embeddings and save to disk.

    Args:
        embeddings: numpy array of shape (n_vectors, dim)
        save_path: path to save the index
        normalize: if True, L2-normalize vectors before indexing

    Returns:
        FaissSearcher instance with the built index
    """
    import faiss

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if normalize:
        embeddings = _normalize_vectors(embeddings)

    n_vectors, dim = embeddings.shape
    logger.info(f"Building FAISS IndexFlatIP: {n_vectors} vectors × {dim} dims")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(save_path))
    logger.info(f"Saved FAISS index to {save_path} ({index.ntotal} vectors)")

    searcher = FaissSearcher()
    searcher.index = index
    searcher.dim = dim
    return searcher


def build_all_indexes(model_dir: str = "data/models"):
    """Build all three FAISS indexes: ALS, CBF, and Neural (if available)."""
    model_dir = Path(model_dir)

    # ── ALS Index ────────────────────────────────────────────────────
    als_path = model_dir / "item_embeddings_als.npy"
    if als_path.exists():
        logger.info("Building ALS FAISS index...")
        als_embeddings = np.load(als_path)
        build_faiss_index(als_embeddings, str(model_dir / "faiss_als.index"))
    else:
        logger.warning(f"ALS embeddings not found at {als_path}, skipping")

    # ── CBF Index ────────────────────────────────────────────────────
    cbf_path = model_dir / "item_pca_embeddings.npy"
    if cbf_path.exists():
        logger.info("Building CBF FAISS index...")
        cbf_embeddings = np.load(cbf_path)
        build_faiss_index(cbf_embeddings, str(model_dir / "faiss_cbf.index"))
    else:
        logger.warning(f"CBF embeddings not found at {cbf_path}, skipping")

    # ── Neural Index ─────────────────────────────────────────────────
    neural_path = model_dir / "item_embeddings_neural.npy"
    if neural_path.exists():
        logger.info("Building Neural FAISS index...")
        neural_embeddings = np.load(neural_path)
        build_faiss_index(neural_embeddings, str(model_dir / "faiss_neural.index"))
    else:
        logger.info("Neural embeddings not found yet — index will be built after Two-Tower training")

    logger.info("✓ FAISS index building complete.")


class FaissSearcher:
    """Searcher class for FAISS indexes with load/search/save operations."""

    def __init__(self):
        self.index = None
        self.dim: int = 0

    def load_index(self, path: str) -> "FaissSearcher":
        """Load a FAISS index from disk using memory-mapping."""
        import faiss

        # Use MMAP flag to save RAM
        self.index = faiss.read_index(
            str(path), 
            faiss.IO_FLAG_MMAP
        )
        self.dim = self.index.d
        logger.info(f"Loaded FAISS index (mmap) from {path}: "
                     f"{self.index.ntotal} vectors × {self.dim} dims")
        return self

    def search(
        self, query_vector: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of a single query vector.

        Args:
            query_vector: shape (dim,) or (1, dim)
            k: number of nearest neighbors

        Returns:
            (distances, indices) — each shape (k,)
        """
        assert self.index is not None, "Index not loaded. Call load_index() first."

        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # L2-normalize query for cosine similarity via inner product
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        query = query / norm

        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]

    def batch_search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of multiple query vectors.

        Args:
            query_vectors: shape (n_queries, dim)
            k: number of nearest neighbors

        Returns:
            (distances, indices) — each shape (n_queries, k)
        """
        assert self.index is not None, "Index not loaded. Call load_index() first."

        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

        # L2-normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        queries = queries / norms

        distances, indices = self.index.search(queries, k)
        return distances, indices

    def add_vectors(self, vectors: np.ndarray, normalize: bool = True):
        """Add vectors to the index."""
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if normalize:
            vectors = _normalize_vectors(vectors)

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim

        self.index.add(vectors)
        logger.info(f"Added {len(vectors)} vectors. Total: {self.index.ntotal}")

    def save_index(self, path: str):
        """Save the index to disk."""
        import faiss

        assert self.index is not None, "No index to save."
        faiss.write_index(self.index, str(path))
        logger.info(f"Saved FAISS index to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_all_indexes()
