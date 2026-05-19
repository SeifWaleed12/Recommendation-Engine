import sys
import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from sqlalchemy import create_engine
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"

def main():
    engine = create_engine(DB_URL)

    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching interactions from database...")
    
    query = """
        SELECT 
            i.user_id,
            u.external_id as user_ext_id,
            i.item_id,
            it.external_id as item_ext_id,
            i.event_type,
            i.timestamp,
            i.weight
        FROM interactions i
        JOIN users u ON i.user_id = u.id
        JOIN items it ON i.item_id = it.id
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        logger.warning("No interactions found in the database. Cannot build matrix.")
        return

    logger.info(f"Loaded {len(df)} interactions. Creating mappings from ALL data...")
    
    # Cast UUID columns to strings for parquet compatibility
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df.to_parquet(data_dir / "interactions_clean.parquet", index=False)

    # Create mappings from the full dataset (important for cold-start benchmark)
    unique_users = df['user_ext_id'].unique()
    unique_items = df['item_ext_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    logger.info("Saving mappings to JSON...")
    with open(data_dir / "user_idx_map.json", "w") as f:
        json.dump(user_to_idx, f)
        
    with open(data_dir / "item_idx_map.json", "w") as f:
        json.dump(item_to_idx, f)
        
    # We must call the temporal split script here, before trying to load train_interactions
    import subprocess
    import sys
    logger.info("Running temporal split...")
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "temporal_split.py")], check=True)

    logger.info("Loading TRAIN split to build CSR matrix...")
    train_df = pd.read_parquet(data_dir / "train_interactions.parquet")
    
    # Group by user and item to get the sum of weights (confidence) from train data
    matrix_df = train_df.groupby(['user_ext_id', 'item_ext_id'])['weight'].sum().reset_index()
    matrix_df.rename(columns={'weight': 'confidence'}, inplace=True)
    
    logger.info("Building CSR matrix...")
    # Map to integer indices
    row_idx = matrix_df['user_ext_id'].map(user_to_idx).values
    col_idx = matrix_df['item_ext_id'].map(item_to_idx).values
    data = matrix_df['confidence'].values
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    interaction_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(n_users, n_items))
    
    # Save the matrix
    from scipy.sparse import save_npz
    save_npz(data_dir / "interaction_matrix.npz", interaction_matrix)
    
    # Calculate sparsity
    nnz = interaction_matrix.nnz
    total_elements = n_users * n_items
    sparsity = (1.0 - (nnz / total_elements)) * 100
    
    logger.info(f"Matrix built successfully.")
    logger.info(f"Shape: {interaction_matrix.shape}")
    logger.info(f"Non-zero elements (nnz): {nnz}")
    logger.info(f"Sparsity: {sparsity:.4f}%")

if __name__ == "__main__":
    main()
