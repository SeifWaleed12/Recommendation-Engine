import sys
import pandas as pd
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    input_path = data_dir / "interactions_clean.parquet"
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info("Loading full interactions dataset...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} interactions.")

    logger.info("Sorting by timestamp and creating 80/20 temporal split...")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Train split: {len(train_df)} interactions (80%)")
    logger.info(f"Test split:  {len(test_df)} interactions (20%)")
    
    if not train_df.empty and not test_df.empty:
        logger.info(f"Train date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        logger.info(f"Test date range:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    train_path = data_dir / "train_interactions.parquet"
    test_path = data_dir / "test_interactions.parquet"
    
    logger.info(f"Saving train split to {train_path}...")
    train_df.to_parquet(train_path, index=False)
    
    logger.info(f"Saving test split to {test_path}...")
    test_df.to_parquet(test_path, index=False)
    
    logger.info("Temporal split completed successfully.")

if __name__ == "__main__":
    main()
