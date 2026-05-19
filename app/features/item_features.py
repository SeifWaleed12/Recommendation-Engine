import sys
import json
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.models.schema import FeatureStore, EntityType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"

def main():
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    logger.info("Fetching data for item features from TRAIN split...")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    train_path = project_root / "data" / "processed" / "train_interactions.parquet"
    
    if not train_path.exists():
        logger.error("train_interactions.parquet not found.")
        return
        
    df = pd.read_parquet(train_path)
    
    # We need category and created_at from the items table
    items_query = "SELECT external_id, category, created_at FROM items"
    items_df = pd.read_sql(items_query, engine)
    
    df = df.merge(items_df, left_on="item_ext_id", right_on="external_id", how="inner")
    
    if df.empty:
        logger.warning("No interactions found. Exiting.")
        return

    logger.info(f"Loaded {len(df)} interactions. Computing features...")

    features_list = []
    now = datetime.utcnow()
    
    # Calculate global metrics
    item_stats = df.groupby('item_ext_id').agg(
        global_view_count=('event_type', lambda x: (x == 'view').sum()),
        global_purchase_count=('event_type', lambda x: (x == 'purchase').sum()),
        avg_interaction_weight=('weight', 'mean'),
        unique_user_count=('user_id', 'nunique'),
        created_at=('created_at', 'first'),
        category=('category', 'first')
    ).reset_index()

    # Derived metrics
    item_stats['conversion_rate'] = item_stats['global_purchase_count'] / (item_stats['global_view_count'] + 1)
    item_stats['days_since_created'] = (now - item_stats['created_at']).dt.days.fillna(0).astype(int)
    
    # Category Rank
    # Rank items within their category based on purchase count
    item_stats['category_rank'] = item_stats.groupby('category')['global_purchase_count'].rank(method='dense', ascending=False)
    
    records = item_stats.to_dict('records')
    for row in records:
        feature_val = {
            "global_view_count": int(row['global_view_count']),
            "global_purchase_count": int(row['global_purchase_count']),
            "conversion_rate": float(row['conversion_rate']),
            "avg_interaction_weight": float(row['avg_interaction_weight']),
            "unique_user_count": int(row['unique_user_count']),
            "days_since_created": int(row['days_since_created']),
            "category_rank": int(row['category_rank']) if pd.notna(row['category_rank']) else None
        }
        
        features_list.append({
            "entity_type": EntityType.item.name,
            "entity_id": str(row['item_ext_id']),
            "feature_name": "item_profile_v1",
            "feature_value_json": feature_val,
            "computed_at": now,
            "version": 1
        })

    logger.info(f"Inserting {len(features_list)} item features...")
    
    from sqlalchemy import text
    session.execute(text("DELETE FROM feature_store WHERE entity_type = 'item' AND feature_name = 'item_profile_v1'"))
    session.commit()
    
    for i in range(0, len(features_list), 100000):
        session.bulk_insert_mappings(FeatureStore, features_list[i:i+100000])
        session.commit()
        
    logger.info("Item features computation complete.")

if __name__ == "__main__":
    main()
