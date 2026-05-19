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

    logger.info("Fetching data for user features from TRAIN split...")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    train_path = project_root / "data" / "processed" / "train_interactions.parquet"
    
    if not train_path.exists():
        logger.error("train_interactions.parquet not found.")
        return
        
    df = pd.read_parquet(train_path)
    
    items_query = "SELECT external_id, category, price FROM items"
    items_df = pd.read_sql(items_query, engine)
    
    df = df.merge(items_df, left_on="item_ext_id", right_on="external_id", how="inner")
    
    if df.empty:
        logger.warning("No interactions found. Exiting.")
        return

    logger.info(f"Loaded {len(df)} interactions. Computing features...")

    # Calculate global metrics for the price percentiles
    # This might be slightly inaccurate if we have items with no interactions, but good enough
    price_bins = df['price'].dropna().quantile([i/10.0 for i in range(11)]).values.copy()
    price_bins[0] = -1 # to ensure minimum price is included
    price_bins[-1] += 1 # ensure max price is included
    
    df['price_bucket'] = pd.cut(df['price'], bins=price_bins, labels=False, include_lowest=True)

    features_list = []
    
    now = datetime.utcnow()
    
    # Fast Vectorized Aggregations
    user_stats = df.groupby('user_ext_id').agg(
        interaction_count=('item_id', 'size'),
        purchase_count=('event_type', lambda x: (x == 'purchase').sum()),
        preferred_event=('event_type', lambda x: x.mode().iloc[0] if not x.mode().empty else 'view'),
        price_percentile=('price_bucket', lambda x: int(x.mode().iloc[0]) if not x.mode().empty else None),
        last_active=('timestamp', 'max')
    ).reset_index()

    user_stats['last_active_days_ago'] = (now - user_stats['last_active']).dt.days

    # Map directly to list of dictionaries
    records = user_stats.to_dict('records')
    features_list = []
    
    for row in records:
        feature_value = {
            "interaction_count": int(row['interaction_count']),
            "purchase_count": int(row['purchase_count']),
            "preferred_event": row['preferred_event'],
            "price_percentile": int(row['price_percentile']) if pd.notna(row['price_percentile']) else None,
            "last_active_days_ago": int(row['last_active_days_ago']) if pd.notna(row['last_active_days_ago']) else None,
            "is_active": True
        }
        
        features_list.append({
            "entity_type": EntityType.user,
            "entity_id": row['user_ext_id'],
            "feature_name": "user_profile_v1",
            "feature_value_json": feature_value,
            "computed_at": now,
            "version": 1
        })
        
    logger.info(f"Inserting {len(features_list)} user features...")
    
    # Delete old features of this type to avoid duplicates on rerun
    from sqlalchemy import text
    session.execute(text("DELETE FROM feature_store WHERE entity_type = 'user' AND feature_name = 'user_profile_v1'"))
    session.commit()
    
    for i in range(0, len(features_list), 100000):
        session.bulk_insert_mappings(FeatureStore, features_list[i:i+100000])
        session.commit()
        
    logger.info("User features computation complete.")

if __name__ == "__main__":
    main()
