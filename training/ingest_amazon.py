import sys
import json
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
from tqdm import tqdm
from datetime import datetime

# Adjust Python path to be able to import app modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.models.schema import User, Item, Interaction, EventType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"

def main(limit=1000000): # Process a sample of 1M rows by default
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    amazon_file = data_dir / "Electronics.jsonl"

    if not amazon_file.exists():
        logger.error(f"File not found: {amazon_file}")
        return

    logger.info(f"Parsing Amazon Reviews (limit: {limit} rows)...")

    rating_weight_map = {
        1.0: 0.5,
        2.0: 1.0,
        3.0: 2.0,
        4.0: 3.5,
        5.0: 5.0
    }

    user_mappings = {}
    item_mappings = {}
    interaction_mappings = []

    # Count lines for tqdm (only if we need a progress bar for the full file, but we are sampling)
    # We will just show progress without a total if limit is huge, or with total=limit
    
    with open(amazon_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=limit, desc="Processing rows")):
            if i >= limit:
                break
                
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = row.get('title')
            # Skip items with no title
            if not title:
                continue

            user_ext_id = f"amz_{row.get('user_id')}"
            item_ext_id = f"amz_{row.get('asin')}"
            rating = float(row.get('rating', 0))
            weight = rating_weight_map.get(rating, 1.0)
            text = row.get('text')
            ts = row.get('timestamp')
            
            try:
                dt = datetime.utcfromtimestamp(ts / 1000.0) if ts else datetime.utcnow()
            except (ValueError, TypeError):
                dt = datetime.utcnow()

            if user_ext_id not in user_mappings:
                user_mappings[user_ext_id] = {
                    "external_id": user_ext_id,
                    "created_at": datetime.utcnow()
                }

            if item_ext_id not in item_mappings:
                item_mappings[item_ext_id] = {
                    "external_id": item_ext_id,
                    "title": title,
                    "description": text,
                    "is_active": True,
                    "created_at": datetime.utcnow()
                }
                
            # We will bulk insert these later after getting DB IDs, but for now just collect
            interaction_mappings.append({
                "user_ext_id": user_ext_id,
                "item_ext_id": item_ext_id,
                "event_type": EventType.purchase.name, # Reviews imply purchase/interaction
                "timestamp": dt,
                "weight": weight
            })

    # Bulk insert Users
    user_vals = list(user_mappings.values())
    logger.info(f"Inserting {len(user_vals)} Amazon users...")
    for i in range(0, len(user_vals), 100000):
        session.bulk_insert_mappings(User, user_vals[i:i+100000])
        session.commit()
        
    # Bulk insert Items
    item_vals = list(item_mappings.values())
    logger.info(f"Inserting {len(item_vals)} Amazon items...")
    for i in range(0, len(item_vals), 100000):
        session.bulk_insert_mappings(Item, item_vals[i:i+100000])
        session.commit()

    # Fetch DB IDs to link interactions
    logger.info("Fetching UUIDs for users and items...")
    # Fetching specifically amz_ prefixed to avoid pulling the whole DB
    from sqlalchemy import text
    user_db_map = dict(session.execute(text("SELECT external_id, id FROM users WHERE external_id LIKE 'amz_%'")).fetchall())
    item_db_map = dict(session.execute(text("SELECT external_id, id FROM items WHERE external_id LIKE 'amz_%'")).fetchall())

    # Map interactions
    final_interactions = []
    logger.info(f"Mapping {len(interaction_mappings)} interactions...")
    for im in interaction_mappings:
        u_id = user_db_map.get(im['user_ext_id'])
        i_id = item_db_map.get(im['item_ext_id'])
        
        if u_id and i_id:
            final_interactions.append({
                "user_id": u_id,
                "item_id": i_id,
                "event_type": im['event_type'],
                "timestamp": im['timestamp'],
                "weight": im['weight']
            })

    logger.info("Inserting Amazon interactions...")
    for i in range(0, len(final_interactions), 100000):
        session.bulk_insert_mappings(Interaction, final_interactions[i:i+100000])
        session.commit()

    logger.info("Amazon ingestion complete.")

if __name__ == "__main__":
    main(limit=500000) # Sample 500k rows
