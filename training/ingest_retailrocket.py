import sys
import pandas as pd
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

def main():
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    events_path = data_dir / "events.csv"
    item_prop1_path = data_dir / "item_properties_part1.csv"
    item_prop2_path = data_dir / "item_properties_part2.csv"

    if not events_path.exists():
        logger.error(f"File not found: {events_path}")
        return

    logger.info("Parsing item properties...")
    # Read both parts
    props1 = pd.read_csv(item_prop1_path)
    props2 = pd.read_csv(item_prop2_path)
    props = pd.concat([props1, props2], ignore_index=True)

    # Keep only the latest property value for each itemid and property type
    props.sort_values(by="timestamp", ascending=False, inplace=True)
    props.drop_duplicates(subset=["itemid", "property"], keep="first", inplace=True)

    # Filter to only the properties we need to save memory and time
    props = props[props["property"].isin(["categoryid", "available", "790"])]

    # Pivot to get properties as columns
    items_df = props.pivot(index="itemid", columns="property", values="value").reset_index()

    logger.info("Parsing events...")
    events_df = pd.read_csv(events_path)

    # Unique users and items
    unique_users = events_df['visitorid'].unique()
    unique_items_from_events = events_df['itemid'].unique()
    unique_items_from_props = items_df['itemid'].unique()
    
    all_item_ids = set(unique_items_from_events).union(set(unique_items_from_props))

    logger.info(f"Inserting {len(unique_users)} users...")
    user_mappings = []
    # Using a dict to simulate set for fast lookups
    inserted_users = set()
    
    for i, uid in enumerate(tqdm(unique_users, desc="Users")):
        str_uid = str(uid)
        if str_uid not in inserted_users:
            user_mappings.append({
                "external_id": str_uid,
                "created_at": datetime.utcnow()
            })
            inserted_users.add(str_uid)
        
        if len(user_mappings) >= 100000:
            session.bulk_insert_mappings(User, user_mappings)
            session.commit()
            user_mappings = []

    if user_mappings:
        session.bulk_insert_mappings(User, user_mappings)
        session.commit()

    logger.info(f"Inserting {len(all_item_ids)} items...")
    item_mappings = []
    
    # Pre-calculate a lookup dictionary for items
    # categoryid -> category, available -> is_active, 790 -> price, title -> title
    items_dict = items_df.set_index('itemid').to_dict('index')

    for i, iid in enumerate(tqdm(list(all_item_ids), desc="Items")):
        str_iid = str(iid)
        
        title = None
        category = None
        is_active = True
        price = None
        
        if iid in items_dict:
            props_row = items_dict[iid]
            title = props_row.get('title')
            category = props_row.get('categoryid')
            
            avail = props_row.get('available')
            if pd.notna(avail):
                is_active = str(avail) == '1'
                
            p = props_row.get('790')
            if pd.notna(p):
                try:
                    price = float(str(p).strip().replace('n', '').split(' ')[0])
                except (ValueError, TypeError):
                    price = None
        
        item_mappings.append({
            "external_id": str_iid,
            "title": str(title) if pd.notna(title) else None,
            "category": str(category) if pd.notna(category) else None,
            "is_active": is_active,
            "price": price,
            "created_at": datetime.utcnow()
        })
        
        if len(item_mappings) >= 100000:
            session.bulk_insert_mappings(Item, item_mappings)
            session.commit()
            item_mappings = []

    if item_mappings:
        session.bulk_insert_mappings(Item, item_mappings)
        session.commit()

    logger.info("Fetching UUIDs for users and items...")
    # Fetch DB IDs to link interactions
    # Doing this in chunks to avoid blowing up memory if the DB is huge
    user_db_map = {}
    for batch in pd.read_sql("SELECT id, external_id FROM users", engine, chunksize=100000):
        user_db_map.update(dict(zip(batch['external_id'], batch['id'])))
        
    item_db_map = {}
    for batch in pd.read_sql("SELECT id, external_id FROM items", engine, chunksize=100000):
        item_db_map.update(dict(zip(batch['external_id'], batch['id'])))

    logger.info(f"Inserting {len(events_df)} interactions...")
    interaction_mappings = []
    
    event_weight_map = {
        'view': 1.0,
        'addtocart': 3.0,
        'transaction': 5.0
    }
    
    event_enum_map = {
        'view': EventType.view,
        'addtocart': EventType.add_to_cart,
        'transaction': EventType.purchase
    }

    # Process interactions
    for idx, row in tqdm(events_df.iterrows(), total=len(events_df), desc="Interactions"):
        user_ext_id = str(row['visitorid'])
        item_ext_id = str(row['itemid'])
        
        user_id = user_db_map.get(user_ext_id)
        item_id = item_db_map.get(item_ext_id)
        
        if not user_id or not item_id:
            continue
            
        event = str(row['event']).lower()
        weight = event_weight_map.get(event, 1.0)
        e_type = event_enum_map.get(event, EventType.view)
        
        ts = pd.to_datetime(row['timestamp'], unit='ms')
        session_id = str(row['transactionid']) if pd.notna(row['transactionid']) else None
        
        interaction_mappings.append({
            "user_id": user_id,
            "item_id": item_id,
            "event_type": e_type.name, # Use name for enum mapping in SQLAlchemy bulk insert
            "timestamp": ts,
            "session_id": session_id,
            "weight": weight
        })
        
        if len(interaction_mappings) >= 100000:
            session.bulk_insert_mappings(Interaction, interaction_mappings)
            session.commit()
            interaction_mappings = []
            
    if interaction_mappings:
        session.bulk_insert_mappings(Interaction, interaction_mappings)
        session.commit()

    logger.info("Retailrocket ingestion complete.")

if __name__ == "__main__":
    main()
