import sys
import json
import random
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.pipeline.recommend import recommend

def sanity_check():
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    # 1. Load interactions to find interesting users
    print("Loading interaction history...")
    df = pd.read_parquet(data_dir / "interactions_clean.parquet")
    
    # 2. Get some users with enough history to be "interesting"
    user_counts = df["user_ext_id"].value_counts()
    frequent_users = user_counts[user_counts >= 10].index.tolist()
    
    if not frequent_users:
        print("No frequent users found, picking random users.")
        frequent_users = df["user_ext_id"].unique().tolist()

    selected_users = random.sample(frequent_users, 3)
    
    print("\n" + "="*80)
    print("RECOMMENDATION ENGINE SANITY CHECK")
    print("="*80)

    for user_id in selected_users:
        print(f"\n[USER ID]: {user_id}")
        
        # Show history
        history = df[df["user_ext_id"] == user_id].sort_values("timestamp", ascending=False).head(5)
        print("\n--- RECENT HISTORY ---")
        for _, row in history.iterrows():
            print(f"  - {row['event_type'].upper()}: {row['item_ext_id']} (at {row['timestamp']})")
        
        # Get Recommendations
        print("\n--- GENERATING TOP 10 RECOMMENDATIONS ---")
        try:
            recs = recommend(user_id, n=10)
            for i, r in enumerate(recs):
                score = r.get('score', 0.0)
                source = r.get('retrieval_source', 'unknown')
                print(f"  {i+1}. Item: {r['item_id']:<10} | Score: {score:.4f} | Source: {source:<8}")
                if r.get('explanation'):
                    print(f"     Why: {r['explanation']}")
        except Exception as e:
            print(f"  Error generating recommendations: {e}")
        
        print("\n" + "-"*40)

    # 3. Test a Brand New User (Cold Start)
    print("\n[USER ID]: NEW_USER_999 (Zero History)")
    print("\n--- RECENT HISTORY ---")
    print("  (No history - Pure Cold Start)")
    print("\n--- GENERATING TOP 10 RECOMMENDATIONS ---")
    try:
        recs = recommend("NEW_USER_999", n=10)
        for i, r in enumerate(recs):
            print(f"  {i+1}. Item: {r['item_id']:<10} | Source: {r.get('retrieval_source', 'unknown')}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*80)
    print("CHECK COMPLETE")

if __name__ == "__main__":
    sanity_check()
