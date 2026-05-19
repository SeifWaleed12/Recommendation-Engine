from fastapi import APIRouter
import redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import os

router = APIRouter()

DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/recsys"
REDIS_URL = "redis://localhost:6379"

engine = create_async_engine(DB_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

@router.get("/health")
async def health_check():
    status = {
        "db_connection": False,
        "redis_connection": False,
        "matrix_file_exists": False,
        "counts": {
            "users": 0,
            "items": 0,
            "interactions": 0
        }
    }
    
    # Check DB
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            if result.scalar() == 1:
                status["db_connection"] = True
                
                # Get counts
                users_cnt = await session.execute(text("SELECT COUNT(*) FROM users"))
                items_cnt = await session.execute(text("SELECT COUNT(*) FROM items"))
                inter_cnt = await session.execute(text("SELECT COUNT(*) FROM interactions"))
                
                status["counts"]["users"] = users_cnt.scalar()
                status["counts"]["items"] = items_cnt.scalar()
                status["counts"]["interactions"] = inter_cnt.scalar()
    except Exception as e:
        status["db_error"] = str(e)
        
    # Check Redis
    try:
        r = redis.from_url(REDIS_URL)
        if r.ping():
            status["redis_connection"] = True
    except Exception as e:
        status["redis_error"] = str(e)
        
    # Check Matrix
    matrix_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "interaction_matrix.npz"
    status["matrix_file_exists"] = matrix_path.exists()
    
    return status

@router.get("/stats")
async def get_stats():
    # Similar to health check but can return more detailed feature stats
    stats = {
        "feature_store_counts": {}
    }
    try:
        async with AsyncSessionLocal() as session:
            user_features = await session.execute(text("SELECT COUNT(*) FROM feature_store WHERE entity_type = 'user'"))
            item_features = await session.execute(text("SELECT COUNT(*) FROM feature_store WHERE entity_type = 'item'"))
            
            stats["feature_store_counts"]["users"] = user_features.scalar()
            stats["feature_store_counts"]["items"] = item_features.scalar()
            
    except Exception as e:
        stats["error"] = str(e)
        
    return stats
