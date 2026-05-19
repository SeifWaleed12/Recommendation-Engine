import os
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.services.cache_service import CacheService
from app.services.recommendation_service import RecommendationService
from app.services.event_service import EventService
from app.services.demo_service import DemoService

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/recsys")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

# Singletons for stateless services
_cache_service = CacheService()
_rec_service = RecommendationService()

async def get_cache_service() -> CacheService:
    return _cache_service

async def get_recommendation_service() -> RecommendationService:
    return _rec_service

async def get_event_service(session: AsyncSession = Depends(get_db)) -> EventService:
    return EventService(session)

async def get_demo_service(session: AsyncSession = Depends(get_db)) -> DemoService:
    return DemoService(session)
