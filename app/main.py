from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.health import router as health_router
from app.api.routes.recommendations import router as recommendations_router
from app.api.routes.users import router as users_router
from app.api.routes.events import router as events_router
from app.api.routes.items import router as items_router
from app.api.routes.storefront import router as storefront_router
from app.api.routes.demo import router as demo_router
from app.api.middleware import add_process_time_header
import asyncio
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load ML pipeline on startup so the first request is instant."""
    logger.info("🚀 Server starting — loading ML pipeline (this takes 1-2 mins on first run)...")
    try:
        from app.ml.pipeline.recommend import _initialize
        # Run in a thread so it doesn't block the event loop
        await asyncio.to_thread(_initialize)
        logger.info("✅ ML pipeline loaded successfully — ready to serve requests")
    except Exception as e:
        logger.error(f"❌ ML pipeline failed to load: {e}")
        logger.warning("Server will start but /recommendations will return cold-start results only")
    yield
    # Cleanup on shutdown (nothing needed here)
    logger.info("Server shutting down")

app = FastAPI(
    title="Recommendation Engine Production API",
    description="Full API serving layer for recommendations and demo functionality.",
    lifespan=lifespan
)

app.add_middleware(BaseHTTPMiddleware, dispatch=add_process_time_header)

app.include_router(health_router, prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(events_router, prefix="/api/v1")
app.include_router(items_router, prefix="/api/v1")
app.include_router(storefront_router, prefix="/api/v1")
app.include_router(demo_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
