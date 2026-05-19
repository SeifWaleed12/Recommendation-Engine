from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ProductCard(BaseModel):
    item_id: str
    title: str
    category: str
    subcategory: str
    price: float
    brand: str
    description: str
    recommendation_score: Optional[float] = None
    retrieval_source: Optional[str] = None
    explanation: Optional[str] = None
    badge: Optional[str] = None

class StorefrontHomeResponse(BaseModel):
    user_id: str
    is_personalized: bool
    hero_items: list[ProductCard]
    for_you: list[ProductCard]
    trending_now: list[ProductCard]
    because_you_viewed: Optional[list[ProductCard]] = None
    category_rows: list[dict]
    generated_at: datetime
    latency_ms: float

class ProductDetailResponse(BaseModel):
    product: ProductCard
    you_may_also_like: list[ProductCard]
    frequently_bought_together: list[ProductCard]
    category_picks: list[ProductCard]
