from datetime import datetime
from pydantic import BaseModel

class RecommendedItem(BaseModel):
    item_id: str
    title: str
    category: str
    price: float
    score: float
    rank: int
    explanation: str
    retrieval_source: str  # "collaborative"|"content"|"neural"|"trending"

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendedItem]
    generated_at: datetime
    model_version: str
    is_personalized: bool
    latency_ms: float

class SimilarItemsResponse(BaseModel):
    item_id: str
    similar_items: list[RecommendedItem]
    generated_at: datetime
