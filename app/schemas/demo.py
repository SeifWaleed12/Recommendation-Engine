from pydantic import BaseModel
from typing import Optional
from app.schemas.storefront import ProductCard

class DemoStep(BaseModel):
    step_number: int
    action: str           # "view", "add_to_cart", "purchase", "get_recs"
    item_id: Optional[str] = None
    item_title: Optional[str] = None
    recommendations_before: Optional[list[ProductCard]] = None
    recommendations_after: Optional[list[ProductCard]] = None
    changed_items: Optional[list[str]] = None
    explanation: str

class DemoSessionResponse(BaseModel):
    user_type: str        # "cold_user" | "warm_user"
    user_id: str
    steps: list[DemoStep]
    total_interactions: int
    personalization_progression: list[float]  
    final_profile_summary: str
