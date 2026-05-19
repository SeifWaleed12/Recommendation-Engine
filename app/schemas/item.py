from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ItemResponse(BaseModel):
    id: str
    external_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None
    created_at: datetime
    is_active: bool
    metadata_json: Optional[dict] = None
