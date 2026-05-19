from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class EventType(str, Enum):
    VIEW = "view"
    ADD_TO_CART = "add_to_cart"
    PURCHASE = "purchase"

class EventRequest(BaseModel):
    user_id: str
    item_id: str
    event_type: EventType
    session_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = {}

class EventResponse(BaseModel):
    accepted: bool
    event_id: str
