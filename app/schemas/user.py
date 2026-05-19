from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserResponse(BaseModel):
    id: str
    external_id: str
    created_at: datetime
    last_active_at: Optional[datetime] = None
    metadata_json: Optional[dict] = None
