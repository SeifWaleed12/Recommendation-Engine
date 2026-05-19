from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api.dependencies import get_db
from app.models.schema import Item
from app.schemas.item import ItemResponse

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Item).where(Item.external_id == item_id)
    res = await db.execute(stmt)
    item = res.scalars().first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemResponse(
        id=str(item.id), external_id=item.external_id, title=item.title,
        description=item.description, category=item.category,
        subcategory=item.subcategory, price=item.price, brand=item.brand,
        created_at=item.created_at, is_active=item.is_active, metadata_json=item.metadata_json
    )

from typing import List
from app.services.recommendation_service import RecommendationService
from app.api.dependencies import get_recommendation_service

@router.get("/{item_id}/similar", response_model=List[ItemResponse])
async def get_similar_items(
    item_id: str, 
    n: int = 10, 
    db: AsyncSession = Depends(get_db),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    similar_ids_dicts = await rec_service.get_similar_items(item_id, n=n)
    similar_ids = [d["item_id"] for d in similar_ids_dicts]
    
    if not similar_ids:
        return []
        
    stmt = select(Item).where(Item.external_id.in_(similar_ids))
    res = await db.execute(stmt)
    items = res.scalars().all()
    
    # Sort them back into the order provided by the model
    item_map = {i.external_id: i for i in items}
    
    response_items = []
    for sid in similar_ids:
        item = item_map.get(sid)
        if item:
            response_items.append(
                ItemResponse(
                    id=str(item.id), external_id=item.external_id, title=item.title,
                    description=item.description, category=item.category,
                    subcategory=item.subcategory, price=item.price, brand=item.brand,
                    created_at=item.created_at, is_active=item.is_active, metadata_json=item.metadata_json
                )
            )
            
    return response_items
