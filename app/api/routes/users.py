from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api.dependencies import get_db
from app.models.schema import User, Interaction, Item
from app.schemas.user import UserResponse

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(User).where(User.external_id == user_id)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=str(user.id), external_id=user.external_id,
        created_at=user.created_at, last_active_at=user.last_active_at,
        metadata_json=user.metadata_json
    )

@router.get("/{user_id}/history")
async def get_user_history(user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Item.external_id)
        .join(Interaction, Item.id == Interaction.item_id)
        .join(User, User.id == Interaction.user_id)
        .where(User.external_id == user_id)
        .order_by(Interaction.timestamp.desc())
        .limit(10)
    )
    res = await db.execute(stmt)
    return {"history": [row[0] for row in res.all()]}
