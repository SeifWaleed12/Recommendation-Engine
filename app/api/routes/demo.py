from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.schemas.demo import DemoSessionResponse
from app.services.demo_service import DemoService
from app.api.dependencies import get_demo_service
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.schema import User, Interaction
from sqlalchemy import func
from app.api.dependencies import get_db

router = APIRouter(prefix="/demo", tags=["demo"])

class DemoRunRequest(BaseModel):
    user_type: str # "cold_user" | "warm_user"

@router.post("/run", response_model=DemoSessionResponse)
async def run_demo(
    request: DemoRunRequest,
    demo_service: DemoService = Depends(get_demo_service)
):
    return await demo_service.run_demo(request.user_type)

@router.get("/users")
async def get_demo_users(db: AsyncSession = Depends(get_db)):
    # warm user = normal user with 5-15 interactions
    stmt_warm = (
        select(User, func.count(Interaction.id).label('cnt'))
        .join(Interaction, User.id == Interaction.user_id)
        .group_by(User.id)
        .having(func.count(Interaction.id).between(5, 15))
        .order_by(func.random())
        .limit(1)
    )
    res = await db.execute(stmt_warm)
    warm = res.first()

    stmt_cold = select(User).where(User.external_id.like('demo_cold_%')).order_by(User.created_at.desc()).limit(1)
    res_c = await db.execute(stmt_cold)
    cold = res_c.scalars().first()

    return {
        "cold_user_id": cold.external_id if cold else None,
        "warm_user_id": warm[0].external_id if warm else None,
        "cold_interaction_count": 0,
        "warm_interaction_count": warm[1] if warm else 0
    }
