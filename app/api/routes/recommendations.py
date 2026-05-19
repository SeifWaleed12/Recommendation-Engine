from fastapi import APIRouter, Depends
from typing import List
from app.services.recommendation_service import RecommendationService
from app.api.dependencies import get_recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

@router.get("/{user_id}", response_model=List[dict])
async def get_user_recommendations(
    user_id: str,
    n: int = 10,
    exclude_interacted: bool = True,
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    return await rec_service.get_recommendations(user_id, n=n, exclude_interacted=exclude_interacted)
