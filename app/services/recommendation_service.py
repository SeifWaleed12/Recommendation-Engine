import asyncio
from typing import List, Dict, Any
from app.ml.pipeline.recommend import recommend, get_similar_items

class RecommendationService:
    async def get_recommendations(
        self, user_id: str, n: int = 10, exclude_interacted: bool = True, context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a user.
        Runs the synchronous ML pipeline recommend function in a thread to avoid blocking.
        """
        return await asyncio.to_thread(
            recommend, user_id, n, exclude_interacted, context
        )

    async def get_similar_items(self, item_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get similar items based on vector distance.
        Runs the synchronous ML function in a thread to avoid blocking.
        """
        return await asyncio.to_thread(
            get_similar_items, item_id, n
        )
