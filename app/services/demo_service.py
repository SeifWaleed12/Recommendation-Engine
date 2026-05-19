import uuid
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.models.schema import User, Item, Interaction, EventType
from app.schemas.demo import DemoStep, DemoSessionResponse
from app.schemas.storefront import ProductCard
from app.services.event_service import EventService
from app.services.recommendation_service import RecommendationService
from app.ml.pipeline.recommend import _cold_start_recommend

logger = logging.getLogger(__name__)

COLD_USER_SCRIPT = [
    {"action": "get_recs",   "item_id": None},           
    {"action": "view",       "item_id": "PICK_POPULAR_1"}, 
    {"action": "get_recs",   "item_id": None},           
    {"action": "view",       "item_id": "PICK_POPULAR_2"}, 
    {"action": "view",       "item_id": "PICK_POPULAR_3"}, 
    {"action": "get_recs",   "item_id": None},           
    {"action": "add_to_cart","item_id": "PICK_POPULAR_1"}, 
    {"action": "get_recs",   "item_id": None},           
    {"action": "purchase",   "item_id": "PICK_POPULAR_1"}, 
    {"action": "get_recs",   "item_id": None},           
]

WARM_USER_SCRIPT = [
    {"action": "get_recs",    "item_id": None},          
    {"action": "view",        "item_id": "PICK_FROM_NEW_CATEGORY"},
    {"action": "get_recs",    "item_id": None},          
    {"action": "add_to_cart", "item_id": "PICK_FROM_NEW_CATEGORY"},
    {"action": "get_recs",    "item_id": None},          
    {"action": "purchase",    "item_id": "PICK_FROM_NEW_CATEGORY"},
    {"action": "get_recs",    "item_id": None},          
]

class DemoService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.event_service = EventService(session)
        self.rec_service = RecommendationService()

    async def _get_popular_items(self, limit: int = 3) -> List[Item]:
        # Simple proxy for popular items: top interacted
        stmt = (
            select(Item, func.count(Interaction.id).label('cnt'))
            .join(Interaction, Item.id == Interaction.item_id)
            .group_by(Item.id)
            .order_by(func.count(Interaction.id).desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        rows = res.all()
        if not rows:
            # Fallback to random items if no interactions
            stmt2 = select(Item).limit(limit)
            res2 = await self.session.execute(stmt2)
            return res2.scalars().all()
        return [row[0] for row in rows]

    async def _get_warm_user(self) -> Optional[User]:
        # Select a "normal" warm user with moderate interaction history (5 to 15 interactions)
        stmt = (
            select(User, func.count(Interaction.id).label('cnt'))
            .join(Interaction, User.id == Interaction.user_id)
            .group_by(User.id)
            .having(func.count(Interaction.id).between(5, 15))
            .order_by(func.random())
            .limit(1)
        )
        res = await self.session.execute(stmt)
        row = res.first()
        return row[0] if row else None

    async def _get_new_category_item(self, user_id: str) -> Optional[Item]:
        # Get user's interacted categories
        interacted_cats_stmt = (
            select(Item.category).distinct()
            .join(Interaction, Item.id == Interaction.item_id)
            .join(User, User.id == Interaction.user_id)
            .where(User.external_id == user_id)
        )
        res = await self.session.execute(interacted_cats_stmt)
        interacted_cats = [r[0] for r in res.all() if r[0]]

        # Find item not in these categories
        stmt = select(Item)
        if interacted_cats:
            stmt = stmt.where(Item.category.not_in(interacted_cats))
        stmt = stmt.limit(1)
        res2 = await self.session.execute(stmt)
        return res2.scalars().first()

    def _map_to_product_card(self, rec_dict: dict, item_details: dict) -> ProductCard:
        item = item_details.get(rec_dict['item_id'])
        if not item:
            # Fallback for missing item in DB
            return ProductCard(
                item_id=rec_dict['item_id'],
                title=f"Item #{rec_dict['item_id']}",
                category="Unknown",
                subcategory="Unknown",
                price=0.0,
                brand="Unknown",
                description="No description available",
                recommendation_score=rec_dict.get('score', 0.0),
                retrieval_source=rec_dict.get('retrieval_source', ''),
                explanation=rec_dict.get('explanation', '')
            )
        return ProductCard(
            item_id=rec_dict['item_id'],
            title=item.title or f"Item #{rec_dict['item_id']}",
            category=item.category or "General",
            subcategory=item.subcategory or "General",
            price=item.price or 0.0,
            brand=item.brand or "Generic",
            description=item.description or "No description available",
            recommendation_score=rec_dict.get('score', 0.0),
            retrieval_source=rec_dict.get('retrieval_source', ''),
            explanation=rec_dict.get('explanation', '')
        )

    async def _fetch_items(self, item_ids: List[str]) -> Dict[str, Item]:
        if not item_ids:
            return {}
        stmt = select(Item).where(Item.external_id.in_(item_ids))
        res = await self.session.execute(stmt)
        return {item.external_id: item for item in res.scalars().all()}

    async def run_demo(self, user_type: str) -> DemoSessionResponse:
        is_cold = user_type == "cold_user"
        
        # 1. Setup User
        if is_cold:
            external_id = f"demo_cold_{uuid.uuid4().hex[:8]}"
            user = User(external_id=external_id, metadata_json={"demo": True})
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            script = COLD_USER_SCRIPT.copy()
        else:
            user = await self._get_warm_user()
            if not user:
                raise ValueError("No warm user found to run demo.")
            external_id = user.external_id
            script = WARM_USER_SCRIPT.copy()

        # Resolve dynamic item IDs
        if is_cold:
            pop_items = await self._get_popular_items(3)
            pop_map = {}
            for i, it in enumerate(pop_items):
                pop_map[f"PICK_POPULAR_{i+1}"] = it.external_id
            for s in script:
                if s["item_id"] in pop_map:
                    s["item_id"] = pop_map[s["item_id"]]
        else:
            new_item = await self._get_new_category_item(external_id)
            for s in script:
                if s["item_id"] == "PICK_FROM_NEW_CATEGORY":
                    s["item_id"] = new_item.external_id if new_item else "dummy"

        # Trending items for personalization score computation
        # Use a true dummy request to capture what the hybrid ranker outputs for cold starts
        baseline_dicts = await self.rec_service.get_recommendations("dummy_cold_user_for_baseline", n=20)
        trending_ids = {d['item_id'] for d in baseline_dicts}

        steps_out = []
        last_recs = []
        personalization_progression = []
        total_interactions = 0
        
        # We need a session_id for this demo session
        demo_session_id = uuid.uuid4().hex
        
        current_score = 0.0

        for i, step in enumerate(script):
            action = step["action"]
            target_item_id = step["item_id"]
            
            recs_before = list(last_recs)
            recs_after = None
            changed_items = None
            explanation = ""
            item_title = None
            
            if action in ["view", "add_to_cart", "purchase"]:
                # Record event
                ev_type = EventType(action)
                await self.event_service.record_event(
                    external_id, target_item_id, ev_type, session_id=demo_session_id, metadata={"demo": True}
                )
                total_interactions += 1
                
                # Fetch title for explanation
                it_map = await self._fetch_items([target_item_id])
                if target_item_id in it_map:
                    item_title = it_map[target_item_id].title or f"Item #{target_item_id}"
                
                explanation = f"User {action}ed item: {item_title or target_item_id}."
                personalization_progression.append(current_score)
                steps_out.append(DemoStep(
                    step_number=i+1,
                    action=action,
                    item_id=target_item_id,
                    item_title=item_title,
                    recommendations_before=None,
                    recommendations_after=None,
                    changed_items=None,
                    explanation=explanation
                ))
            elif action == "get_recs":
                raw_recs = await self.rec_service.get_recommendations(external_id, n=10)
                item_ids = [r["item_id"] for r in raw_recs]
                details_map = await self._fetch_items(item_ids)
                recs_after = [self._map_to_product_card(r, details_map) for r in raw_recs]
                
                before_ids = set([r.item_id for r in recs_before])
                after_ids = set([r.item_id for r in recs_after])
                changed = list(before_ids.symmetric_difference(after_ids))
                
                explanation = f"Fetched recommendations. {len(changed)} items changed."
                
                # compute personalization score
                overlap = len(after_ids.intersection(trending_ids))
                score = 1.0 - (overlap / max(1, len(after_ids)))
                current_score = score
                personalization_progression.append(current_score)
                
                steps_out.append(DemoStep(
                    step_number=i+1,
                    action=action,
                    item_id=None,
                    item_title=None,
                    recommendations_before=recs_before if recs_before else None,
                    recommendations_after=recs_after,
                    changed_items=changed,
                    explanation=explanation
                ))
                last_recs = list(recs_after)

        # Profile summary
        final_profile_summary = (
            f"Demo completed with {total_interactions} interactions. "
            f"Personalization score went from {personalization_progression[0]:.2f} "
            f"to {personalization_progression[-1]:.2f}." if personalization_progression else "Demo completed."
        )

        return DemoSessionResponse(
            user_type=user_type,
            user_id=external_id,
            steps=steps_out,
            total_interactions=total_interactions,
            personalization_progression=personalization_progression,
            final_profile_summary=final_profile_summary
        )
