import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.schema import Interaction, EventType, User, Item
from app.ml.pipeline.recommend import notify_interaction

logger = logging.getLogger(__name__)

class EventService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_event(
        self, user_id: str, item_id: str, event_type: EventType, session_id: str = "", metadata: dict = None, background_tasks = None
    ) -> Interaction:
        # We look up the internal UUIDs from external string IDs
        user_stmt = select(User).where(User.external_id == user_id)
        user_res = await self.session.execute(user_stmt)
        user = user_res.scalars().first()

        if not user:
            user = User(external_id=user_id)
            self.session.add(user)
            await self.session.flush()

        item_stmt = select(Item).where(Item.external_id == item_id)
        item_res = await self.session.execute(item_stmt)
        item = item_res.scalars().first()

        if not item:
            item = Item(external_id=item_id, category="unknown", is_active=True)
            self.session.add(item)
            await self.session.flush()

        # Basic weighting logic
        weight = 1.0
        if event_type == EventType.add_to_cart:
            weight = 2.0
        elif event_type == EventType.purchase:
            weight = 5.0

        # Create the interaction record
        interaction = Interaction(
            user_id=user.id,
            item_id=item.id,
            event_type=event_type,
            session_id=session_id,
            weight=weight
        )
        
        self.session.add(interaction)
        await self.session.commit()
        await self.session.refresh(interaction)

        # Notify ML pipeline for real-time updates without blocking
        def _safe_notify():
            try:
                notify_interaction(user_id, item_id)
            except Exception as e:
                logger.warning(f"Could not notify ML pipeline of interaction: {e}")
        
        if background_tasks:
            background_tasks.add_task(_safe_notify)
        else:
            _safe_notify()

        return interaction
