from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.schemas.event import EventRequest, EventResponse
from app.services.event_service import EventService
from app.api.dependencies import get_event_service

router = APIRouter(prefix="/events", tags=["events"])

@router.post("", response_model=EventResponse)
async def record_event(
    event: EventRequest,
    background_tasks: BackgroundTasks,
    event_service: EventService = Depends(get_event_service)
):
    try:
        interaction = await event_service.record_event(
            user_id=event.user_id,
            item_id=event.item_id,
            event_type=event.event_type,
            session_id=event.session_id,
            metadata=event.metadata,
            background_tasks=background_tasks
        )
        return EventResponse(accepted=True, event_id=str(interaction.id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
