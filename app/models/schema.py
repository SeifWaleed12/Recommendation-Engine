import enum
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base

class EventType(enum.Enum):
    view = "view"
    add_to_cart = "add_to_cart"
    purchase = "purchase"

class EntityType(enum.Enum):
    user = "user"
    item = "item"

class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime)
    metadata_json = Column(JSONB)

class Item(Base):
    __tablename__ = 'items'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String)
    description = Column(Text)
    category = Column(String)
    subcategory = Column(String)
    price = Column(Float)
    brand = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata_json = Column(JSONB)

class Interaction(Base):
    __tablename__ = 'interactions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    item_id = Column(UUID(as_uuid=True), ForeignKey('items.id'), nullable=False)
    event_type = Column(Enum(EventType), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String)
    weight = Column(Float, nullable=False)

    __table_args__ = (
        Index('idx_interaction_user_time', 'user_id', 'timestamp'),
        Index('idx_interaction_item_time', 'item_id', 'timestamp'),
    )

class UserItemMatrix(Base):
    __tablename__ = 'user_item_matrix'

    # Using integer IDs for the matrix to match ALS integer mappings easily
    user_idx = Column(Integer, primary_key=True)
    item_idx = Column(Integer, primary_key=True)
    confidence = Column(Float, nullable=False)
    interaction_count = Column(Integer, nullable=False, default=0)

class FeatureStore(Base):
    __tablename__ = 'feature_store'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(Enum(EntityType), nullable=False)
    entity_id = Column(String, nullable=False) # Maps to external_id
    feature_name = Column(String, nullable=False)
    feature_value_json = Column(JSONB, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)
    version = Column(Integer, default=1)

    __table_args__ = (
        Index('idx_feature_store_lookup', 'entity_type', 'entity_id', 'feature_name'),
    )
