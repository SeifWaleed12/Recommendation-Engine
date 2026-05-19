import time
import json
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.api.dependencies import get_db, get_recommendation_service, get_cache_service
from app.schemas.storefront import StorefrontHomeResponse, ProductDetailResponse, ProductCard
from app.services.recommendation_service import RecommendationService
from app.services.cache_service import CacheService
from app.models.schema import Item, Interaction, User, FeatureStore
from app.ml.pipeline.recommend import _cold_start_recommend

router = APIRouter(prefix="/storefront", tags=["storefront"])

async def _fetch_items_by_ids(db: AsyncSession, item_ids: List[str]) -> dict:
    if not item_ids:
        return {}
    stmt = select(Item).where(Item.external_id.in_(item_ids))
    res = await db.execute(stmt)
    return {i.external_id: i for i in res.scalars().all()}

def _to_product_card(item_id: str, item_map: dict, rec_info: dict = None, badge: str = None) -> ProductCard:
    item = item_map.get(item_id)
    if not item:
        # Fallback for missing item in DB
        return ProductCard(
            item_id=item_id,
            title=f"Item #{item_id}",
            category="Unknown",
            subcategory="Unknown",
            price=0.0,
            brand="Unknown",
            description="No description available",
            badge=badge
        )
    return ProductCard(
        item_id=item_id,
        title=item.title or f"Item #{item_id}",
        category=item.category or "General",
        subcategory=item.subcategory or "General",
        price=item.price or 0.0,
        brand=item.brand or "Generic",
        description=item.description or "No description available",
        recommendation_score=rec_info.get("score") if rec_info else None,
        retrieval_source=rec_info.get("retrieval_source") if rec_info else None,
        explanation=rec_info.get("explanation") if rec_info else None,
        badge=badge
    )

@router.get("/home/{user_id}", response_model=StorefrontHomeResponse)
async def get_storefront_home(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    cache_service: CacheService = Depends(get_cache_service)
):
    cache_key = f"storefront_home_{user_id}"
    cached = await cache_service.get(cache_key)
    if cached:
        return json.loads(cached)

    start_time = time.time()
    
    # 2. Personalized recs
    personalized_raw = await rec_service.get_recommendations(user_id, n=20)
    
    # 3. Trending
    trending_raw = _cold_start_recommend(n=10)
    
    # 4. Last viewed
    stmt_last = (
        select(Item.external_id, Item.title)
        .join(Interaction, Item.id == Interaction.item_id)
        .join(User, User.id == Interaction.user_id)
        .where(User.external_id == user_id, Interaction.event_type == 'view')
        .order_by(Interaction.timestamp.desc())
        .limit(1)
    )
    res_last = await db.execute(stmt_last)
    last_viewed_row = res_last.first()
    because_you_viewed_raw = []
    last_viewed_title = ""
    if last_viewed_row:
        last_viewed_title = last_viewed_row[1]
        # Get similars using ALS model (simplification: get top from same category)
        stmt_sim = select(Item.external_id).where(Item.category == (select(Item.category).where(Item.external_id == last_viewed_row[0]).scalar_subquery())).limit(10)
        res_sim = await db.execute(stmt_sim)
        because_you_viewed_raw = [{"item_id": r[0], "retrieval_source": "content"} for r in res_sim.all()]

    # 5. Top 3 categories
    # simplified: top categories from items
    stmt_cat = (
        select(Item.category, func.count(Item.id))
        .group_by(Item.category)
        .order_by(func.count(Item.id).desc())
        .limit(3)
    )
    res_cat = await db.execute(stmt_cat)
    top_categories = [r[0] for r in res_cat.all() if r[0]]

    category_rows = []
    for cat in top_categories:
        stmt_items = select(Item.external_id).where(Item.category == cat).limit(6)
        res_items = await db.execute(stmt_items)
        category_rows.append({"category": cat, "items": [{"item_id": r[0]} for r in res_items.all()]})

    # Gather all item IDs
    all_item_ids = set()
    for r in personalized_raw: all_item_ids.add(r['item_id'])
    for r in trending_raw: all_item_ids.add(r['item_id'])
    for r in because_you_viewed_raw: all_item_ids.add(r['item_id'])
    for row in category_rows:
        for r in row['items']: all_item_ids.add(r['item_id'])

    item_map = await _fetch_items_by_ids(db, list(all_item_ids))

    hero_raw = personalized_raw[:3]
    for_you_raw = personalized_raw[3:13]

    hero_items = [_to_product_card(r['item_id'], item_map, r, badge=None) for r in hero_raw]
    
    for_you = []
    for r in for_you_raw:
        badge = "Recommended for You" if r.get('retrieval_source') == 'collaborative' else None
        for_you.append(_to_product_card(r['item_id'], item_map, r, badge=badge))

    trending_now = [_to_product_card(r['item_id'], item_map, r, badge="Trending") for r in trending_raw]
    
    because_you_viewed = None
    if because_you_viewed_raw:
        because_you_viewed = [_to_product_card(r['item_id'], item_map, r, badge=f"Because you viewed {last_viewed_title}") for r in because_you_viewed_raw]

    final_cat_rows = []
    for row in category_rows:
        cat_items = [_to_product_card(r['item_id'], item_map) for r in row['items']]
        final_cat_rows.append({"category": row['category'], "items": cat_items})

    latency = (time.time() - start_time) * 1000

    resp = StorefrontHomeResponse(
        user_id=user_id,
        is_personalized=len(personalized_raw) > 0 and personalized_raw[0].get('retrieval_source') != 'cold_start',
        hero_items=hero_items,
        for_you=for_you,
        trending_now=trending_now,
        because_you_viewed=because_you_viewed,
        category_rows=final_cat_rows,
        generated_at=datetime.utcnow(),
        latency_ms=latency
    )

    await cache_service.set(cache_key, resp.model_dump_json(), ttl=180)
    return resp

@router.get("/product/{item_id}", response_model=ProductDetailResponse)
async def get_storefront_product(
    item_id: str,
    user_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    item_map = await _fetch_items_by_ids(db, [item_id])
    if item_id not in item_map:
        raise HTTPException(status_code=404, detail="Item not found")
    
    product = _to_product_card(item_id, item_map)

    # Similars
    stmt_sim = select(Item.external_id).where(Item.category == item_map[item_id].category).limit(8)
    res_sim = await db.execute(stmt_sim)
    sim_ids = [r[0] for r in res_sim.all()]
    
    # Frequently bought together
    stmt_fbt = select(Item.external_id).limit(6)
    res_fbt = await db.execute(stmt_fbt)
    fbt_ids = [r[0] for r in res_fbt.all()]
    
    # Category picks
    stmt_cat = select(Item.external_id).where(Item.category == item_map[item_id].category).limit(6)
    res_cat = await db.execute(stmt_cat)
    cat_ids = [r[0] for r in res_cat.all()]

    all_extra_ids = list(set(sim_ids + fbt_ids + cat_ids))
    extra_map = await _fetch_items_by_ids(db, all_extra_ids)

    return ProductDetailResponse(
        product=product,
        you_may_also_like=[_to_product_card(i, extra_map) for i in sim_ids],
        frequently_bought_together=[_to_product_card(i, extra_map) for i in fbt_ids],
        category_picks=[_to_product_card(i, extra_map) for i in cat_ids]
    )

@router.get("/category/{category_name}")
async def get_storefront_category(
    category_name: str,
    user_id: Optional[str] = None,
    sort_by: str = "recommended",
    limit: int = 24,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    if sort_by == "recommended" and user_id:
        recs = await rec_service.get_recommendations(user_id, n=100)
        cat_recs = [r['item_id'] for r in recs]
        # Filter DB by cat_recs
        stmt = select(Item).where(Item.external_id.in_(cat_recs), Item.category == category_name).limit(limit).offset(offset)
        res = await db.execute(stmt)
        items = res.scalars().all()
    else:
        stmt = select(Item).where(Item.category == category_name)
        if sort_by == "price_asc":
            stmt = stmt.order_by(Item.price.asc())
        elif sort_by == "price_desc":
            stmt = stmt.order_by(Item.price.desc())
        stmt = stmt.limit(limit).offset(offset)
        res = await db.execute(stmt)
        items = res.scalars().all()
    
    item_map = {i.external_id: i for i in items}
    cards = [_to_product_card(i.external_id, item_map) for i in items]
    return cards
