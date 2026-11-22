import logging

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.services import llm_service
from app.services.menu_sync import sync_menus_once

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/menus", tags=["menus"])


@router.get("")
async def get_menus():
    return llm_service.menu_catalog_summary()


@router.post("/resync")
@router.post("/reload")
async def resync_menus():
    try:
        return await run_in_threadpool(sync_menus_once)
    except Exception as e:
        logger.warning("[MENU] 수동 재동기화 실패: %s", e)
        raise HTTPException(status_code=502, detail=f"백엔드 메뉴 동기화 실패: {e}")
