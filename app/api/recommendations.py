"""
app/api/recommendations.py - 메뉴 추천 RAG 라우터
음성 주문 파싱과 분리해서, 메뉴 추천만 따로 확인할 수 있게 둔다.
"""
import logging

from fastapi import APIRouter

from app.schemas import MenuRecommendationRequest, MenuRecommendationResponse
from app.services.recommendation_service import recommend_menus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.post("/menu", response_model=MenuRecommendationResponse)
async def recommend_menu(request: MenuRecommendationRequest):
    """고객 요청 문장에 맞는 메뉴 후보를 추천한다."""
    try:
        result = await recommend_menus(request.query, top_k=request.top_k)
        return MenuRecommendationResponse(**result)
    except Exception as e:
        logger.error("[RAG] 메뉴 추천 처리 중 예상치 못한 오류: %s", e, exc_info=True)
        return MenuRecommendationResponse(
            status="error",
            query=request.query,
            recommendations=[],
            retrieved_menu_ids=[],
            error_msg="메뉴 추천 중 오류가 발생했습니다.",
        )
