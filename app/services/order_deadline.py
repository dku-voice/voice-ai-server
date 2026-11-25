import asyncio
import logging

from app.config import VOICE_RECOMMENDATION_MIN_SECONDS, VOICE_RESPONSE_DEADLINE_SECONDS
from app.services import llm_service

logger = logging.getLogger(__name__)


def order_budget(elapsed_seconds: float) -> float:
    remaining = VOICE_RESPONSE_DEADLINE_SECONDS - max(0.0, elapsed_seconds)
    return max(VOICE_RECOMMENDATION_MIN_SECONDS, remaining)


async def extract_order_within_deadline(extract, text: str, language: str, elapsed_seconds: float) -> dict:
    budget = order_budget(elapsed_seconds)
    try:
        return await asyncio.wait_for(extract(text, language), timeout=budget)
    except asyncio.TimeoutError:
        logger.warning("[WS] 주문 파싱 %.1f초 초과, 키워드 매칭 사용 (앞 단계 %.1f초)", budget, elapsed_seconds)
        items = llm_service._fallback_keyword_parse(text, language=language)
        return {
            "status": "fallback",
            "items": items,
            "error_msg": "LLM 응답 지연, 키워드 매칭으로 대체",
        }
