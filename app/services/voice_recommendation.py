import asyncio
import logging
import re

from app.config import VOICE_RECOMMENDATION_BUDGET_SECONDS
from app.services import recommendation_service

logger = logging.getLogger(__name__)

RECOMMENDATION_REQUEST_PATTERN = re.compile(
    r"추천|뭐가\s*맛있|뭐가\s*좋|뭐\s*먹|뭘\s*먹|뭐\s*마실|뭘\s*마실|어떤\s*게\s*좋|골라\s*줘|고를까"
    r"|おすすめ|オススメ|お勧め|お薦め|何がいい|何がおいしい|何が美味しい"
    r"|recomi|recomend|sugier|qu[eé]\s+est[aá]\s+bueno"
    r"|แนะนำ|อะไรอร่อย",
    re.IGNORECASE,
)


def is_recommendation_request(text: str) -> bool:
    return bool(text and RECOMMENDATION_REQUEST_PATTERN.search(text))


def has_preference_cue(text: str) -> bool:
    if not text:
        return False
    if any(pattern.search(text) for _, pattern in recommendation_service.FALLBACK_FEATURE_TERMS):
        return True
    if any(pattern.search(text) for pattern, _, _ in recommendation_service.FALLBACK_NUTRITION_RULES):
        return True
    return bool(recommendation_service.FALLBACK_PRICE_PATTERN.search(text))


def _rule_based(query: str, top_k: int) -> list:
    documents = recommendation_service.retrieve_menu_documents(query, top_k=top_k)
    return recommendation_service._fallback_recommendations(query, documents, top_k)


async def recommend_for_voice(query: str, top_k: int = 3) -> dict:
    try:
        result = await asyncio.wait_for(
            recommendation_service.recommend_menus(query, top_k=top_k),
            timeout=VOICE_RECOMMENDATION_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("[WS] 음성 추천 %.1f초 초과, 규칙 기반 추천 사용", VOICE_RECOMMENDATION_BUDGET_SECONDS)
        return {"source": "fallback", "items": _rule_based(query, top_k)}
    except Exception as e:
        logger.error("[WS] 음성 추천 실패, 규칙 기반 추천 사용: %s", e)
        return {"source": "fallback", "items": _rule_based(query, top_k)}

    items = list(result.get("recommendations") or [])
    if not items:
        items = _rule_based(query, top_k)
        return {"source": "fallback", "items": items}
    return {"source": "llm" if result.get("status") == "success" else "fallback", "items": items}


def recommendation_message(recognized_text: str, language: str, result: dict) -> dict:
    return {
        "type": "recommendation",
        "recognized_text": recognized_text,
        "language": language,
        "source": result["source"],
        "items": [
            {"menu_id": item.menu_id, "name": item.name, "reason": item.reason}
            for item in result["items"]
        ],
    }
