"""
app/services/recommendation_service.py - 메뉴 추천 RAG 기초 파이프라인

지금 단계에서는 메뉴 DB나 벡터 DB까지 붙이지 않는다.
현재 서버가 알고 있는 menu_id를 LangChain Document 형태로 만들고,
간단한 키워드 검색 결과만 LLM에 넘겨 추천 JSON을 받는다.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from app.config import LLM_MODEL
from app.schemas import MenuRecommendationItem
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseValidationError,
    MENU_ALIASES,
    MENU_CATALOG,
    get_llm_client,
)
from app.services.threadpool import run_llm_task

logger = logging.getLogger(__name__)

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None


@dataclass(frozen=True)
class SimpleDocument:
    """langchain-core가 없는 로컬 테스트 환경에서 쓰는 최소 Document."""
    page_content: str
    metadata: dict[str, Any]


MenuDocument = Any


MENU_RECOMMENDATION_PROFILES = [
    {
        "menu_id": "burger_01",
        "category": "main",
        "tags": ("식사", "버거", "든든함", "단품"),
        "description": "기본 햄버거 메뉴. 한 끼 식사로 고르기 좋은 메인 메뉴입니다.",
    },
    {
        "menu_id": "cola_01",
        "category": "drink",
        "tags": ("음료", "탄산", "차가운 음료", "세트 추가"),
        "description": "콜라 음료. 버거, 치킨, 감자튀김과 같이 고르기 좋습니다.",
    },
    {
        "menu_id": "fries_01",
        "category": "side",
        "tags": ("사이드", "감자", "간단한 추가", "세트 추가"),
        "description": "감자튀김 사이드 메뉴. 메인 메뉴와 함께 추가하기 좋습니다.",
    },
    {
        "menu_id": "pizza_01",
        "category": "main",
        "tags": ("식사", "피자", "나눠먹기", "메인"),
        "description": "피자 메뉴. 여러 사람이 나눠 먹기 좋은 메인 메뉴입니다.",
    },
    {
        "menu_id": "chicken_01",
        "category": "main",
        "tags": ("식사", "치킨", "나눠먹기", "메인"),
        "description": "치킨 메뉴. 메인 메뉴나 함께 먹는 메뉴로 고르기 좋습니다.",
    },
]

RECOMMENDATION_INTENT_TERMS = (
    "추천",
    "메뉴",
    "먹",
    "마시",
    "음식",
    "식사",
    "간식",
    "고르",
    "주문",
    "배고",
    "출출",
)

RECOMMENDATION_SYSTEM_PROMPT = """당신은 패스트푸드점 Kiosk의 메뉴 추천 마이크로서비스입니다.
반드시 제공된 메뉴 문서에 있는 menu_id만 추천하세요.
메뉴 문서에 없는 가격, 알레르기, 매운맛, 재고 정보는 지어내지 마세요.
응답은 JSON 객체 하나로만 출력하세요."""

RECOMMENDATION_FORMAT_INSTRUCTION = """
아래 형식의 JSON 객체만 출력하세요.
{
  "recommendations": [
    {"menu_id": "burger_01", "reason": "고객 요청과 맞는 이유"}
  ]
}
"""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _create_document(page_content: str, metadata: dict[str, Any]) -> MenuDocument:
    if LangChainDocument is not None:
        return LangChainDocument(page_content=page_content, metadata=metadata)
    return SimpleDocument(page_content=page_content, metadata=metadata)


def _profile_to_document(profile: dict[str, Any]) -> MenuDocument:
    menu_id = profile["menu_id"]
    name = MENU_CATALOG[menu_id]
    aliases = MENU_ALIASES.get(menu_id, ())
    search_terms = tuple(dict.fromkeys((name, *aliases, profile["category"], *profile["tags"])))

    page_content = "\n".join(
        [
            f"menu_id: {menu_id}",
            f"name: {name}",
            f"category: {profile['category']}",
            f"tags: {', '.join(profile['tags'])}",
            f"description: {profile['description']}",
        ]
    )
    return _create_document(
        page_content=page_content,
        metadata={
            "menu_id": menu_id,
            "name": name,
            "category": profile["category"],
            "tags": list(profile["tags"]),
            "search_terms": list(search_terms),
        },
    )


def build_menu_documents() -> list[MenuDocument]:
    """현재 서버에 있는 메뉴 목록을 RAG 검색용 문서로 만든다."""
    return [_profile_to_document(profile) for profile in MENU_RECOMMENDATION_PROFILES]


def _score_document(query: str, document: MenuDocument) -> int:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0

    score = 0
    for term in document.metadata["search_terms"]:
        normalized_term = _normalize_text(term)
        if normalized_term and normalized_term in normalized_query:
            score += 10 + len(normalized_term)

    content = _normalize_text(document.page_content)
    for token in re.findall(r"[0-9A-Za-z가-힣]+", query.lower()):
        if len(token) >= 2 and token in content:
            score += 1

    return score


def _looks_like_recommendation_query(query: str) -> bool:
    normalized_query = _normalize_text(query)
    return any(term in normalized_query for term in RECOMMENDATION_INTENT_TERMS)


def retrieve_menu_documents(query: str, top_k: int = 3) -> list[MenuDocument]:
    """
    메뉴 문서 검색.
    제5주차의 embedding/vector DB 최적화 전까지는 간단한 키워드 점수로만 고른다.
    """
    query = query.strip()
    if not query:
        return []

    documents = build_menu_documents()
    scored = [
        (_score_document(query, document), index, document)
        for index, document in enumerate(documents)
    ]
    matched = [item for item in scored if item[0] > 0]

    # "추천해줘"처럼 조건이 아직 넓은 문장은 기본 메뉴 문서를 안정적인 순서로 넘긴다.
    if matched:
        candidates = matched
    elif _looks_like_recommendation_query(query):
        candidates = scored
    else:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [document for _, _, document in candidates[:top_k]]


def _extract_json_object(raw_text: str) -> str:
    cleaned = raw_text.strip()

    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        raise LLMResponseValidationError("LLM 응답에서 JSON 객체를 찾지 못했습니다.")

    return cleaned[start_idx:end_idx + 1]


def _document_by_menu_id(documents: Sequence[MenuDocument]) -> dict[str, MenuDocument]:
    return {str(document.metadata["menu_id"]): document for document in documents}


def _coerce_recommendation(
    entry: Any,
    documents_by_menu_id: dict[str, MenuDocument],
    query: str,
) -> MenuRecommendationItem:
    if not isinstance(entry, dict):
        raise LLMResponseValidationError("추천 항목은 JSON 객체여야 합니다.")

    menu_id = str(entry.get("menu_id", "")).strip()
    if menu_id not in documents_by_menu_id:
        raise LLMResponseValidationError(f"검색 결과에 없는 menu_id입니다: {menu_id or '[empty]'}")

    document = documents_by_menu_id[menu_id]

    return MenuRecommendationItem(
        menu_id=menu_id,
        name=MENU_CATALOG[menu_id],
        reason=_fallback_reason(query, document),
    )


def _parse_recommendation_response(
    raw_text: str,
    documents: Sequence[MenuDocument],
    top_k: int,
    query: str = "",
) -> list[MenuRecommendationItem]:
    cleaned = _extract_json_object(raw_text)
    parsed = json.loads(cleaned)

    if not isinstance(parsed, dict):
        raise LLMResponseValidationError("추천 응답이 JSON 객체가 아닙니다.")

    raw_recommendations = parsed.get("recommendations")
    if not isinstance(raw_recommendations, list):
        raise LLMResponseValidationError("recommendations 필드는 리스트여야 합니다.")

    documents_by_menu_id = _document_by_menu_id(documents)
    recommendations: list[MenuRecommendationItem] = []
    seen_menu_ids: set[str] = set()
    for entry in raw_recommendations:
        item = _coerce_recommendation(entry, documents_by_menu_id, query)
        if item.menu_id in seen_menu_ids:
            continue
        recommendations.append(item)
        seen_menu_ids.add(item.menu_id)
        if len(recommendations) >= top_k:
            break

    if not recommendations and documents:
        raise LLMResponseValidationError("추천 결과가 비어 있습니다.")

    return recommendations


def _format_documents_for_prompt(documents: Sequence[MenuDocument]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def _call_recommendation_llm_sync(
    query: str,
    documents: Sequence[MenuDocument],
    top_k: int,
) -> tuple[str, list[MenuRecommendationItem]]:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"고객 요청: \"{query}\"\n\n"
                    f"검색된 메뉴 문서:\n{_format_documents_for_prompt(documents)}\n\n"
                    f"최대 {top_k}개까지만 추천하세요.\n"
                    f"{RECOMMENDATION_FORMAT_INSTRUCTION}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise LLMResponseValidationError("LLM 응답 content가 None입니다.")

    return ("success", _parse_recommendation_response(raw, documents, top_k, query=query))


def _fallback_reason(query: str, document: MenuDocument) -> str:
    normalized_query = _normalize_text(query)
    for term in document.metadata["search_terms"]:
        normalized_term = _normalize_text(term)
        if normalized_term and normalized_term in normalized_query:
            return f"'{term}' 조건과 연결되는 메뉴입니다."
    return "현재 메뉴 문서에서 추천 후보로 선택했습니다."


def _fallback_recommendations(
    query: str,
    documents: Sequence[MenuDocument],
    top_k: int,
) -> list[MenuRecommendationItem]:
    recommendations: list[MenuRecommendationItem] = []
    for document in documents[:top_k]:
        menu_id = document.metadata["menu_id"]
        recommendations.append(
            MenuRecommendationItem(
                menu_id=menu_id,
                name=document.metadata["name"],
                reason=_fallback_reason(query, document),
            )
        )
    return recommendations


async def recommend_menus(query: str, top_k: int = 3) -> dict:
    """
    메뉴 추천 RAG 기초 파이프라인.

    메뉴 문서 검색은 가볍게 동기 실행하고, LLM 호출만 threadpool에서 돌린다.
    """
    query = query.strip()
    if not query:
        return {
            "status": "error",
            "query": "",
            "recommendations": [],
            "retrieved_menu_ids": [],
            "error_msg": "추천 요청 문장이 비어 있습니다.",
        }

    top_k = max(1, min(top_k, 5))
    documents = retrieve_menu_documents(query, top_k=top_k)
    retrieved_menu_ids = [document.metadata["menu_id"] for document in documents]

    if not documents:
        return {
            "status": "error",
            "query": query,
            "recommendations": [],
            "retrieved_menu_ids": [],
            "error_msg": "추천에 사용할 메뉴 문서를 찾지 못했습니다.",
        }

    try:
        phase, recommendations = await run_llm_task(
            _call_recommendation_llm_sync,
            query,
            documents,
            top_k,
        )
        return {
            "status": phase,
            "query": query,
            "recommendations": recommendations,
            "retrieved_menu_ids": retrieved_menu_ids,
            "error_msg": None,
        }
    except LLMConfigurationError as e:
        logger.warning("[RAG] LLM 설정 오류, 메뉴 문서 fallback 사용: %s", e)
    except (json.JSONDecodeError, LLMResponseValidationError) as e:
        logger.warning("[RAG] LLM 추천 응답 검증 실패, 메뉴 문서 fallback 사용: %s", e)
    except Exception as e:
        logger.error("[RAG] 메뉴 추천 LLM 호출 실패, 메뉴 문서 fallback 사용: %s", e)

    return {
        "status": "fallback",
        "query": query,
        "recommendations": _fallback_recommendations(query, documents, top_k),
        "retrieved_menu_ids": retrieved_menu_ids,
        "error_msg": "LLM 추천 실패, 메뉴 문서 검색 결과로 대체",
    }
