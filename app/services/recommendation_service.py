"""
app/services/recommendation_service.py - 메뉴 추천 RAG 파이프라인

메뉴 DB나 외부 벡터 DB까지는 아직 안 붙인다.
현재 서버가 알고 있는 menu_id를 LangChain Document 형태로 만들고,
키워드 매치 + 본문 토큰 + 메뉴 메타데이터 기반 해시 TF-IDF 임베딩 점수를
합쳐서 후보를 정렬한 뒤 LLM에 넘겨 추천 JSON을 받는다.
"""
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from app.config import LLM_MODEL
from app.schemas import MenuRecommendationItem
# MENU_CATALOG 등은 백엔드 동기화 때 llm_service 쪽에서 통째로 교체된다.
# 이름만 import하면 옛 객체를 계속 보게 되므로, 모듈을 통해 호출 시점에 읽는다.
from app.services import llm_service
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseValidationError,
    completion_options,
    get_llm_client,
)
from app.services.menu_embedding import MenuEmbeddingIndex, embedding_boost
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
고객 요청은 한국어, 일본어, 스페인어, 태국어 중 하나일 수 있습니다.
반드시 제공된 메뉴 문서에 있는 menu_id만 추천하세요.
메뉴 문서에 없는 가격, 알레르기, 매운맛, 재고 정보는 지어내지 마세요.
요청은 아래 기준으로 해석하세요.
- 기름진/느끼한 요청은 문서의 지방 수치가 높은 메뉴를, 담백한/가벼운 요청은 지방과 칼로리 수치가 낮은 메뉴를 우선 추천하세요.
- 짭짤한 요청은 나트륨이 높은 메뉴를, 단백질이 높은 요청은 단백질 수치가 높은 메뉴를 우선 추천하세요.
- 매콤한 요청이나 치즈, 채소 같은 맛/재료 요청은 문서의 설명에 해당 표현이 있는 메뉴를 우선 추천하세요.
- 설명 끝의 "특징:" 항목은 메뉴의 특성입니다. 고객 요청과 뜻이 맞는 특징이 있는 메뉴를 우선 추천하세요.
  (들고 다니며 먹기 좋은 거→간편함, 아이가 먹기 좋은 거→아이 추천, 처음 왔는데 뭐가 맛있어요→대표 메뉴)
- 싼/저렴한/가성비 요청은 문서의 price가 낮은 메뉴를 우선 추천하세요.
- 문서에 근거가 없으면 추천하지 마세요.
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
    name = llm_service.MENU_CATALOG[menu_id]
    aliases = llm_service.MENU_ALIASES.get(menu_id, ())
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


# 영양 정보 표시 단위 (백엔드 마이그레이션 기준: 칼로리 kcal, 탄수화물/지방/단백질 g, 나트륨 mg)
NUTRITION_UNITS = {"칼로리": "kcal", "탄수화물": "g", "지방": "g", "단백질": "g", "나트륨": "mg"}


def _format_number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _format_nutrition(nutrition: dict[str, Any]) -> str:
    return ", ".join(
        f"{label} {_format_number(value)}{NUTRITION_UNITS.get(label, '')}"
        for label, value in nutrition.items()
    )


def _live_document(menu_id: str, name: str) -> MenuDocument:
    """백엔드에서 동기화된 메뉴 하나를 추천 문서로 만든다."""
    meta = llm_service.MENU_META_BY_ID.get(menu_id, {})
    aliases = llm_service.MENU_ALIASES.get(menu_id, ())
    foreign_names = llm_service.MENU_FOREIGN_NAMES.get(menu_id, ())
    category = meta.get("category", "")
    description = meta.get("description", "")
    allergens = tuple(meta.get("allergens", ()))
    nutrition = meta.get("nutrition", {})
    price = meta.get("price")

    # 외국어 이름도 검색어에 넣어서 일본어/스페인어/태국어 질문이 메뉴 이름으로 매칭되게 한다.
    search_terms = tuple(dict.fromkeys(term for term in (name, *aliases, *foreign_names, category) if term))

    lines = [f"menu_id: {menu_id}", f"name: {name}"]
    other_names = [n for n in dict.fromkeys(foreign_names) if n and n != name]
    if other_names:
        lines.append(f"other_names: {' / '.join(other_names)}")
    if category:
        lines.append(f"category: {category}")
    if description:
        lines.append(f"description: {description}")
    if allergens:
        lines.append(f"allergens: {', '.join(allergens)}")
    if nutrition:
        lines.append(f"nutrition: {_format_nutrition(nutrition)}")
    if price is not None:
        lines.append(f"price: {price:,}원")

    # 본문 토큰 점수는 이 텍스트로만 계산한다. 'nutrition', '단백질' 같은 항목 이름 때문에
    # 모든 문서가 똑같이 점수를 얻는 것을 막기 위해서다.
    match_text = " ".join(part for part in (name, *aliases, *foreign_names, category, description, *allergens) if part)
    return _create_document(
        page_content="\n".join(lines),
        metadata={
            "menu_id": menu_id,
            "name": name,
            "category": category,
            "tags": [],
            "search_terms": list(search_terms),
            "match_text": match_text,
        },
    )


def build_menu_documents() -> list[MenuDocument]:
    """
    현재 서버가 알고 있는 메뉴 목록을 RAG 검색용 문서로 만든다.
    백엔드 동기화 전(static)에는 기존 고정 프로필을, 동기화 후에는 실제 메뉴를 쓴다.
    """
    if llm_service.MENU_SOURCE == "static":
        return [
            _profile_to_document(profile)
            for profile in MENU_RECOMMENDATION_PROFILES
            if profile["menu_id"] in llm_service.MENU_CATALOG
        ]
    return [_live_document(menu_id, name) for menu_id, name in llm_service.MENU_CATALOG.items()]


def _embedding_source(document: MenuDocument) -> str:
    """임베딩 인덱스에 넣을 메뉴 메타데이터 텍스트. 설명문은 일부러 빼고 정형 필드만 쓴다."""
    return " ".join(document.metadata["search_terms"])


_MENU_DOCUMENTS: list[MenuDocument] = []
_MENU_EMBEDDING_INDEX: MenuEmbeddingIndex | None = None
_menu_state: tuple[list[MenuDocument], MenuEmbeddingIndex, tuple] | None = None
_menu_state_lock = threading.Lock()


def _menu_state_key() -> tuple:
    return (llm_service.MENU_SOURCE, llm_service.MENU_VERSION)


def _get_menu_state() -> tuple[list[MenuDocument], MenuEmbeddingIndex]:
    """
    (문서 목록, 임베딩 인덱스)를 돌려준다.
    메뉴가 동기화돼서 버전이 바뀌었을 때만 다시 만든다.
    """
    global _MENU_DOCUMENTS, _MENU_EMBEDDING_INDEX, _menu_state

    state = _menu_state
    if state is not None and state[2] == _menu_state_key():
        return state[0], state[1]

    with _menu_state_lock:
        key = _menu_state_key()
        state = _menu_state
        if state is None or state[2] != key:
            documents = build_menu_documents()
            index = MenuEmbeddingIndex(
                [(document.metadata["menu_id"], _embedding_source(document)) for document in documents]
            )
            state = (documents, index, key)
            _menu_state = state
            _MENU_DOCUMENTS = documents
            _MENU_EMBEDDING_INDEX = index
            logger.info("[RAG] 추천 문서 %d건 생성 (source=%s, version=%s)", len(documents), key[0], key[1])
    return state[0], state[1]


_get_menu_state()


def _score_document(
    query: str,
    query_vector: dict[int, float],
    document: MenuDocument,
    index: MenuEmbeddingIndex | None = None,
) -> int:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0

    score = 0
    for term in document.metadata["search_terms"]:
        normalized_term = _normalize_text(term)
        if normalized_term and normalized_term in normalized_query:
            score += 10 + len(normalized_term)

    # 동기화된 실제 메뉴(match_text 있음)는 이름/카테고리/설명에 들어 있는 단어가 더 믿을 만한 신호라 가중치를 높인다.
    match_text = document.metadata.get("match_text")
    token_weight = 3 if match_text else 1
    content = _normalize_text(match_text or document.page_content)
    for token in re.findall(r"[0-9A-Za-z가-힣]+", query.lower()):
        if len(token) >= 2 and token in content:
            score += token_weight

    similarity = (index or _MENU_EMBEDDING_INDEX).similarity(query_vector, document.metadata["menu_id"])
    score += embedding_boost(similarity)

    return score


def _looks_like_recommendation_query(query: str) -> bool:
    normalized_query = _normalize_text(query)
    return any(term in normalized_query for term in RECOMMENDATION_INTENT_TERMS)


def _contains_hangul(text: str) -> bool:
    return re.search(r"[가-힣]", text) is not None


# 메뉴가 이 개수 이하이면 점수와 상관없이 후보를 전부 LLM에 넘긴다.
# (점수순 상위 몇 개만 넘기면 "추천해줘"처럼 점수가 같은 질문에서 앞쪽 메뉴만 보게 된다.)
FULL_POOL_LIMIT = 20
# 메뉴가 더 많아지면 점수순 상위 이 개수만 넘긴다.
CANDIDATE_POOL_SIZE = 8


def _pool_limit(total: int, top_k: int) -> int:
    if total <= FULL_POOL_LIMIT:
        return total
    return max(top_k, CANDIDATE_POOL_SIZE)


def retrieve_menu_documents(query: str, top_k: int = 3) -> list[MenuDocument]:
    """
    메뉴 문서 검색.
    제5주차 RAG 검색 최적화: 키워드 매치 + 본문 토큰 + 메뉴 메타데이터 기반
    해시 TF-IDF 코사인 유사도를 합쳐서 후보 순서를 만든다.

    LLM에 넘길 후보 풀을 돌려주므로 top_k보다 많을 수 있다.
    최종 top_k 개수 제한은 LLM 응답을 파싱할 때 적용한다.
    """
    query = query.strip()
    if not query:
        return []

    documents, embedding_index = _get_menu_state()
    query_vector = embedding_index.query_vector(query)
    scored = [
        (_score_document(query, query_vector, document, embedding_index), index, document)
        for index, document in enumerate(documents)
    ]
    matched = [item for item in scored if item[0] > 0]

    # 한글이 없는 질문(일본어/스페인어/태국어/영어 등)은 점수가 신뢰하기 어렵다.
    # (해시 임베딩은 한글/영문 bigram만 보고, 영문 단어가 우연히 겹쳐서 일부 메뉴만 걸리기도 한다.)
    # 그래서 점수 순서로 정렬은 하되 전체 메뉴를 후보로 넘기고, LLM이 원문 질문을 보고 고르게 한다.
    # "추천해줘"처럼 조건이 아직 넓은 한국어 문장도 같은 방식으로 전체를 넘긴다.
    if not _contains_hangul(query):
        candidates = scored
    elif matched:
        candidates = matched
    elif _looks_like_recommendation_query(query):
        candidates = scored
    else:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    limit = _pool_limit(len(documents), top_k)
    return [document for _, _, document in candidates[:limit]]


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
        name=llm_service.MENU_CATALOG.get(menu_id) or document.metadata["name"],
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
        **completion_options(),
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