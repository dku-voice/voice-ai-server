import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.recommendations import router
from app.schemas import MenuRecommendationItem
from app.services.llm_service import LLMConfigurationError, LLMResponseValidationError
from app.services.recommendation_service import (
    _fallback_recommendations,
    _parse_recommendation_response,
    recommend_menus,
    retrieve_menu_documents,
)
from app.services.recommendation_service import _create_document


class RecommendationServiceTest(unittest.TestCase):
    def test_retrieve_side_and_drink_documents(self):
        docs = retrieve_menu_documents("사이드랑 음료 추천해줘", top_k=2)
        menu_ids = [doc.metadata["menu_id"] for doc in docs]

        self.assertIn("fries_01", menu_ids)
        self.assertIn("cola_01", menu_ids)

    def test_retrieve_broad_recommendation_returns_default_candidates(self):
        # 메뉴가 적으면 top_k와 상관없이 전체 메뉴를 점수순 후보로 넘긴다.
        docs = retrieve_menu_documents("메뉴 추천해줘", top_k=2)

        self.assertGreaterEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["menu_id"], "burger_01")

    def test_retrieve_unrelated_query_returns_all_documents(self):
        # 키워드가 하나도 안 걸려도 후보를 비우지 않고 전체를 넘긴다. (판단은 LLM이 한다)
        docs = retrieve_menu_documents("오늘 날씨 어때", top_k=3)
        menu_ids = {doc.metadata["menu_id"] for doc in docs}

        self.assertEqual(menu_ids, {"burger_01", "cola_01", "fries_01", "chicken_01"})

    def test_retrieve_query_without_keyword_does_not_drop_menus(self):
        # 이름/설명에 없는 말("싼 거" 등)이어도 후보가 0개가 되지 않는다.
        docs = retrieve_menu_documents("싼 거 추천해줘", top_k=3)

        self.assertGreaterEqual(len(docs), 3)

    def test_retrieve_partial_alias_uses_embedding(self):
        """`후렌치`처럼 별칭 일부만 입력해도 임베딩 boost로 fries 후보가 가장 앞에 온다."""
        docs = retrieve_menu_documents("후렌치 줘", top_k=1)
        menu_ids = [doc.metadata["menu_id"] for doc in docs]

        self.assertEqual(menu_ids[0], "fries_01")

    def test_retrieve_main_meal_query_prefers_main_category(self):
        """`메인 식사`류 쿼리는 main 카테고리 메뉴를 위쪽에 둔다."""
        docs = retrieve_menu_documents("메인 식사로 든든한 거", top_k=3)
        menu_ids = [doc.metadata["menu_id"] for doc in docs]

        self.assertIn("burger_01", menu_ids)
        self.assertNotIn(menu_ids[0], {"cola_01", "fries_01"})

    def test_retrieve_returns_candidate_pool_at_least_top_k(self):
        # 검색은 LLM에 넘길 후보 풀을 돌려주므로 top_k 이상이다. 최종 top_k 제한은 응답 파싱 단계에서 적용된다.
        docs = retrieve_menu_documents("메뉴 추천해줘", top_k=3)

        self.assertGreaterEqual(len(docs), 3)

    def test_parse_recommendation_response_fills_name_and_deduplicates(self):
        docs = retrieve_menu_documents("사이드랑 음료 추천해줘", top_k=2)
        raw = """
        {
          "recommendations": [
            {"menu_id": "fries_01", "reason": "사이드 요청과 맞습니다."},
            {"menu_id": "fries_01", "reason": "중복입니다."},
            {"menu_id": "cola_01", "reason": "음료 요청과 맞습니다."}
          ]
        }
        """

        recommendations = _parse_recommendation_response(raw, docs, top_k=3)

        self.assertEqual([item.menu_id for item in recommendations], ["fries_01", "cola_01"])
        self.assertEqual(recommendations[0].name, "감자튀김")

    def test_parse_recommendation_response_rejects_menu_outside_retrieval(self):
        # 후보 풀이 전체 메뉴이므로, 후보가 한 개뿐인 상황을 직접 만들어 후보 밖 menu_id가 거부되는지 본다.
        docs = [doc for doc in retrieve_menu_documents("음료 추천해줘", top_k=1) if doc.metadata["menu_id"] == "cola_01"]
        raw = '{"recommendations": [{"menu_id": "burger_01", "reason": "없는 후보입니다."}]}'

        with self.assertRaises(LLMResponseValidationError):
            _parse_recommendation_response(raw, docs, top_k=1)

    def test_parse_recommendation_response_uses_llm_reason(self):
        docs = retrieve_menu_documents("음료 추천해줘", top_k=1)
        raw = '{"recommendations": [{"menu_id": "cola_01", "reason": "  음료 요청과\\n맞는 메뉴입니다.  "}]}'

        recommendations = _parse_recommendation_response(raw, docs, top_k=1, query="음료 추천해줘")

        self.assertEqual(recommendations[0].menu_id, "cola_01")
        # LLM이 준 근거를 쓰되 공백/줄바꿈은 정리한다.
        self.assertEqual(recommendations[0].reason, "음료 요청과 맞는 메뉴입니다.")

    def test_parse_recommendation_response_uses_fallback_reason_when_missing(self):
        docs = retrieve_menu_documents("음료 추천해줘", top_k=1)
        raw = '{"recommendations": [{"menu_id": "cola_01", "reason": "  "}, {"menu_id": "fries_01"}]}'

        recommendations = _parse_recommendation_response(raw, docs, top_k=2, query="음료 추천해줘")

        self.assertEqual([item.menu_id for item in recommendations], ["cola_01", "fries_01"])
        self.assertIn("음료", recommendations[0].reason)
        self.assertTrue(recommendations[1].reason)

    def test_parse_recommendation_response_limits_reason_length(self):
        docs = retrieve_menu_documents("음료 추천해줘", top_k=1)
        raw = '{"recommendations": [{"menu_id": "cola_01", "reason": "' + ("가" * 500) + '"}]}'

        recommendations = _parse_recommendation_response(raw, docs, top_k=1, query="음료 추천해줘")

        self.assertEqual(len(recommendations[0].reason), 120)

    def test_fallback_recommendation_uses_retrieved_documents(self):
        docs = retrieve_menu_documents("음료 추천해줘", top_k=1)

        recommendations = _fallback_recommendations("음료 추천해줘", docs, top_k=1)

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].menu_id, "cola_01")
        self.assertIn("음료", recommendations[0].reason)


def _live_like_document(menu_id, name, category, features, price, nutrition=None):
    return _create_document(
        page_content=f"menu_id: {menu_id}\nname: {name}",
        metadata={
            "menu_id": menu_id,
            "name": name,
            "category": category,
            "tags": [],
            "features": tuple(features),
            "price": price,
            "nutrition": nutrition or {},
            "search_terms": [name, category],
            "match_text": f"{name} {category}",
        },
    )


class FallbackSelectionTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            _live_like_document(
                "1", "데리버거", "버거", ["달콤함", "아이 추천"], 2800,
                {"칼로리": 356, "지방": 16, "단백질": 12, "나트륨": 590},
            ),
            _live_like_document(
                "5", "핫크리스피 치킨버거", "버거", ["매콤함", "바삭함"], 4500,
                {"칼로리": 503, "지방": 7.3, "단백질": 22, "나트륨": 900},
            ),
            _live_like_document(
                "9", "양념감자", "사이드", ["바삭함", "아이 추천"], 2700,
                {"칼로리": 359, "지방": 4.3, "단백질": 5, "나트륨": 520},
            ),
            _live_like_document("14", "콜라", "음료", ["달콤함"], 2100, {"칼로리": 128, "지방": 0, "나트륨": 15}),
            _live_like_document("17", "아이스아메리카노", "음료", [], 2600, {"칼로리": 5, "지방": 0, "나트륨": 0}),
        ]

    def _ids(self, query, top_k=3):
        return [item.menu_id for item in _fallback_recommendations(query, self.docs, top_k)]

    def test_fallback_picks_by_feature_tag(self):
        self.assertEqual(self._ids("매운 거 추천해줘"), ["5"])

    def test_fallback_picks_cheapest_first_for_price_query(self):
        self.assertEqual(self._ids("싼 거"), ["14", "17", "9"])

    def test_fallback_price_query_respects_category(self):
        self.assertEqual(self._ids("싼 버거 추천해줘"), ["1", "5"])

    def test_fallback_child_query_ignores_iced_drink_name(self):
        # "아이스"는 "아이" 조건으로 보지 않는다.
        self.assertEqual(self._ids("아이스 아메리카노 추천해줘"), ["1", "5", "9"])  # 태그로 좁혀지지 않음
        self.assertEqual(self._ids("아이가 먹기 좋은 거"), ["1", "9"])

    def test_fallback_keeps_score_order_without_matching_condition(self):
        self.assertEqual(self._ids("추천해줘"), ["1", "5", "9"])

    def test_fallback_orders_by_fat_for_greasy_query(self):
        # 음료는 카테고리를 말하지 않으면 제외하고, 지방이 높은 순으로 고른다.
        self.assertEqual(self._ids("기름진 거"), ["1", "5", "9"])

    def test_fallback_orders_by_low_fat_for_light_query(self):
        self.assertEqual(self._ids("담백한 거"), ["9", "5", "1"])

    def test_fallback_low_fat_query_for_drinks_keeps_drinks(self):
        self.assertEqual(self._ids("담백한 음료"), ["14", "17"])

    def test_fallback_orders_by_sodium_for_salty_query(self):
        self.assertEqual(self._ids("짭짤한 거"), ["5", "1", "9"])

    def test_fallback_orders_by_protein_and_skips_menus_without_value(self):
        # 콜라/아메리카노는 단백질 수치가 없어서 빠진다.
        self.assertEqual(self._ids("단백질 높은 거", top_k=5), ["5", "1", "9"])

    def test_fallback_nutrition_query_respects_category(self):
        self.assertEqual(self._ids("기름진 버거 추천해줘"), ["1", "5"])

    def test_fallback_reason_mentions_nutrition_value(self):
        item = _fallback_recommendations("기름진 거", self.docs, 1)[0]

        self.assertIn("지방", item.reason)
        self.assertIn("16g", item.reason)

    def test_fallback_reason_mentions_tag_or_price(self):
        tag_item = _fallback_recommendations("매운 거", self.docs, 1)[0]
        price_item = _fallback_recommendations("싼 거", self.docs, 1)[0]

        self.assertIn("매콤함", tag_item.reason)
        self.assertIn("2,100원", price_item.reason)


class RecommendationServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_recommend_menus_falls_back_when_llm_is_not_configured(self):
        with patch(
            "app.services.recommendation_service._call_recommendation_llm_sync",
            side_effect=LLMConfigurationError("missing key"),
        ):
            result = await recommend_menus("음료 추천해줘", top_k=1)

        self.assertEqual(result["status"], "fallback")
        # 후보는 전체 메뉴이고, 점수가 가장 높은 음료가 맨 앞에 온다.
        self.assertEqual(result["retrieved_menu_ids"][0], "cola_01")
        self.assertEqual(result["recommendations"][0].menu_id, "cola_01")

    async def test_recommend_menus_falls_back_when_llm_response_is_invalid_json(self):
        # LLM이 좋지 않은 JSON을 돌려줘도 검색된 menu_id 안에서 fallback이 나와야 한다.
        with patch(
            "app.services.recommendation_service._call_recommendation_llm_sync",
            side_effect=LLMResponseValidationError("invalid json"),
        ):
            result = await recommend_menus("음료 추천해줘", top_k=1)

        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["retrieved_menu_ids"][0], "cola_01")
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0].menu_id, "cola_01")
        self.assertIn("LLM", result["error_msg"])

    async def test_recommend_menus_returns_error_for_empty_query(self):
        result = await recommend_menus("   ", top_k=1)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["recommendations"], [])

    async def test_recommend_menus_passes_all_candidates_for_unrelated_query(self):
        # 무관한 질문도 후보를 비우지 않고 LLM에 넘긴다. (이전에는 후보가 없어서 바로 error였다.)
        # LLM이 설정되지 않은 환경을 흉내 내서 후보 구성만 확인한다.
        with patch(
            "app.services.recommendation_service._call_recommendation_llm_sync",
            side_effect=LLMConfigurationError("missing key"),
        ):
            result = await recommend_menus("오늘 날씨 어때", top_k=3)

        self.assertEqual(result["status"], "fallback")
        self.assertEqual(
            set(result["retrieved_menu_ids"]),
            {"burger_01", "cola_01", "fries_01", "chicken_01"},
        )


class RecommendationApiTest(unittest.TestCase):
    def test_recommendation_route_returns_service_result(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        service_result = {
            "status": "success",
            "query": "음료 추천해줘",
            "recommendations": [
                MenuRecommendationItem(
                    menu_id="cola_01",
                    name="콜라",
                    reason="음료 요청과 맞습니다.",
                )
            ],
            "retrieved_menu_ids": ["cola_01"],
            "error_msg": None,
        }

        with patch(
            "app.api.recommendations.recommend_menus",
            new=AsyncMock(return_value=service_result),
        ):
            response = client.post(
                "/api/recommendations/menu",
                json={"query": "음료 추천해줘", "top_k": 1},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["recommendations"][0]["menu_id"], "cola_01")

    def test_recommendation_route_rejects_too_long_query(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/api/recommendations/menu",
            json={"query": "가" * 501, "top_k": 1},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()