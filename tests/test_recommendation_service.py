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


class RecommendationServiceTest(unittest.TestCase):
    def test_retrieve_side_and_drink_documents(self):
        docs = retrieve_menu_documents("사이드랑 음료 추천해줘", top_k=2)
        menu_ids = [doc.metadata["menu_id"] for doc in docs]

        self.assertIn("fries_01", menu_ids)
        self.assertIn("cola_01", menu_ids)

    def test_retrieve_broad_recommendation_returns_default_candidates(self):
        docs = retrieve_menu_documents("메뉴 추천해줘", top_k=2)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["menu_id"], "burger_01")

    def test_retrieve_unrelated_query_returns_no_documents(self):
        docs = retrieve_menu_documents("오늘 날씨 어때", top_k=3)

        self.assertEqual(docs, [])

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
        docs = retrieve_menu_documents("음료 추천해줘", top_k=1)
        raw = '{"recommendations": [{"menu_id": "burger_01", "reason": "없는 후보입니다."}]}'

        with self.assertRaises(LLMResponseValidationError):
            _parse_recommendation_response(raw, docs, top_k=1)

    def test_parse_recommendation_response_uses_server_side_reason(self):
        docs = retrieve_menu_documents("음료 추천해줘", top_k=1)
        raw = '{"recommendations": [{"menu_id": "cola_01", "reason": "100원이고 매운맛입니다."}]}'

        recommendations = _parse_recommendation_response(raw, docs, top_k=1, query="음료 추천해줘")

        self.assertEqual(recommendations[0].menu_id, "cola_01")
        self.assertNotIn("100원", recommendations[0].reason)
        self.assertNotIn("매운맛", recommendations[0].reason)
        self.assertIn("음료", recommendations[0].reason)

    def test_fallback_recommendation_uses_retrieved_documents(self):
        docs = retrieve_menu_documents("음료 추천해줘", top_k=1)

        recommendations = _fallback_recommendations("음료 추천해줘", docs, top_k=1)

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].menu_id, "cola_01")
        self.assertIn("음료", recommendations[0].reason)


class RecommendationServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_recommend_menus_falls_back_when_llm_is_not_configured(self):
        with patch(
            "app.services.recommendation_service._call_recommendation_llm_sync",
            side_effect=LLMConfigurationError("missing key"),
        ):
            result = await recommend_menus("음료 추천해줘", top_k=1)

        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["retrieved_menu_ids"], ["cola_01"])
        self.assertEqual(result["recommendations"][0].menu_id, "cola_01")

    async def test_recommend_menus_returns_error_for_empty_query(self):
        result = await recommend_menus("   ", top_k=1)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["recommendations"], [])

    async def test_recommend_menus_returns_error_for_unrelated_query(self):
        result = await recommend_menus("오늘 날씨 어때", top_k=3)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["retrieved_menu_ids"], [])


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
