import asyncio
import unittest
from unittest.mock import patch

from app.schemas import MenuRecommendationItem
from app.services import voice_recommendation as V


class RecommendationIntentTest(unittest.TestCase):
    def test_recommendation_requests(self):
        for text in [
            "산뜻한 거 추천해줘", "뭐가 맛있어요?", "뭐 먹을까", "음료 뭐 마실까",
            "おすすめは何ですか", "¿Qué me recomiendas?", "แนะนำเมนูหน่อย",
        ]:
            self.assertTrue(V.is_recommendation_request(text), text)

    def test_orders_are_not_recommendation_requests(self):
        for text in ["데리버거 두 개 주세요", "콜라 하나 치즈 추가", "コーラをふたつください", "dos colas por favor"]:
            self.assertFalse(V.is_recommendation_request(text), text)

    def test_preference_cue(self):
        self.assertTrue(V.has_preference_cue("산뜻한 거 있어요?"))
        self.assertTrue(V.has_preference_cue("싼 거 주세요"))
        self.assertTrue(V.has_preference_cue("기름진 거"))
        self.assertFalse(V.has_preference_cue("고마워"))
        self.assertFalse(V.has_preference_cue("아이스아메리카노"))


class RecommendForVoiceTest(unittest.TestCase):
    def test_llm_result_is_used(self):
        async def fake(query, top_k=3, language="ko"):
            return {"status": "success", "recommendations": [MenuRecommendationItem(menu_id="2", name="리아 새우버거", reason="산뜻함")]}

        with patch.object(V.recommendation_service, "recommend_menus", fake):
            result = asyncio.run(V.recommend_for_voice("산뜻한 거 추천해줘"))
        self.assertEqual(result["source"], "llm")
        self.assertEqual([i.menu_id for i in result["items"]], ["2"])

    def test_slow_llm_falls_back_within_budget(self):
        async def slow(query, top_k=3, language="ko"):
            await asyncio.sleep(5)
            return {"status": "success", "recommendations": []}

        rule_items = [MenuRecommendationItem(menu_id="5", name="핫크리스피 치킨버거", reason="'매콤함' 특징이 있는 메뉴입니다.")]
        with patch.object(V, "VOICE_RECOMMENDATION_BUDGET_SECONDS", 0.05), \
                patch.object(V.recommendation_service, "recommend_menus", slow), \
                patch.object(V, "_rule_based", return_value=rule_items):
            result = asyncio.run(V.recommend_for_voice("매운 거 추천해줘"))
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["items"], rule_items)

    def test_empty_llm_result_uses_rules(self):
        async def empty(query, top_k=3, language="ko"):
            return {"status": "success", "recommendations": []}

        rule_items = [MenuRecommendationItem(menu_id="14", name="콜라", reason="가격이 낮은 순으로 골랐습니다.")]
        with patch.object(V.recommendation_service, "recommend_menus", empty), \
                patch.object(V, "_rule_based", return_value=rule_items):
            result = asyncio.run(V.recommend_for_voice("싼 거 추천"))
        self.assertEqual(result["items"], rule_items)

    def test_language_is_passed_to_recommendation(self):
        seen = {}

        async def fake(query, top_k=3, language="ko"):
            seen["language"] = language
            return {"status": "success", "recommendations": [MenuRecommendationItem(menu_id="2", name="리아 새우버거", reason="さっぱり")]}

        with patch.object(V.recommendation_service, "recommend_menus", fake):
            asyncio.run(V.recommend_for_voice("おすすめは?", language="ja"))
        self.assertEqual(seen["language"], "ja")

    def test_message_shape(self):
        result = {"source": "llm", "items": [MenuRecommendationItem(menu_id="2", name="리아 새우버거", reason="산뜻함")]}
        message = V.recommendation_message("산뜻한 거 추천해줘", "ko", result)
        self.assertEqual(message, {
            "type": "recommendation",
            "recognized_text": "산뜻한 거 추천해줘",
            "language": "ko",
            "source": "llm",
            "items": [{"menu_id": "2", "name": "리아 새우버거", "reason": "산뜻함"}],
        })
        self.assertNotIn("status", message)


if __name__ == "__main__":
    unittest.main()
