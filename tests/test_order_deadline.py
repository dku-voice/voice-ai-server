import asyncio
import unittest
from unittest.mock import patch

from app.schemas import OrderItem
from app.services import order_deadline as D


class OrderDeadlineTest(unittest.TestCase):
    def test_budget_uses_remaining_time(self):
        with patch.object(D, "VOICE_RESPONSE_DEADLINE_SECONDS", 11.0), \
                patch.object(D, "VOICE_RECOMMENDATION_MIN_SECONDS", 1.0):
            self.assertAlmostEqual(D.order_budget(5.0), 6.0)
            self.assertEqual(D.order_budget(10.8), 1.0)

    def test_fast_result_is_returned(self):
        async def fast(text, language):
            return {"status": "success", "items": [OrderItem(menu_id="14", quantity=2, options=[])], "error_msg": None}

        result = asyncio.run(D.extract_order_within_deadline(fast, "コーラをふたつ", "ja", 2.0))
        self.assertEqual(result["status"], "success")

    def test_slow_llm_falls_back_to_keywords(self):
        async def slow(text, language):
            await asyncio.sleep(5)
            return {"status": "success", "items": [], "error_msg": None}

        seen = {}

        def keywords(text, *, language="ko", log_result=True):
            seen["language"] = language
            return [OrderItem(menu_id="14", quantity=2, options=[])]

        with patch.object(D, "order_budget", return_value=0.05), \
                patch.object(D.llm_service, "_fallback_keyword_parse", side_effect=keywords):
            result = asyncio.run(D.extract_order_within_deadline(slow, "コーラをふたつ", "ja", 10.0))
        self.assertEqual(result["status"], "fallback")
        self.assertEqual([(i.menu_id, i.quantity) for i in result["items"]], [("14", 2)])
        self.assertEqual(seen["language"], "ja")


if __name__ == "__main__":
    unittest.main()
