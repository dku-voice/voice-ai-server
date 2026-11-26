import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm_service import (
    LLMResponseValidationError,
    _call_llm_sync,
    _fallback_keyword_parse,
    _parse_llm_response,
)


class LLMServiceValidationTest(unittest.TestCase):
    def test_parse_markdown_json_and_normalize_fields(self):
        raw = """```json
        [{"menu_id": "burger_01", "quantity": "두 개", "options": "양파 빼주세요"}]
        ```"""

        items = _parse_llm_response(raw, "버거 두 개 양파 빼주세요")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].menu_id, "burger_01")
        self.assertEqual(items[0].quantity, 2)
        self.assertEqual(items[0].options, ["양파 제거"])

    def test_reject_unknown_menu_id(self):
        raw = '[{"menu_id": "unknown_01", "quantity": 1, "options": []}]'

        with self.assertRaises(LLMResponseValidationError):
            _parse_llm_response(raw, "햄버거 하나 주세요")

    def test_reject_llm_result_when_keyword_menu_is_missing(self):
        raw = '[{"menu_id": "burger_01", "quantity": 1, "options": []}]'

        with self.assertRaises(LLMResponseValidationError):
            _parse_llm_response(raw, "햄버거 하나랑 콜라 하나 주세요")

    def test_reject_llm_result_when_menu_has_no_text_evidence(self):
        raw = (
            '[{"menu_id": "burger_01", "quantity": 1, "options": []}, '
            '{"menu_id": "cola_01", "quantity": 1, "options": []}]'
        )

        with self.assertRaises(LLMResponseValidationError):
            _parse_llm_response(raw, "햄버거 하나 주세요")

    def test_reject_llm_result_when_quantity_does_not_match_text(self):
        raw = '[{"menu_id": "cola_01", "quantity": 20, "options": []}]'

        with self.assertRaises(LLMResponseValidationError):
            _parse_llm_response(raw, "콜라 하나 주세요")

    def test_keyword_fallback_handles_before_and_after_quantities(self):
        items = _fallback_keyword_parse("콜라 두 개랑 버거 하나 주세요", log_result=False)
        by_menu = {item.menu_id: item.quantity for item in items}

        self.assertEqual(by_menu["cola_01"], 2)
        self.assertEqual(by_menu["burger_01"], 1)

    def test_keyword_fallback_handles_quantity_before_menu(self):
        items = _fallback_keyword_parse("두 개 콜라 주세요", log_result=False)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].menu_id, "cola_01")
        self.assertEqual(items[0].quantity, 2)

    def test_keyword_fallback_handles_multi_digit_quantity_after_menu(self):
        cases = {
            "콜라 12개 주세요": 12,
            "콜라 20개 주세요": 20,
        }

        for text, expected_quantity in cases.items():
            with self.subTest(text=text):
                items = _fallback_keyword_parse(text, log_result=False)

                self.assertEqual(len(items), 1)
                self.assertEqual(items[0].menu_id, "cola_01")
                self.assertEqual(items[0].quantity, expected_quantity)

    def test_keyword_fallback_handles_multi_digit_quantity_before_menu(self):
        cases = {
            "12개 콜라 주세요": 12,
            "20개 콜라 주세요": 20,
        }

        for text, expected_quantity in cases.items():
            with self.subTest(text=text):
                items = _fallback_keyword_parse(text, log_result=False)

                self.assertEqual(len(items), 1)
                self.assertEqual(items[0].menu_id, "cola_01")
                self.assertEqual(items[0].quantity, expected_quantity)

    def test_keyword_fallback_rejects_out_of_range_multi_digit_quantity(self):
        cases = [
            "콜라 21개 주세요",
            "21개 콜라 주세요",
        ]

        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(LLMResponseValidationError):
                    _fallback_keyword_parse(text, log_result=False)

    def test_keyword_fallback_merges_repeated_menu_mentions(self):
        items = _fallback_keyword_parse("콜라 하나 콜라 두 개 주세요", log_result=False)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].menu_id, "cola_01")
        self.assertEqual(items[0].quantity, 3)

    def test_keyword_fallback_skips_excluded_menu_mentions(self):
        cases = [
            "햄버거 하나 콜라 빼고 주세요",
            "콜라 말고 햄버거 하나 주세요",
            "콜라는 빼고 감자튀김 주세요",
            "콜라 제외하고 햄버거 하나 주세요",
            "노콜라 버거 하나 주세요",
        ]

        for text in cases:
            with self.subTest(text=text):
                items = _fallback_keyword_parse(text, log_result=False)
                menu_ids = {item.menu_id for item in items}

                self.assertNotIn("cola_01", menu_ids)

    def test_llm_validation_allows_response_without_excluded_menu(self):
        raw = '[{"menu_id": "burger_01", "quantity": 1, "options": []}]'

        items = _parse_llm_response(raw, "햄버거 하나 콜라 빼고 주세요")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].menu_id, "burger_01")

    def test_keyword_fallback_does_not_read_igeo_as_quantity_two(self):
        items = _fallback_keyword_parse("콜라 이거 주세요", log_result=False)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].menu_id, "cola_01")
        self.assertEqual(items[0].quantity, 1)

    def test_keyword_fallback_does_not_match_fries_inside_fried_chicken(self):
        items = _fallback_keyword_parse("프라이드 치킨 하나 주세요", log_result=False)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].menu_id, "chicken_01")
        self.assertEqual(items[0].quantity, 1)

    def test_common_cola_stt_mishearing_is_allowed_as_text_evidence(self):
        raw = '[{"menu_id": "cola_01", "quantity": 1, "options": []}]'

        items = _parse_llm_response(raw, "코랄 한 개 주세요")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].menu_id, "cola_01")
        self.assertEqual(items[0].quantity, 1)

    def test_empty_json_is_valid_when_text_has_no_menu(self):
        self.assertEqual(_parse_llm_response("[]", "안녕하세요"), [])

    def test_phase2_runs_for_json_parse_error(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                content = "주문은 콜라입니다."
                if self.calls == 2:
                    content = '[{"menu_id": "cola_01", "quantity": 1, "options": []}]'
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        completions = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with patch("app.services.llm_service._get_client", return_value=fake_client):
            phase, items = _call_llm_sync("콜라 하나 주세요")

        self.assertEqual(phase, "success")
        self.assertEqual(completions.calls, 2)
        self.assertEqual(items[0].menu_id, "cola_01")

    def test_phase2_is_skipped_for_semantic_validation_error(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='[{"menu_id": "burger_01", "quantity": 1, "options": []}]'
                            )
                        )
                    ]
                )

        completions = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with patch("app.services.llm_service._get_client", return_value=fake_client):
            phase, items = _call_llm_sync("콜라 하나 주세요")

        self.assertEqual(phase, "fallback")
        self.assertEqual(completions.calls, 1)
        self.assertEqual(items[0].menu_id, "cola_01")

    def test_foreign_language_order_skips_korean_keyword_check(self):
        raw = '[{"menu_id": "cola_01", "quantity": 2, "options": []}]'
        items = _parse_llm_response(raw, "コーラをふたつください", "ja")
        self.assertEqual(items[0].menu_id, "cola_01")
        self.assertEqual(items[0].quantity, 2)

    def test_foreign_language_order_still_rejects_unknown_menu(self):
        raw = '[{"menu_id": "sushi_99", "quantity": 1, "options": []}]'
        with self.assertRaises(LLMResponseValidationError):
            _parse_llm_response(raw, "寿司をください", "ja")

    def test_foreign_language_prompt_includes_language(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                content = '[{"menu_id": "zero-cola", "quantity": 1, "options": []}]'
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        with patch("app.services.llm_service._get_client", return_value=fake_client):
            phase, items = _call_llm_sync("una coca cola zero, por favor", "es")

        self.assertEqual(phase, "success")
        self.assertEqual(items[0].menu_id, "zero-cola")
        self.assertIn("스페인어", captured["messages"][1]["content"])
        self.assertIn("max_tokens", captured)

    def test_korean_order_uses_keyword_matching_without_llm(self):
        import asyncio
        from app.services.llm_service import extract_order

        with patch("app.services.llm_service._get_client", side_effect=AssertionError("LLM called")):
            result = asyncio.run(extract_order("클래식 버거 한 개랑 콜라 두 개", "ko"))

        self.assertEqual(result["status"], "success")
        quantities = {item.menu_id: item.quantity for item in result["items"]}
        self.assertEqual(quantities, {"classic-burger": 1, "cola_01": 2})


    def _korean(self, text, llm_result):
        import asyncio
        from app.services import llm_service

        calls = []

        async def fake_llm(func, *args):
            calls.append(args)
            if isinstance(llm_result, Exception):
                raise llm_result
            return llm_result

        with patch("app.services.llm_service.run_llm_task", side_effect=fake_llm):
            result = asyncio.run(llm_service.extract_order(text, "ko"))
        return result, calls

    def test_korean_simple_order_skips_llm(self):
        result, calls = self._korean("콜라 두 개랑 클래식 버거 하나", ("success", []))
        self.assertEqual(calls, [])
        self.assertEqual(len(result["items"]), 2)

    def test_korean_complex_order_uses_llm_result(self):
        from app.schemas import OrderItem

        llm_items = [OrderItem(menu_id="cola_01", quantity=1, options=[])]
        result, calls = self._korean("콜라 두 개 중에 하나만 주세요", ("success", llm_items))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "success")
        self.assertEqual([(i.menu_id, i.quantity) for i in result["items"]], [("cola_01", 1)])

    def test_korean_unmatched_text_skips_llm(self):
        result, calls = self._korean("코올라 두 개", ("success", []))
        self.assertEqual(calls, [])
        self.assertEqual(result["items"], [])

    def test_korean_llm_failure_keeps_keyword_result(self):
        result, calls = self._korean("콜라 두 개 중에 하나만 주세요", RuntimeError("503"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["items"])

    def test_korean_misheard_burger_skips_llm(self):
        from app.services import llm_service

        cola = llm_service._fallback_keyword_parse("콜라 세 개", log_result=False)
        self.assertIsNone(llm_service.korean_llm_reason("이버거 2개, 콜라 3개", cola))

    def test_korean_preference_request_skips_llm(self):
        result, calls = self._korean("매운 거 있어요?", ("success", []))
        self.assertEqual(calls, [])
        self.assertEqual(result["items"], [])


    def test_recheck_accepts_menu_with_similar_sound(self):
        from app.services.llm_service import _parse_llm_response

        raw = '[{"menu_id": "classic-burger", "quantity": 2, "options": []}, {"menu_id": "cola_01", "quantity": 3, "options": []}]'
        items = _parse_llm_response(raw, "클레식 보거 두 개랑 골라 세 개", "ko", recheck=True)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("classic-burger", 2), ("cola_01", 3)])

    def test_recheck_rejects_menu_without_any_evidence(self):
        from app.services.llm_service import LLMResponseValidationError, _parse_llm_response

        raw = '[{"menu_id": "fries_01", "quantity": 1, "options": []}]'
        with self.assertRaises(LLMResponseValidationError):
            _parse_llm_response(raw, "안녕하세요 반가워요", "ko", recheck=True)

    def test_recheck_allows_dropping_and_changing_quantity(self):
        from app.services.llm_service import _parse_llm_response

        raw = '[{"menu_id": "cola_01", "quantity": 1, "options": []}]'
        items = _parse_llm_response(raw, "감자튀김 말고 콜라 하나만", "ko", recheck=True)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 1)])

    def test_strict_korean_validation_is_unchanged(self):
        from app.services.llm_service import LLMResponseValidationError, _parse_llm_response

        raw = '[{"menu_id": "classic-burger", "quantity": 2, "options": []}]'
        with self.assertRaises(LLMResponseValidationError):
            _parse_llm_response(raw, "클레식 보거 두 개", "ko")


    def test_korean_quantity_after_comma(self):
        from app.services.llm_service import _fallback_keyword_parse

        items = _fallback_keyword_parse("콜라, 네 잔 주세요", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 4)])
        items = _fallback_keyword_parse("콜라, 두 잔 주세요", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 2)])


    def test_korean_quantity_is_not_reused_by_next_menu(self):
        from app.services.llm_service import _fallback_keyword_parse

        items = _fallback_keyword_parse("콜라 두 개 감자튀김 주세요", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 2), ("fries_01", 1)])
        items = _fallback_keyword_parse("두 개 감자튀김", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("fries_01", 2)])


    def test_korean_misheard_three_and_four(self):
        from app.services.llm_service import _fallback_keyword_parse

        items = _fallback_keyword_parse("콜라, 내장, 감자튀김, 새잔 주세요", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 4), ("fries_01", 3)])
        items = _fallback_keyword_parse("콜라 새전 주세요", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 3)])

    def test_korean_menu_with_quantity_before_malgo_is_excluded(self):
        from app.services.llm_service import _fallback_keyword_parse

        items = _fallback_keyword_parse("콜라 두 잔 말고 감자튀김 두 개 주세요", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("fries_01", 2)])
        items = _fallback_keyword_parse("감자튀김 두 개 빼고 콜라 하나", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 1)])


    def test_korean_quantity_correction(self):
        from app.services.llm_service import _fallback_keyword_parse

        cases = {
            "콜라 두 잔 말고 세 잔으로": [("cola_01", 3)],
            "감자튀김 하나 말고 두 개": [("fries_01", 2)],
            "콜라 세 잔, 아니 네 잔": [("cola_01", 4)],
            "콜라 세잔, 아니 콜라 내잔으로": [("cola_01", 4)],
            "감자튀김 두 개 콜라 아니 치킨 하나": [("fries_01", 2), ("chicken_01", 1)],
            "콜라 두 잔 말고 감자튀김 세 개": [("fries_01", 3)],
        }
        for text, expected in cases.items():
            items = _fallback_keyword_parse(text, log_result=False)
            self.assertEqual([(i.menu_id, i.quantity) for i in items], expected, text)


    def test_korean_correction_words_skip_llm(self):
        result, calls = self._korean("클래식 버거 말고 콜라 하나", ("success", []))
        self.assertEqual(calls, [])
        self.assertEqual([(i.menu_id, i.quantity) for i in result["items"]], [("cola_01", 1)])

    def test_korean_quantity_after_ro(self):
        from app.services.llm_service import _fallback_keyword_parse

        items = _fallback_keyword_parse("콜라로 두 잔 주세요", log_result=False)
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("cola_01", 2)])


if __name__ == "__main__":
    unittest.main()
