import unittest

from app.services.llm_service import (
    LLMResponseValidationError,
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
        self.assertEqual(items[0].options, ["양파 빼주세요"])

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


if __name__ == "__main__":
    unittest.main()
