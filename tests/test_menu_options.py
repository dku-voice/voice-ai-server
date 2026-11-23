import unittest

from app.services import llm_service, menu_sync

PAYLOAD = {
    "success": True,
    "data": {
        "menus": [
            {"menuId": 1, "nameKo": "데리버거", "status": "ACTIVE", "options": [
                {"nameKo": "치즈 추가"}, {"nameKo": "토마토 제거"}, {"nameKo": "양파 제거"}, {"nameKo": "소스 적게"},
                {"nameKo": "비프패티 추가"}, {"nameKo": "반숙계란 추가"}, {"nameKo": "베이컨 추가"}, {"nameKo": "토마토 추가"},
            ]},
            {"menuId": 14, "nameKo": "콜라", "status": "ACTIVE", "options": []},
            {"menuId": 20, "nameKo": "치즈스틱", "status": "ACTIVE"},
        ]
    },
}


class MenuOptionTest(unittest.TestCase):
    def setUp(self):
        self._saved = (
            llm_service.MENU_CATALOG,
            llm_service.MENU_ALIASES,
            llm_service.MENU_FOREIGN_NAMES,
            llm_service.MENU_NAMES_BY_LANGUAGE,
            llm_service.MENU_OPTIONS_BY_ID,
            llm_service.MENU_SOURCE,
        )
        catalog, aliases, foreign = menu_sync.build_menu_catalog(PAYLOAD)
        llm_service.apply_menu_catalog(
            catalog, aliases, foreign, "backend",
            menu_sync.build_names_by_language(PAYLOAD, catalog),
            menu_sync.build_options_by_id(PAYLOAD, catalog),
        )

    def tearDown(self):
        (
            llm_service.MENU_CATALOG,
            llm_service.MENU_ALIASES,
            llm_service.MENU_FOREIGN_NAMES,
            llm_service.MENU_NAMES_BY_LANGUAGE,
            llm_service.MENU_OPTIONS_BY_ID,
            llm_service.MENU_SOURCE,
        ) = self._saved

    def parse(self, text):
        return [(i.menu_id, i.quantity, i.options) for i in llm_service._fallback_keyword_parse(text, log_result=False)]

    def test_cheese_add_is_option_not_cheese_sticks(self):
        self.assertEqual(self.parse("데리버거 하나에 치즈 추가"), [("1", 1, ["치즈 추가"])])

    def test_multiple_options(self):
        self.assertEqual(
            self.parse("데리버거 두 개 토마토 빼고 양파도 빼주세요 소스는 적게"),
            [("1", 2, ["토마토 제거", "양파 제거", "소스 적게"])],
        )

    def test_add_options_with_aliases(self):
        self.assertEqual(
            self.parse("데리버거 패티 추가 반숙 계란 추가 베이컨 추가 토마토 추가"),
            [("1", 1, ["비프패티 추가", "반숙계란 추가", "베이컨 추가", "토마토 추가"])],
        )

    def test_option_attaches_to_preceding_menu(self):
        self.assertEqual(
            self.parse("데리버거 치즈 추가 콜라 두 잔"),
            [("1", 1, ["치즈 추가"]), ("14", 2, [])],
        )

    def test_option_not_supported_by_menu_is_dropped(self):
        self.assertEqual(self.parse("콜라 치즈 추가"), [("14", 1, [])])

    def test_cheese_sticks_still_ordered(self):
        self.assertEqual(self.parse("치즈스틱 두 개"), [("20", 2, [])])

    def test_llm_options_are_canonicalized(self):
        item = llm_service._coerce_order_item({"menu_id": "1", "quantity": 1, "options": ["양파 빼주세요", "치즈추가", "unknown"]})
        self.assertEqual(item.options, ["양파 제거", "치즈 추가"])

    def test_sauce_options_are_detected_by_name(self):
        self.assertEqual(self.parse("치즈스틱 칠리 소스로 주세요"), [("20", 1, ["칠리 소스"])])

    def test_korean_hotwords_include_options(self):
        names = llm_service.menu_names_for_language("ko")
        self.assertIn("치즈 추가", names)
        self.assertIn("치즈스틱", names)


if __name__ == "__main__":
    unittest.main()
