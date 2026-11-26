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
            {"menuId": 9, "nameKo": "양념감자", "nameEn": "Seasoned Fries", "nameJa": "シーズニングポテト",
             "nameEs": "Patatas sazonadas", "nameTh": "มันฝรั่งทอดปรุงรส", "status": "ACTIVE", "options": [
                {"nameKo": "어니언 소스"}, {"nameKo": "치즈 소스"}, {"nameKo": "칠리 소스"},
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

    def test_less_sauce_mishearings(self):
        self.assertEqual(
            self.parse("데리버거, 치즈 추가, 토마토 빼고, 양파도 빼, 그리고 소스도 적혀져"),
            [("1", 1, ["치즈 추가", "토마토 제거", "양파 제거", "소스 적게"])],
        )
        self.assertEqual(self.parse("데리버거 소스 저게 주세요"), [("1", 1, ["소스 적게"])])

    def test_one_trigger_for_several_ingredients(self):
        self.assertEqual(self.parse("데리버거 치즈랑 토마토 추가 해줘"), [("1", 1, ["치즈 추가", "토마토 추가"])])
        self.assertEqual(self.parse("데리버거 치즈하고 베이컨 추가"), [("1", 1, ["치즈 추가", "베이컨 추가"])])
        self.assertEqual(self.parse("데리버거 양파랑 토마토 빼줘"), [("1", 1, ["양파 제거", "토마토 제거"])])
        self.assertEqual(self.parse("데리버거 치즈 빼고 베이컨 추가"), [("1", 1, ["베이컨 추가"])])

    def test_comma_between_ingredient_and_trigger(self):
        self.assertEqual(self.parse("데리버거 양파, 제거"), [("1", 1, ["양파 제거"])])
        self.assertEqual(self.parse("데리버거 토마토랑 양파, 제거하죠"), [("1", 1, ["토마토 제거", "양파 제거"])])
        self.assertEqual(self.parse("데리버거 소스, 적게"), [("1", 1, ["소스 적게"])])

    def test_more_list_connectors(self):
        self.assertEqual(self.parse("데리버거, 비프패티란, 베이컨, 반숙계란 추가"), [("1", 1, ["비프패티 추가", "베이컨 추가", "반숙계란 추가"])])
        self.assertEqual(self.parse("데리버거 치즈하구 베이컨 추가"), [("1", 1, ["치즈 추가", "베이컨 추가"])])
        self.assertEqual(self.parse("데리버거 치즈 또 베이컨 추가"), [("1", 1, ["치즈 추가", "베이컨 추가"])])

    def test_onion_spelled_differently(self):
        self.assertEqual(self.parse("데리버거 노 오니언"), [("1", 1, ["양파 제거"])])

    def test_more_korean_option_expressions(self):
        self.assertEqual(self.parse("데리버거 양파는 넣지 마세요"), [("1", 1, ["양파 제거"])])
        self.assertEqual(self.parse("데리버거 노 어니언"), [("1", 1, ["양파 제거"])])
        self.assertEqual(self.parse("데리버거 토마토 제외"), [("1", 1, ["토마토 제거"])])
        self.assertEqual(self.parse("데리버거 소스 살살"), [("1", 1, ["소스 적게"])])

    def test_option_word_mishearings(self):
        for heard in ("젖어", "적께", "쩍게", "작게"):
            self.assertEqual(self.parse(f"데리버거 소스 {heard}"), [("1", 1, ["소스 적게"])], heard)
        for heard in ("뻬", "재거", "제고"):
            self.assertEqual(self.parse(f"데리버거 양파 {heard}"), [("1", 1, ["양파 제거"])], heard)
        for heard in ("추기", "츄가"):
            self.assertEqual(self.parse(f"데리버거 치즈 {heard}"), [("1", 1, ["치즈 추가"])], heard)

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


class SeasonedFriesSauceTest(MenuOptionTest):
    def items(self, text, language="ko"):
        return [(item.menu_id, item.quantity, item.options) for item in llm_service._fallback_keyword_parse(text, log_result=False, language=language)]

    def test_short_sauce_words(self):
        self.assertEqual(self.items("양념감자 칠리로 주세요"), [("9", 1, ["칠리 소스"])])
        self.assertEqual(self.items("양념감자 치즈맛"), [("9", 1, ["치즈 소스"])])
        self.assertEqual(self.items("양념감자 어니언으로"), [("9", 1, ["어니언 소스"])])
        self.assertEqual(self.items("양념감자, 오니언, 로아나"), [("9", 1, ["어니언 소스"])])
        self.assertEqual(self.items("양념감자 하나랑 치즈스틱 하나"), [("9", 1, []), ("20", 1, [])])

    def test_foreign_sauce_words(self):
        cases = [
            ("seasoned fries with cheese sauce", "en", "치즈 소스"),
            ("seasoned fries with chili", "en", "칠리 소스"),
            ("シーズニングポテトをチリソースで", "ja", "칠리 소스"),
            ("patatas sazonadas con salsa de queso", "es", "치즈 소스"),
            ("มันฝรั่งทอดปรุงรสซอสพริก", "th", "칠리 소스"),
        ]
        for text, language, sauce in cases:
            self.assertEqual(self.items(text, language), [("9", 1, [sauce])], text)

    def test_gemini_result_gets_sauce_from_text(self):
        items = [llm_service.OrderItem(menu_id="9", quantity=1, options=[])]
        filled = llm_service._fill_missing_sauces(items, "seasoned fries with onion please", "en")
        self.assertEqual(filled[0].options, ["어니언 소스"])
        self.assertEqual(llm_service._filter_options("9", ["치즈 추가"]), ["치즈 소스"])

