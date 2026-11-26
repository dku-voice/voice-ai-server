import asyncio
import unittest
from unittest.mock import patch

from app.services import llm_service, menu_sync

PAYLOAD = {
    "success": True,
    "data": {
        "menus": [
            {
                "menuId": 1, "nameKo": "데리버거", "nameEn": "Teri Burger",
                "nameJa": "テリバーガー", "nameEs": "Hamburguesa Teri", "nameTh": "เทริเบอร์เกอร์",
                "status": "ACTIVE",
                "options": [
                    {"nameKo": "치즈 추가", "nameEn": "Add Cheese", "nameJa": "チーズ追加", "nameEs": "Extra de queso", "nameTh": "เพิ่มชีส"},
                    {"nameKo": "양파 제거", "nameEn": "No Onion", "nameJa": "玉ねぎ抜き", "nameEs": "Sin cebolla", "nameTh": "ไม่ใส่หอมหัวใหญ่"},
                ],
            },
            {
                "menuId": 2, "nameKo": "리아 새우버거", "nameEn": "Ria Shrimp Burger",
                "nameJa": "リア エビバーガー", "nameEs": "Hamburguesa de camarón Ria", "nameTh": "ริอา เบอร์เกอร์กุ้ง",
                "status": "ACTIVE", "options": [],
            },
            {
                "menuId": 11, "nameKo": "치즈스틱", "nameEn": "Cheese Sticks",
                "nameJa": "チーズスティック", "nameEs": "Palitos de queso", "nameTh": "ชีสสติ๊ก",
                "status": "ACTIVE", "options": [],
            },
            {
                "menuId": 14, "nameKo": "콜라", "nameEn": "Cola",
                "nameJa": "コーラ", "nameEs": "Cola", "nameTh": "โคล่า",
                "status": "ACTIVE", "options": [],
            },
            {
                "menuId": 15, "nameKo": "제로슈거콜라", "nameEn": "Zero Sugar Cola",
                "nameJa": "ゼロシュガーコーラ", "nameEs": "Cola sin azúcar", "nameTh": "โคล่าซีโร่ชูการ์",
                "status": "ACTIVE", "options": [],
            },
        ]
    },
}


class ForeignFallbackTest(unittest.TestCase):
    def setUp(self):
        self._saved = (
            llm_service.MENU_CATALOG,
            llm_service.MENU_ALIASES,
            llm_service.MENU_FOREIGN_NAMES,
            llm_service.MENU_NAMES_BY_LANGUAGE,
            llm_service.MENU_OPTIONS_BY_ID,
            llm_service.MENU_META_BY_ID,
            llm_service.OPTION_NAMES_BY_LANGUAGE,
            llm_service.MENU_SOURCE,
        )
        catalog, aliases, foreign = menu_sync.build_menu_catalog(PAYLOAD)
        llm_service.apply_menu_catalog(
            catalog, aliases, foreign, "backend",
            names_by_language=menu_sync.build_names_by_language(PAYLOAD, catalog),
            options_by_id=menu_sync.build_options_by_id(PAYLOAD, catalog),
            option_names_by_language=menu_sync.build_option_names_by_language(PAYLOAD, catalog),
        )

    def tearDown(self):
        (
            llm_service.MENU_CATALOG,
            llm_service.MENU_ALIASES,
            llm_service.MENU_FOREIGN_NAMES,
            llm_service.MENU_NAMES_BY_LANGUAGE,
            llm_service.MENU_OPTIONS_BY_ID,
            llm_service.MENU_META_BY_ID,
            llm_service.OPTION_NAMES_BY_LANGUAGE,
            llm_service.MENU_SOURCE,
        ) = self._saved

    def parse(self, text, language):
        items = llm_service._fallback_keyword_parse(text, log_result=False, language=language)
        return [(i.menu_id, i.quantity, i.options) for i in items]

    def test_japanese_quantity_words(self):
        self.assertEqual(self.parse("コーラをふたつください", "ja"), [("14", 2, [])])
        self.assertEqual(self.parse("チーズスティック三つ", "ja"), [("11", 3, [])])
        self.assertEqual(self.parse("コーラを2つ", "ja"), [("14", 2, [])])

    def test_japanese_quantity_stays_with_its_menu(self):
        self.assertEqual(self.parse("テリバーガーひとつとコーラ", "ja"), [("1", 1, []), ("14", 1, [])])

    def test_japanese_option_and_brandless_name(self):
        self.assertEqual(self.parse("テリバーガー玉ねぎ抜きで二つ", "ja"), [("1", 2, ["양파 제거"])])
        self.assertEqual(self.parse("エビバーガーください", "ja"), [("2", 1, [])])

    def test_spanish_quantity_before_menu(self):
        self.assertEqual(self.parse("Una hamburguesa Teri y dos colas", "es"), [("1", 1, []), ("14", 2, [])])

    def test_spanish_plural_and_accentless(self):
        self.assertEqual(self.parse("dos hamburguesas teri", "es"), [("1", 2, [])])
        self.assertEqual(self.parse("una cola sin azucar", "es"), [("15", 1, [])])
        self.assertEqual(self.parse("hamburguesa de camaron", "es"), [("2", 1, [])])

    def test_spanish_option(self):
        self.assertEqual(self.parse("hamburguesa teri sin cebolla", "es"), [("1", 1, ["양파 제거"])])

    def test_thai_quantity_and_option(self):
        self.assertEqual(self.parse("ขอโคล่าสองแก้ว", "th"), [("14", 2, [])])
        self.assertEqual(self.parse("เทริเบอร์เกอร์ไม่ใส่หอมหัวใหญ่", "th"), [("1", 1, ["양파 제거"])])

    def test_option_not_on_menu_is_dropped(self):
        self.assertEqual(self.parse("コーラ、チーズ追加", "ja"), [("14", 1, [])])

    def test_japanese_hiragana_and_misheard_names(self):
        self.assertEqual(self.parse("てりぼうが、おひとつ、たまねき、ぬきで。", "ja"), [("1", 1, ["양파 제거"])])
        self.assertEqual(self.parse("こーらをふたつください", "ja"), [("14", 2, [])])
        self.assertEqual(self.parse("ぜろしゅがーこーら", "ja"), [("15", 1, [])])

    def test_japanese_misheard_long_order_keeps_every_menu(self):
        self.assertEqual(
            self.parse("テリバーガーをタッツタマネギに抜きて それぞれチーズスティックを 土地と コラウミッツ ください", "ja"),
            [("1", 1, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )
        self.assertEqual(
            self.parse("テリバがオフタチタマネギを抜きで、それとチーズスティックを一つどこらを見つくださり。", "ja"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )
        self.assertEqual(
            self.parse("テリバーガーをふたつ、玉ねぎ抜きで、それとチーズスティックをひとつとコーラをみっつください", "ja"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )

    def test_japanese_common_mishearings(self):
        self.assertEqual(
            self.parse("テリバーガーを2つたまねぎ切れ、それとチーズスティックをひとつとコーラを見つくだしたい。", "ja"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )
        self.assertEqual(
            self.parse("てりばがおふたちたまねぎのきて それとチーズスティックをひとすと コーラをみつづくださり", "ja"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )
        self.assertEqual(
            self.parse("テリバーガーを立ちたまねぎ抜きて それとチーズスティックを ひとつとコーラを見つくださえ", "ja"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )

    def test_japanese_mitsu_without_small_tsu(self):
        self.assertEqual(
            self.parse("テリバーガーを2つ、玉ねぎ抜きで、それとチーズスティックを一つとコーラをみつくださいます。", "ja"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )

    def test_spanish_everyday_menu_words(self):
        self.assertEqual(
            self.parse("¡Dosteri burjerse en cebolla! ¡Unches estiquití tres colas, por favor!", "es"),
            [("1", 2, ["양파 제거"]), ("14", 3, [])],
        )
        self.assertEqual(
            self.parse("¡Dos teri burgers sin cebolla! Unos cheese sticks y tres colas, por favor", "es"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )
        self.assertEqual(self.parse("dos coca colas y unos palitos", "es"), [("14", 2, []), ("11", 1, [])])

    def test_english_order_with_quantities_and_options(self):
        self.assertEqual(
            self.parse("Can I get two teri burgers with no onions, a cheese stick and three cokes please", "en"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )

    def test_english_menu_name_alone_is_one(self):
        self.assertEqual(self.parse("Cola please", "en"), [("14", 1, [])])

    def test_english_option_name_from_menu_data(self):
        self.assertEqual(self.parse("one Teri Burger, Add Cheese", "en"), [("1", 1, ["치즈 추가"])])

    def test_japanese_onion_misheard_as_uki(self):
        self.assertEqual(
            self.parse("テリバーガーを2つ玉ねぎに浮きて それとチーズスティックコーヒーとつとコーラを見つくださり", "ja"),
            [("1", 2, ["양파 제거"]), ("11", 1, []), ("14", 3, [])],
        )

    def test_thai_cola_mishearings(self):
        self.assertEqual(self.parse("ขอโคลาสอนเกี่ยว", "th"), [("14", 2, [])])
        self.assertEqual(self.parse("ขอคุณล่าสองแกลง", "th"), [("14", 2, [])])

    def test_empty_llm_answer_uses_keywords_for_foreign(self):
        phase, items = llm_service._prefer_keywords_when_llm_empty([], "ขอโคล่าสองแก้ว", "th")
        self.assertEqual(phase, "fallback")
        self.assertEqual([(i.menu_id, i.quantity) for i in items], [("14", 2)])
        phase, items = llm_service._prefer_keywords_when_llm_empty([], "สวัสดีครับ", "th")
        self.assertEqual((phase, items), ("success", []))

    def _check(self, text, language, menu_ids):
        from app.schemas import OrderItem

        items = [OrderItem(menu_id=menu_id, quantity=1, options=[]) for menu_id in menu_ids]
        return [item.menu_id for item in llm_service._validate_foreign_items(items, text, language)]

    def test_foreign_llm_menu_without_sound_evidence_is_dropped(self):
        self.assertEqual(self._check("อร์คุณลาสอง เกอร์, ขภอร์เกอร์", "th", ["1"]), [])
        self.assertEqual(self._check("Hi!", "en", ["14"]), [])

    def test_foreign_llm_menu_with_sound_evidence_is_kept(self):
        self.assertEqual(self._check("ขอโคล่าสองแก้ว", "th", ["14"]), ["14"])
        self.assertEqual(self._check("デリーバラガーを2個", "ja", ["1"]), ["1"])
        self.assertEqual(self._check("Can I get two Teri burgers, so it's onions?", "en", ["1"]), ["1"])

    def test_english_to_before_menu_means_two(self):
        self.assertEqual(self.parse("Can I get to Teri burgers?", "en"), [("1", 2, [])])
        self.assertEqual(self.parse("I want to order a cola", "en"), [("14", 1, [])])
        self.assertEqual(self.parse("I want to order cola", "en"), [("14", 1, [])])

    def test_japanese_small_talk_adds_nothing(self):
        self.assertEqual(self.parse("ありがとう", "ja"), [])
        self.assertEqual(self.parse("こんにちは", "ja"), [])

    def test_korean_path_unchanged(self):
        self.assertEqual(self.parse("데리버거 두 개 치즈 추가", "ko"), [("1", 2, ["치즈 추가"])])

    def test_extract_order_uses_foreign_fallback_when_llm_fails(self):
        with patch("app.services.llm_service._get_client", side_effect=RuntimeError("503 high demand")):
            result = asyncio.run(llm_service.extract_order("コーラをふたつください", language="ja"))
        self.assertEqual(result["status"], "fallback")
        self.assertEqual([(i.menu_id, i.quantity) for i in result["items"]], [("14", 2)])


if __name__ == "__main__":
    unittest.main()
