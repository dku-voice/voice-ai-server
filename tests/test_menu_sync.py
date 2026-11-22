import unittest
from unittest.mock import patch

from app.services import llm_service, menu_sync

BACKEND_PAYLOAD = {
    "menus": [
        {"menuId": 1, "nameKo": "데리버거", "nameEn": "Teri Burger", "status": "ACTIVE"},
        {"menuId": 2, "nameKo": "리아 새우버거", "nameEn": "Ria Shrimp Burger", "status": "ACTIVE"},
        {"menuId": 3, "nameKo": "리아 불고기버거", "nameEn": "Ria Bulgogi Burger", "nameJa": "リア プルコギバーガー",
         "nameEs": "Hamburguesa Bulgogi Ria", "nameTh": "ริอา บูลโกกิเบอร์เกอร์", "status": "ACTIVE"},
        {"menuId": 4, "nameKo": "클래식 치즈버거", "nameEn": "Classic Cheeseburger", "nameJa": "クラシックチーズバーガー",
         "nameEs": "Hamburguesa clásica con queso", "nameTh": "คลาสสิกชีสเบอร์เกอร์", "status": "ACTIVE"},
        {"menuId": 7, "nameKo": "더블 클래식 치즈버거", "nameEn": "Double Classic Cheeseburger", "status": "ACTIVE"},
        {"menuId": 14, "nameKo": "콜라", "nameEn": "Cola", "nameJa": "コーラ", "nameEs": "Cola", "nameTh": "โคล่า",
         "status": "ACTIVE"},
        {"menuId": 15, "nameKo": "제로슈거콜라", "nameEn": "Zero Sugar Cola", "nameJa": "ゼロシュガーコーラ",
         "status": "SOLD_OUT"},
        {"menuId": 99, "nameKo": "단종 메뉴", "status": "INACTIVE"},
        {"menuId": "bad", "nameKo": "잘못된 메뉴"},
    ]
}


class MenuSyncTest(unittest.TestCase):
    def setUp(self):
        self._saved = (
            llm_service.MENU_CATALOG,
            llm_service.MENU_ALIASES,
            llm_service.MENU_FOREIGN_NAMES,
            llm_service.MENU_SOURCE,
        )

    def tearDown(self):
        (
            llm_service.MENU_CATALOG,
            llm_service.MENU_ALIASES,
            llm_service.MENU_FOREIGN_NAMES,
            llm_service.MENU_SOURCE,
        ) = self._saved

    def _sync(self):
        with patch.object(menu_sync, "fetch_menu_payload", return_value=BACKEND_PAYLOAD):
            return menu_sync.sync_menus_once()

    def test_build_catalog_uses_numeric_ids_and_skips_invalid_menus(self):
        catalog, aliases, foreign = menu_sync.build_menu_catalog(BACKEND_PAYLOAD)
        self.assertEqual(set(catalog), {"1", "2", "3", "4", "7", "14", "15"})
        self.assertIn("불고기버거", aliases["3"])
        self.assertIn("コーラ", foreign["14"])

    def test_sync_replaces_catalog_and_keyword_matching(self):
        summary = self._sync()
        self.assertEqual(summary["source"], "backend")
        self.assertEqual(summary["count"], 7)

        def parsed(text):
            return {item.menu_id: item.quantity for item in llm_service._fallback_keyword_parse(text, log_result=False)}

        self.assertEqual(parsed("콜라 두 잔 주세요"), {"14": 2})
        self.assertEqual(parsed("제로슈거콜라 하나"), {"15": 1})
        self.assertEqual(parsed("불고기버거 한 개"), {"3": 1})
        self.assertEqual(parsed("클래식 치즈버거 하나랑 콜라 하나"), {"4": 1, "14": 1})
        self.assertEqual(parsed("더블 클래식 치즈버거 주세요"), {"7": 1})
        self.assertEqual(parsed("コーラください"), {"14": 1})
        self.assertEqual(parsed("콜라 2개랑 대리버거 하나 주세요"), {"14": 2, "1": 1})
        self.assertEqual(parsed("세우버거 하나"), {"2": 1})

    def test_prompt_lists_backend_menus_with_foreign_names(self):
        self._sync()
        prompt = llm_service._user_prompt("コーラをふたつ", "ja")
        self.assertIn("- 14: 콜라", prompt)
        self.assertIn("コーラ", prompt)
        self.assertNotIn("burger_01", prompt)

    def test_llm_result_with_numeric_menu_id_is_accepted(self):
        self._sync()
        items = llm_service._parse_llm_response('[{"menu_id": 14, "quantity": 2, "options": []}]', "コーラをふたつ", "ja")
        self.assertEqual(items[0].menu_id, "14")
        with self.assertRaises(llm_service.LLMResponseValidationError):
            llm_service._parse_llm_response('[{"menu_id": "cola_01", "quantity": 1}]', "コーラ", "ja")

    def test_failed_fetch_keeps_static_catalog(self):
        before = llm_service.MENU_CATALOG
        with patch.object(menu_sync, "fetch_menu_payload", side_effect=OSError("down")):
            with self.assertRaises(OSError):
                menu_sync.sync_menus_once()
        self.assertIs(llm_service.MENU_CATALOG, before)

    def test_backend_api_response_wrapper_is_unwrapped(self):
        wrapped = {"success": True, "data": BACKEND_PAYLOAD, "message": None}
        catalog, _, _ = menu_sync.build_menu_catalog(wrapped)
        self.assertEqual(set(catalog), {"1", "2", "3", "4", "7", "14", "15"})

    def test_empty_menu_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            menu_sync.build_menu_catalog({"menus": []})


if __name__ == "__main__":
    unittest.main()
