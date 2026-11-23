import json
import logging
import threading
import urllib.request

from app.config import (
    MENU_SYNC_ENABLED,
    MENU_SYNC_MAX_ATTEMPTS,
    MENU_SYNC_RETRY_SECONDS,
    MENU_SYNC_URL,
)
from app.services import llm_service

logger = logging.getLogger(__name__)

BRAND_PREFIXES = ("리아",)
FOREIGN_NAME_FIELDS = ("nameEn", "nameJa", "nameEs", "nameTh")
LANGUAGE_NAME_FIELDS = {"ko": "nameKo", "ja": "nameJa", "es": "nameEs", "th": "nameTh"}
ORDERABLE_STATUSES = {"ACTIVE", "SOLD_OUT"}

_stop_event = threading.Event()


def _clean(value) -> str:
    return value.strip() if isinstance(value, str) else ""


HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
JUNGSEONG_AE = 1
JUNGSEONG_E = 5


def _swap_vowel(text: str, source: int, target: int) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if HANGUL_BASE <= code <= HANGUL_END:
            offset = code - HANGUL_BASE
            jung = (offset // 28) % 21
            if jung == source:
                code += (target - source) * 28
        chars.append(chr(code))
    return "".join(chars)


def _korean_aliases(name_ko: str) -> tuple[str, ...]:
    bases = [name_ko]
    for prefix in BRAND_PREFIXES:
        if name_ko.startswith(prefix):
            stripped = name_ko[len(prefix):].strip()
            if len(stripped) >= 2:
                bases.append(stripped)
    aliases = []
    for base in bases:
        aliases.extend((
            base,
            _swap_vowel(base, JUNGSEONG_E, JUNGSEONG_AE),
            _swap_vowel(base, JUNGSEONG_AE, JUNGSEONG_E),
        ))
    return tuple(dict.fromkeys(aliases))


def build_menu_catalog(payload: dict) -> tuple[dict, dict, dict]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    menus = payload.get("menus") if isinstance(payload, dict) else None
    if not isinstance(menus, list):
        raise ValueError("메뉴 응답에 menus 배열이 없습니다.")

    catalog: dict[str, str] = {}
    aliases: dict[str, tuple[str, ...]] = {}
    foreign: dict[str, tuple[str, ...]] = {}

    for menu in menus:
        if not isinstance(menu, dict):
            continue
        menu_id = menu.get("menuId")
        name_ko = _clean(menu.get("nameKo"))
        status = _clean(menu.get("status")) or "ACTIVE"
        if not isinstance(menu_id, int) or isinstance(menu_id, bool) or not name_ko:
            continue
        if status not in ORDERABLE_STATUSES:
            continue

        key = str(menu_id)
        names = tuple(dict.fromkeys(n for n in (_clean(menu.get(f)) for f in FOREIGN_NAME_FIELDS) if n))
        catalog[key] = name_ko
        foreign[key] = names
        aliases[key] = tuple(dict.fromkeys(_korean_aliases(name_ko) + names))

    if not catalog:
        raise ValueError("주문 가능한 메뉴가 없습니다.")
    return catalog, aliases, foreign


def fetch_menu_payload(url: str = MENU_SYNC_URL, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_names_by_language(payload: dict, catalog: dict[str, str]) -> dict[str, tuple[str, ...]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    result: dict[str, list[str]] = {lang: [] for lang in LANGUAGE_NAME_FIELDS}
    for menu in payload.get("menus", []) if isinstance(payload, dict) else []:
        if not isinstance(menu, dict) or str(menu.get("menuId")) not in catalog:
            continue
        for lang, field in LANGUAGE_NAME_FIELDS.items():
            name = _clean(menu.get(field))
            if name:
                result[lang].append(name)
    return {lang: tuple(dict.fromkeys(names)) for lang, names in result.items() if names}


def build_options_by_id(payload: dict, catalog: dict[str, str]) -> dict[str, tuple[str, ...]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    result: dict[str, tuple[str, ...]] = {}
    for menu in payload.get("menus", []) if isinstance(payload, dict) else []:
        if not isinstance(menu, dict) or str(menu.get("menuId")) not in catalog:
            continue
        options = menu.get("options")
        if not isinstance(options, list):
            continue
        names = []
        for option in options:
            name = _clean(option.get("nameKo")) if isinstance(option, dict) else _clean(option)
            if name:
                names.append(name)
        result[str(menu["menuId"])] = tuple(dict.fromkeys(names))
    return result


def sync_menus_once(url: str = MENU_SYNC_URL) -> dict:
    payload = fetch_menu_payload(url)
    catalog, aliases, foreign = build_menu_catalog(payload)
    llm_service.apply_menu_catalog(
        catalog,
        aliases,
        foreign,
        source="backend",
        names_by_language=build_names_by_language(payload, catalog),
        options_by_id=build_options_by_id(payload, catalog),
    )
    return llm_service.menu_catalog_summary()


def _sync_with_retry() -> None:
    for attempt in range(1, MENU_SYNC_MAX_ATTEMPTS + 1):
        if _stop_event.is_set():
            return
        try:
            summary = sync_menus_once()
            logger.info("[MENU] 백엔드 메뉴 동기화 성공: %d종 (시도 %d)", summary["count"], attempt)
            return
        except Exception as e:
            logger.warning("[MENU] 백엔드 메뉴 동기화 실패 (시도 %d/%d): %s", attempt, MENU_SYNC_MAX_ATTEMPTS, e)
            _stop_event.wait(MENU_SYNC_RETRY_SECONDS)
    logger.error("[MENU] 백엔드 메뉴 동기화를 포기하고 기본 메뉴 목록을 사용합니다.")


def start_menu_sync() -> None:
    if not MENU_SYNC_ENABLED:
        logger.info("[MENU] 메뉴 동기화 비활성화, 기본 메뉴 목록 사용")
        return
    _stop_event.clear()
    threading.Thread(target=_sync_with_retry, name="menu-sync", daemon=True).start()


def stop_menu_sync() -> None:
    _stop_event.set()
