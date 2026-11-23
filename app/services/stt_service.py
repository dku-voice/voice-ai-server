"""
app/services/stt_service.py - faster-whisper 기반 STT 서비스
v1.0의 openai-whisper에서 faster-whisper(CTranslate2)로 교체

처음에는 transcribe를 async 핸들러 안에서 바로 불렀는데,
CPU에서 몇 초씩 걸리면서 다른 요청도 같이 멈췄다.
그래서 실제 STT 호출은 threadpool에서 돌린다.
"""
import numpy as np
import logging
import io
import threading
from types import SimpleNamespace
import soundfile as sf
from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    STT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    LANGUAGE_MIN_PROBABILITY,
    STT_HOTWORD_LANGUAGES,
)
from app.services.threadpool import run_model_task

logger = logging.getLogger(__name__)


# 모델은 전역으로 하나만 둔다.
# 요청마다 새로 로드하면 너무 느리고 메모리도 많이 쓴다.
_whisper_model: WhisperModel | None = None
_whisper_model_error: str | None = None
_whisper_model_lock = threading.Lock()
_whisper_transcribe_lock = threading.Lock()


def load_model():
    """
    서버 시작할 때 STT 모델을 미리 올린다.
    실패해도 서버 자체는 켜지고, 실제 STT 요청에서 에러를 돌려준다.
    """
    global _whisper_model, _whisper_model_error
    if _whisper_model is not None:
        logger.info("[STT] 모델 이미 로딩되어 있음, skip")
        return

    with _whisper_model_lock:
        if _whisper_model is not None:
            logger.info("[STT] 모델 이미 로딩되어 있음, skip")
            return

        logger.info(
            "[STT] faster-whisper 모델 로딩: size=%s, device=%s, compute=%s",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )

        try:
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        except Exception as e:
            _whisper_model_error = str(e)
            logger.error("[STT] 모델 로딩 실패: %s", e, exc_info=True)
            raise

        _whisper_model_error = None
        logger.info("[STT] 모델 로딩 완료!")


def _get_model() -> WhisperModel:
    """로드된 STT 모델을 가져온다."""
    if _whisper_model is None:
        try:
            load_model()
        except Exception as e:
            detail = _whisper_model_error or str(e)
            raise RuntimeError(f"[STT] 모델 로딩 실패: {detail}") from e
    if _whisper_model is None:
        raise RuntimeError("[STT] 모델 로딩 상태가 올바르지 않습니다.")
    return _whisper_model


def get_model_status() -> dict:
    """헬스체크에서 보여줄 STT 모델 상태."""
    return {
        "ready": _whisper_model is not None,
        "error": _whisper_model_error,
    }


def _pick_language(info) -> str:
    detected = info.language
    probability = info.language_probability or 0.0

    if detected not in SUPPORTED_LANGUAGES:
        candidates = [
            (lang, prob)
            for lang, prob in (getattr(info, "all_language_probs", None) or [])
            if lang in SUPPORTED_LANGUAGES
        ]
        if not candidates:
            return DEFAULT_LANGUAGE
        detected, probability = max(candidates, key=lambda c: c[1])

    if detected != DEFAULT_LANGUAGE and probability < LANGUAGE_MIN_PROBABILITY:
        return DEFAULT_LANGUAGE
    return detected


HOTWORD_BRAND_PREFIXES = ("리아",)


def _strip_brand(name: str) -> str:
    for prefix in HOTWORD_BRAND_PREFIXES:
        if name.startswith(prefix):
            stripped = name[len(prefix):].strip()
            if len(stripped) >= 2:
                return stripped
    return name


def _hotwords_for(language: str | None) -> str | None:
    if not language or language not in STT_HOTWORD_LANGUAGES:
        return None
    from app.services.llm_service import menu_names_for_language

    names = menu_names_for_language(language)
    if not names:
        return None
    if language == "ko":
        names = tuple(dict.fromkeys(_strip_brand(name) for name in names))
    return ", ".join(names)[:400]


def _transcribe_sync(audio_data: np.ndarray) -> tuple[str, str]:
    """
    실제 faster-whisper 호출.
    이 함수는 무거우므로 async 핸들러에서 바로 부르면 안 된다.

    audio_data: float32 numpy array (VAD에서 넘어옴)
    returns: (인식된 텍스트, 언어 코드)
    """
    model = _get_model()

    with _whisper_transcribe_lock:
        segments, info = model.transcribe(
            audio_data,
            language=STT_LANGUAGE,
            beam_size=5,
            hotwords=_hotwords_for(STT_LANGUAGE) if STT_LANGUAGE else None,
        )

        language = STT_LANGUAGE or _pick_language(info)
        hotwords = _hotwords_for(language)
        if not STT_LANGUAGE and (language != info.language or hotwords):
            if language != info.language:
                logger.info(
                    "[STT] 감지 언어 %s(prob=%.2f) 대신 %s로 다시 인식",
                    info.language,
                    info.language_probability,
                    language,
                )
            detected_probability = info.language_probability
            segments, info = model.transcribe(
                audio_data,
                language=language,
                beam_size=5,
                hotwords=hotwords,
            )
            if language == info.language:
                info = SimpleNamespace(language=language, language_probability=detected_probability)

        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())

    result_text = " ".join(text_parts).strip()

    logger.info(
        "[STT] 인식 결과: '%s' (lang=%s, prob=%.2f)",
        result_text,
        language,
        info.language_probability,
    )
    return result_text, language


async def transcribe_with_language(audio_data: np.ndarray) -> tuple[str, str]:
    """
    FastAPI 쪽에서 쓰는 비동기 wrapper.
    내부 STT는 동기 함수라 제한된 threadpool로 감싼다.
    """
    logger.debug("[STT] 비동기 transcribe 시작 (audio shape: %s)", audio_data.shape)

    text, language = await run_model_task(_transcribe_sync, audio_data)

    logger.info("[STT] 결과: %s (lang=%s)", text, language)
    return text, language


async def transcribe(audio_data: np.ndarray) -> str:
    text, _ = await transcribe_with_language(audio_data)
    return text


async def transcribe_bytes(audio_bytes: bytes) -> str:
    """
    raw bytes를 바로 STT에 넣는 테스트용 함수.
    WebSocket 실제 흐름에서는 보통 VAD를 먼저 거친다.
    """
    try:
        # soundfile이 float32로 변환해준다.
        audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # 스테레오면 일단 첫 번째 채널만 쓴다.
        if len(audio_np.shape) > 1:
            audio_np = audio_np[:, 0]

        return await transcribe(audio_np)

    except Exception as e:
        logger.warning("[STT] bytes 변환 에러: %s", e)
        logger.error("[STT] transcribe_bytes 실패: %s", e)
        raise
