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
import soundfile as sf
from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    STT_LANGUAGE,
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
            logger.error(f"[STT] 모델 로딩 실패: {e}", exc_info=True)
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


def _transcribe_sync(audio_data: np.ndarray) -> str:
    """
    실제 faster-whisper 호출.
    이 함수는 무거우므로 async 핸들러에서 바로 부르면 안 된다.

    audio_data: float32 numpy array (VAD에서 넘어옴)
    returns: 인식된 텍스트
    """
    model = _get_model()

    with _whisper_transcribe_lock:
        # v1.0처럼 임시 파일을 만들지 않고 ndarray를 바로 넘긴다.
        segments, info = model.transcribe(
            audio_data,
            language=STT_LANGUAGE,
            beam_size=5,  # 올리면 조금 더 정확할 수 있지만 느려진다.
        )

        # segments는 generator라 실제 추론이 순회 시점에 일어날 수 있다.
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())

    result_text = " ".join(text_parts).strip()

    logger.info(f"[STT] 인식 결과: '{result_text}' (lang={info.language}, prob={info.language_probability:.2f})")
    return result_text


async def transcribe(audio_data: np.ndarray) -> str:
    """
    FastAPI 쪽에서 쓰는 비동기 wrapper.
    내부 STT는 동기 함수라 제한된 threadpool로 감싼다.
    """
    logger.debug("[STT] 비동기 transcribe 시작 (audio shape: %s)", audio_data.shape)

    result = await run_model_task(_transcribe_sync, audio_data)

    logger.info("[STT] 결과: %s", result)
    return result


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
        logger.error(f"[STT] transcribe_bytes 실패: {e}")
        raise
