"""
app/services/stt_service.py - faster-whisper 기반 STT 서비스
v1.0의 openai-whisper에서 faster-whisper(CTranslate2)로 교체

# 🚨 주의: v1.0에서 메인 스레드 블로킹으로 서버 뻗어서 비동기 처리 도입함
# model.transcribe()가 CPU에서 5~10초 걸리는데
# FastAPI의 async 핸들러에서 동기로 호출하면 이벤트 루프가 멈춤
# → 다른 클라이언트 요청 전부 대기 → 서버 먹통
# → run_in_threadpool로 별도 스레드에서 실행하도록 수정
"""
import numpy as np
import logging
import io
import threading
import soundfile as sf
from faster_whisper import WhisperModel
from starlette.concurrency import run_in_threadpool

from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    STT_LANGUAGE,
)

logger = logging.getLogger(__name__)


# ============================================
# 전역 싱글톤 모델
# 서버 시작할 때 한 번만 로드 (매 요청마다 로드하면 메모리 터짐)
# ============================================
_whisper_model: WhisperModel | None = None
_whisper_model_error: str | None = None
_whisper_model_lock = threading.Lock()


def load_model():
    """
    서버 startup 시 호출 - 모델을 전역으로 올림
    main.py (혹은 app factory)에서 lifespan으로 호출해야 함
    """
    global _whisper_model, _whisper_model_error
    if _whisper_model is not None:
        print("[STT] 모델 이미 로딩되어 있음, skip")
        return

    with _whisper_model_lock:
        if _whisper_model is not None:
            print("[STT] 모델 이미 로딩되어 있음, skip")
            return

        print(f"[STT] faster-whisper 모델 로딩: size={WHISPER_MODEL_SIZE}, "
              f"device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE_TYPE}")

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
        print("[STT] 모델 로딩 완료!")


def _get_model() -> WhisperModel:
    """모델 가져오기 (없으면 에러)"""
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
    """헬스체크에서 사용할 STT 모델 상태를 반환한다."""
    return {
        "ready": _whisper_model is not None,
        "error": _whisper_model_error,
    }


def _transcribe_sync(audio_data: np.ndarray) -> str:
    """
    동기 transcribe (이걸 직접 async 핸들러에서 부르면 서버 뻗음!!)

    audio_data: float32 numpy array (VAD에서 넘어옴)
    returns: 인식된 텍스트
    """
    model = _get_model()

    # faster-whisper는 파일 경로 or ndarray 받음
    # v1.0에선 디스크에 임시파일 저장했는데, v2.0에선 메모리에서 바로 처리
    # ndarray를 직접 넘기면 됨 (이게 훨씬 빠름)
    segments, info = model.transcribe(
        audio_data,
        language=STT_LANGUAGE,
        beam_size=5,  # 정확도 올리려면 beam_size 키우면 되는데 느려짐
    )

    # segments는 generator라서 list로 모아야 함
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())

    result_text = " ".join(text_parts).strip()

    logger.info(f"[STT] 인식 결과: '{result_text}' (lang={info.language}, prob={info.language_probability:.2f})")
    return result_text


async def transcribe(audio_data: np.ndarray) -> str:
    """
    ✅ 비동기 transcribe - FastAPI 핸들러에서는 이걸 사용!!

    run_in_threadpool: Starlette 내장 함수
    내부적으로 anyio thread pool 사용해서 동기 함수를 별도 스레드에서 실행
    → 메인 이벤트 루프 블로킹 안 됨 → 서버 안 뻗음

    # v1.0 실수: model.transcribe()를 async 함수 안에서 그냥 호출
    # → uvicorn 워커 1개일 때 동시 요청 들어오면 뒤에 요청 timeout
    # → 교수님한테 "왜 서버 느려요?" 라고 혼남 ㅋㅋ
    """
    print(f"[STT] 비동기 transcribe 시작 (audio shape: {audio_data.shape})")

    result = await run_in_threadpool(_transcribe_sync, audio_data)

    print(f"[STT] 결과: {result}")
    return result


async def transcribe_bytes(audio_bytes: bytes) -> str:
    """
    raw bytes에서 바로 STT 돌리는 헬퍼
    VAD 안 거치고 바로 STT 하고 싶을 때 사용
    (HTTP endpoint 테스트용으로 유용함)
    """
    try:
        # bytes → numpy float32 변환
        # soundfile로 읽으면 알아서 float32로 변환해줌 (편함)
        audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # 모노로 변환 (스테레오면 첫 번째 채널만 사용)
        if len(audio_np.shape) > 1:
            audio_np = audio_np[:, 0]

        return await transcribe(audio_np)

    except Exception as e:
        print(f"[STT] bytes 변환 에러: {e}")
        logger.error(f"[STT] transcribe_bytes 실패: {e}")
        raise
