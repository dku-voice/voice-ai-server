"""
app/services/vad_service.py - Silero VAD로 음성 구간 감지
v1.0에선 VAD 없이 전체 오디오 다 STT 돌렸는데, 무음 구간도 처리해서 느렸음
v2.0에선 말하는 구간만 잘라서 STT에 넘김
"""
import numpy as np
import torch
import logging
import threading
from silero_vad import get_speech_timestamps, load_silero_vad

from app.config import VAD_THRESHOLD, VAD_SAMPLE_RATE
from app.services.threadpool import run_model_task

logger = logging.getLogger(__name__)


# bytes를 numpy로 바꿀 때 dtype이 틀리면 소리가 바로 깨진다.
# 지금은 프론트에서 raw PCM int16으로 보낸다는 전제로 처리한다.

# Silero VAD 모델도 한 번만 로드해서 재사용한다.
_vad_model = None
_vad_model_lock = threading.Lock()
SPEECH_PADDING_MS = 120


def _load_vad_model():
    """VAD 모델을 처음 한 번만 로드한다."""
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    with _vad_model_lock:
        if _vad_model is not None:
            return _vad_model

        logger.info("[VAD] Silero VAD 모델 로딩 중...")
        _vad_model = load_silero_vad()
        logger.info("[VAD] 모델 로딩 완료!")
    return _vad_model


def _collect_speech_audio(audio_float: np.ndarray, timestamps: list[dict]) -> np.ndarray:
    """VAD가 찾은 구간만 STT에 넘기도록 잘라낸다."""
    padding = int(VAD_SAMPLE_RATE * SPEECH_PADDING_MS / 1000)
    chunks = []

    for timestamp in timestamps:
        start = max(0, int(timestamp["start"]) - padding)
        end = min(len(audio_float), int(timestamp["end"]) + padding)
        if start < end:
            chunks.append(audio_float[start:end])

    if not chunks:
        return audio_float

    return np.concatenate(chunks).astype(np.float32, copy=False)


def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """
    raw PCM bytes를 float32 numpy array로 바꾼다.
    int16 값을 -1.0 ~ 1.0 범위로 맞추기 위해 32768.0으로 나눈다.
    """
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_np.astype(np.float32) / 32768.0
    return audio_float


def detect_speech(audio_bytes: bytes) -> dict:
    """
    음성 데이터에서 사람 말하는 구간 감지

    Returns:
        {
            "has_speech": bool,
            "speech_audio": np.ndarray or None (float32),
            "confidence": float
        }
    """
    try:
        model = _load_vad_model()
        audio_float = bytes_to_float32(audio_bytes)

        # Silero VAD는 16kHz만 됨 (8kHz, 48kHz 넣으면 이상한 결과 나옴)
        # 프론트에서 sample rate 맞춰서 보내야 함
        audio_tensor = torch.from_numpy(audio_float)

        # 처음에는 모델에 전체 tensor를 바로 넣었다가 512 samples 에러가 났다.
        # confidence = model(audio_tensor, VAD_SAMPLE_RATE).item()
        # 지금은 공식 API인 get_speech_timestamps를 사용한다.
        timestamps = get_speech_timestamps(
            audio_tensor, model,
            sampling_rate=VAD_SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
        )

        has_speech = len(timestamps) > 0

        # timestamp API는 전체 confidence를 따로 주지 않아서 간단한 값만 넣는다.
        confidence = 0.99 if has_speech else 0.0

        speech_audio = _collect_speech_audio(audio_float, timestamps) if has_speech else None

        if has_speech:
            logger.info(f"[VAD] 음성 감지됨! (구간 수: {len(timestamps)}개)")
        else:
            logger.debug(f"[VAD] 무음 구간. (배경 소음)")

        return {
            "has_speech": has_speech,
            "speech_audio": speech_audio,
            "confidence": confidence,
        }

    except Exception as e:
        # numpy 관련 에러가 대부분임 (dtype, shape 문제)
        logger.error("[VAD] 처리 실패: %s", e)
        # VAD가 실패해도 서버는 죽이지 않는다.
        # 일단 음성이 있다고 보고 STT로 넘겨서 다음 단계에서 처리하게 둔다.
        fallback_audio = None
        if 'audio_float' in locals():
            fallback_audio = audio_float
        else:
            try:
                fallback_audio = bytes_to_float32(audio_bytes)
            except Exception:
                pass  # 이것마저 실패하면 None → websockets.py에서 처리
        return {
            "has_speech": True,
            "speech_audio": fallback_audio,
            "confidence": 0.0,
        }


async def detect_speech_async(audio_bytes: bytes) -> dict:
    """
    WebSocket 핸들러에서 쓰는 비동기 wrapper.
    VAD도 모델 추론이라 threadpool에서 실행한다.
    """
    return await run_model_task(detect_speech, audio_bytes)
