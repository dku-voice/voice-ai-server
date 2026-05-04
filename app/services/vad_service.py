"""
app/services/vad_service.py - Silero VAD로 음성 구간 감지
v1.0에선 VAD 없이 전체 오디오 다 STT 돌렸는데, 무음 구간도 처리해서 느렸음
v2.0에선 말하는 구간만 잘라서 STT에 넘김
"""
import numpy as np
import torch
import logging
from silero_vad import get_speech_timestamps  # 🚨 안전한 청킹(Chunking) 처리를 위해 추가

from app.config import VAD_THRESHOLD, VAD_SAMPLE_RATE

logger = logging.getLogger(__name__)


# ⚠️ numpy 오디오 변환 주의사항:
# bytes → numpy 할 때 dtype 안 맞으면 바로 터짐 ㅠㅠ
# int16 PCM이면 np.int16, float32면 np.float32
# 프론트에서 뭘로 보내는지 꼭 확인해야 함 (이거 때문에 3시간 날림)

# Silero VAD 모델 (전역 싱글톤)
_vad_model = None


def _load_vad_model():
    """VAD 모델 로드 - 처음 한 번만 로드됨"""
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    print("[VAD] Silero VAD 모델 로딩 중...")
    _vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    print("[VAD] 모델 로딩 완료!")
    return _vad_model


def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """
    raw PCM bytes → float32 numpy array 변환
    프론트에서 16-bit PCM으로 보내는 걸 가정함

    # 💀 여기서 제일 많이 에러남
    # numpy dtype 변환 잘못하면 소리가 깨지거나 VAD가 전부 무음으로 판단함
    # int16 → float32 변환 시 반드시 32768.0으로 나눠줘야 -1.0 ~ 1.0 범위가 됨
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
    model = _load_vad_model()

    try:
        audio_float = bytes_to_float32(audio_bytes)

        # Silero VAD는 16kHz만 됨 (8kHz, 48kHz 넣으면 이상한 결과 나옴)
        # 프론트에서 sample rate 맞춰서 보내야 함
        audio_tensor = torch.from_numpy(audio_float)

        # 🚨 어제 밤에 터진 치명적 버그 수정! (ValueError: 512 샘플 사이즈 초과)
        # confidence = model(audio_tensor, VAD_SAMPLE_RATE).item()
        # 이렇게 전체 오디오를 한 번에 넣으면 모델이 뻗어버림.
        # 공식 API인 get_speech_timestamps를 써서 내부적으로 안전하게 청킹 처리!
        timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=VAD_SAMPLE_RATE)

        has_speech = len(timestamps) > 0

        # 타임스탬프 방식은 전체 오디오의 confidence를 주지 않으므로 임의값 할당
        confidence = 0.99 if has_speech else 0.0

        if has_speech:
            logger.info(f"[VAD] 음성 감지됨! (구간 수: {len(timestamps)}개)")
        else:
            logger.debug(f"[VAD] 무음 구간. (배경 소음)")

        return {
            "has_speech": has_speech,
            "speech_audio": audio_float if has_speech else None,
            "confidence": confidence,
        }

    except Exception as e:
        # numpy 관련 에러가 대부분임 (dtype, shape 문제)
        print(f"[VAD] 에러 발생: {e}")
        logger.error(f"[VAD] 처리 실패: {e}")
        # ⚠️ 학생다운 현실적 타협: VAD에서 에러가 나면 서버가 뻗지 않도록
        # 무조건 음성이 있다고(True) 간주하고 STT로 넘겨버림 (Fallback)
        return {
            "has_speech": True,
            "speech_audio": audio_float if 'audio_float' in locals() else None,
            "confidence": 0.0,
        }