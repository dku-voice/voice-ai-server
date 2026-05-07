"""
app/config.py - 서버 설정 모아놓은 파일
v2.0에서 환경변수로 관리하려고 만듦
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def configure_console_encoding() -> None:
    """Windows 콘솔 인코딩 차이로 한글 로그 출력이 실패하지 않도록 보정한다."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # 일부 테스트 러너나 리다이렉션 환경에서는 reconfigure가 제한될 수 있다.
            pass


configure_console_encoding()


def _get_bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# --- STT 설정 ---
# faster-whisper 모델 사이즈 (base가 속도/정확도 밸런스 젤 나음)
# large-v3 쓰면 정확한데 CPU에서 너무 느려서 base로 타협함...
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# device: cpu / cuda (학교 서버에 GPU 없어서 cpu 기본)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

# compute type: int8이 cpu에서 제일 빠름
# float16은 GPU 전용이라 cpu에서 쓰면 에러남 ㅋㅋ
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# STT 언어 (한국어 고정)
STT_LANGUAGE = "ko"


# --- VAD 설정 ---
# Silero VAD threshold (0.5가 기본인데, 키오스크 환경 시끄러워서 좀 올림)
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.6"))

# 샘플레이트 (Silero VAD는 16kHz만 지원함 - 이거 때문에 삽질 많이함)
VAD_SAMPLE_RATE = 16000


# --- Server 설정 ---
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# 로그 레벨
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# WebSocket에서 다음 메시지를 기다리는 최대 시간
WS_RECEIVE_TIMEOUT_SECONDS = float(os.getenv("WS_RECEIVE_TIMEOUT_SECONDS", "30"))


# --- LLM 환경 설정 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# --- Vision / DeepFace 설정 ---
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
VISION_MAX_IMAGE_PIXELS = int(os.getenv("VISION_MAX_IMAGE_PIXELS", str(6_000_000)))
SENIOR_AGE_THRESHOLD = int(os.getenv("SENIOR_AGE_THRESHOLD", "60"))
VISION_WARMUP_ON_STARTUP = _get_bool_env("VISION_WARMUP_ON_STARTUP", "false")
