"""
app/config.py - 서버 설정 모아놓은 파일
v2.0에서 환경변수로 관리하려고 만듦
"""
import os


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
