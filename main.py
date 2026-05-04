"""
main.py - Voice AI Server v2.0 진입점
FastAPI 앱 생성 + 라우터 등록 + 모델 로딩

v1.0: 단일 파일에 전부 때려넣음 → 유지보수 불가
v2.0: 서비스별 모듈 분리 + WebSocket 스트리밍
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.websockets import router as ws_router
from app.services.stt_service import load_model as load_stt_model


# --- 로깅 설정 (학생 수준으로 간단하게) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작/종료 시 실행되는 lifecycle 관리
    startup: 모델 미리 로딩 (첫 요청에서 로딩하면 느림)
    shutdown: 정리 작업
    """
    # === Startup ===
    print("=" * 50)
    print("🚀 Voice AI Server v2.0 시작!")
    print("=" * 50)

    # STT 모델 로딩 (서버 시작 시 한 번만)
    print("[Startup] STT 모델 로딩 중...")
    load_stt_model()

    # VAD 모델은 첫 호출 시 lazy loading (torch.hub.load가 좀 오래 걸림)
    print("[Startup] VAD 모델은 첫 요청 시 로딩됨 (lazy)")

    print("[Startup] 서버 준비 완료! ✅")
    print("=" * 50)

    yield  # 여기서 서버 실행됨

    # === Shutdown ===
    print("[Shutdown] 서버 종료 중...")
    logger.info("서버 정상 종료")


# --- FastAPI 앱 생성 ---
app = FastAPI(
    title="Voice AI Server",
    description="음성 인식 기반 키오스크 주문 처리 API (v2.0)",
    version="2.0.0",
    lifespan=lifespan,
)

# --- WebSocket 라우터 등록 ---
app.include_router(ws_router)


# ============================================
# [Deprecated] V1.0 레거시 API (테스트용)
# 프론트엔드 팀이 아직 WebSocket으로 마이그레이션 안 했을 때
# 임시로 사용할 수 있도록 남겨둠
# 실제 STT/LLM 처리는 안 하고 더미 데이터 반환
# ============================================
@app.post("/api/voice")
async def process_voice_legacy():
    """
    v1.0 레거시 엔드포인트
    이전에는 여기서 whisper.transcribe() 동기 호출해서 서버 뻗었음 ㅋㅋ
    지금은 WebSocket(/ws/audio)으로 전환됨
    """
    # 더미 데이터 (테스트 확인용)
    return {
        "status": "deprecated",
        "recognized_text": "[v1.0 레거시] 이 엔드포인트는 더 이상 사용되지 않습니다. /ws/audio를 사용하세요.",
        "items": [
            {"menu_id": "burger_01", "quantity": 1, "options": []},
        ],
        "error_msg": "이 API는 deprecated 되었습니다. WebSocket /ws/audio로 전환하세요.",
    }


# --- 헬스체크 ---
@app.get("/")
def health_check():
    """서버 생존 확인용 (프론트/백엔드 연결 테스트)"""
    return {
        "message": "Voice AI Server v2.0 is running",
        "version": "2.0.0",
        "endpoints": {
            "websocket": "/ws/audio",
            "legacy_http": "/api/voice (deprecated)",
        },
    }


if __name__ == "__main__":
    import uvicorn
    from app.config import SERVER_HOST, SERVER_PORT

    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,  # 개발 중엔 auto-reload 편함
    )
