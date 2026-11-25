"""
main.py - Voice AI Server v2.0 진입점
FastAPI 앱 생성 + 라우터 등록 + 모델 로딩

처음에는 main.py에 거의 다 넣었는데 점점 보기 힘들어져서
지금은 API 라우터와 서비스 로직을 나눠둔 상태다.
"""
import logging
from contextlib import asynccontextmanager

import numpy as np

from fastapi import FastAPI

from app.api.recommendations import router as recommendations_router
from app.api.menus import router as menus_router
from app.services.menu_sync import start_menu_sync, stop_menu_sync
from app.api.vision import router as vision_router
from app.api.websockets import router as ws_router
from app.config import VISION_WARMUP_ON_STARTUP
from app.services.vad_service import _load_vad_model, detect_speech
from app.services.stt_service import get_model_status as get_stt_model_status
from app.services.stt_service import load_model as load_stt_model
from app.services.stt_service import _transcribe_sync
from app.services.threadpool import run_model_task
from app.services.vision_service import warm_up_deepface


# 로깅은 일단 기본 설정만 사용한다.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _warm_up_voice_models():
    noise = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.01).astype(np.float32)
    detect_speech((noise * 32767).astype(np.int16).tobytes())
    _transcribe_sync(noise)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작/종료 때 실행되는 부분.
    STT는 첫 요청에서 로딩하면 너무 느려서 시작할 때 먼저 올려본다.
    """
    # === Startup ===
    logger.info("Voice AI Server v2.0 시작")

    # STT 모델은 서버 시작 시 한 번만 로딩한다.
    logger.info("[Startup] STT 모델 로딩 중...")
    try:
        await run_model_task(load_stt_model)
        app.state.stt_ready = True
        app.state.stt_error = None
    except Exception as e:
        app.state.stt_ready = False
        app.state.stt_error = str(e)
        logger.error("[Startup] STT 모델 로딩 실패. 서버는 degraded 상태로 계속 실행됩니다: %s", e, exc_info=True)

    logger.info("[Startup] VAD 모델 로딩 중...")
    try:
        await run_model_task(_load_vad_model)
        logger.info("[Startup] VAD 모델 로딩 완료")
    except Exception as e:
        logger.warning("[Startup] VAD 모델 미리 로딩 실패, 첫 요청 때 다시 시도합니다: %s", e)

    logger.info("[Startup] 음성 모델 워밍업 중...")
    try:
        await run_model_task(_warm_up_voice_models)
        logger.info("[Startup] 음성 모델 워밍업 완료")
    except Exception as e:
        logger.warning("[Startup] 음성 모델 워밍업 실패, 첫 요청이 느릴 수 있습니다: %s", e)

    if VISION_WARMUP_ON_STARTUP:
        logger.info("[Startup] DeepFace 연령 추정 모델 워밍업 중...")
        try:
            await run_model_task(warm_up_deepface)
            app.state.vision_ready = True
            app.state.vision_error = None
        except Exception as e:
            app.state.vision_ready = False
            app.state.vision_error = str(e)
            logger.warning("[Startup] DeepFace 모델 워밍업 실패. 연령 추정 요청이 오면 에러를 돌려줍니다: %s", e)
    else:
        app.state.vision_ready = False
        app.state.vision_error = None
        logger.info("[Startup] DeepFace 모델은 첫 vision 요청 시 로딩됨 (lazy)")

    start_menu_sync()
    logger.info("[Startup] 서버 준비 완료")

    yield

    # === Shutdown ===
    logger.info("[Shutdown] 서버 종료 중...")
    stop_menu_sync()
    logger.info("서버 정상 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="Voice AI Server",
    description="음성 인식 기반 키오스크 주문 처리 API (v2.0)",
    version="2.0.0",
    lifespan=lifespan,
)

# 라우터 등록
app.include_router(ws_router)
app.include_router(vision_router)
app.include_router(recommendations_router)
app.include_router(menus_router)


# 예전 HTTP API.
# 지금 실제 음성 주문은 WebSocket을 쓰지만, 기존 연결 확인용으로만 남겨둔다.
@app.post("/api/voice")
async def process_voice_legacy():
    """
    v1.0에서 쓰던 HTTP 엔드포인트.
    실제 STT/LLM 처리는 이제 /ws/audio에서 한다.
    """
    # 연결 확인용 더미 응답
    return {
        "status": "deprecated",
        "recognized_text": "[v1.0 레거시] 이 엔드포인트는 더 이상 사용되지 않습니다. /ws/audio를 사용하세요.",
        "items": [
            {"menu_id": "burger_01", "quantity": 1, "options": []},
        ],
        "error_msg": "이 API는 deprecated 되었습니다. WebSocket /ws/audio로 전환하세요.",
    }


# 헬스체크
@app.get("/")
async def health_check():
    """서버가 켜져 있는지 확인할 때 사용."""
    stt_status = get_stt_model_status()
    return {
        "message": "Voice AI Server v2.0 is running",
        "version": "2.0.0",
        "endpoints": {
            "websocket": "/ws/audio",
            "vision_age": "/api/vision/analyze_age",
            "menu_recommendation": "/api/recommendations/menu",
            "legacy_http": "/api/voice (deprecated)",
        },
        "models": {
            "stt_ready": stt_status["ready"],
            "stt_error": stt_status["error"],
            "vision_warmup_on_startup": VISION_WARMUP_ON_STARTUP,
            "vision_ready": getattr(app.state, "vision_ready", False),
            "vision_error": getattr(app.state, "vision_error", None),
        },
    }


if __name__ == "__main__":
    import uvicorn
    from app.config import SERVER_HOST, SERVER_PORT

    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )
