"""
app/api/vision.py - 스냅샷 이미지 분석 라우터
키오스크 시작 시점에 프론트엔드가 보낸 사진으로 고령자 UI 전환 여부를 판단한다.
"""
import logging

from fastapi import APIRouter, File, UploadFile

from app.config import VISION_MAX_IMAGE_BYTES
from app.schemas import AgeAnalysisResponse
from app.services.vision_service import (
    VisionAnalysisError,
    VisionConfigurationError,
    analyze_age,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def _error_response(message: str) -> AgeAnalysisResponse:
    return AgeAnalysisResponse(
        status="error",
        estimated_age=None,
        age_group="unknown",
        is_senior=False,
        error_msg=message,
    )


@router.post("/analyze_age", response_model=AgeAnalysisResponse)
async def analyze_age_snapshot(file: UploadFile = File(...)):
    """
    DeepFace 스냅샷 연령 추정 엔드포인트.
    이미지는 저장하지 않고 요청 단위 메모리에서만 처리한다.
    """
    if file.content_type and file.content_type not in SUPPORTED_IMAGE_TYPES:
        await file.close()
        return _error_response("지원하지 않는 이미지 형식입니다. jpg, png, webp 파일을 전송하세요.")

    image_bytes = await file.read(VISION_MAX_IMAGE_BYTES + 1)
    await file.close()

    if len(image_bytes) > VISION_MAX_IMAGE_BYTES:
        return _error_response("이미지 파일이 너무 큽니다.")

    try:
        result = await analyze_age(image_bytes)
        return AgeAnalysisResponse(
            status="success",
            estimated_age=result["estimated_age"],
            age_group=result["age_group"],
            is_senior=result["is_senior"],
            error_msg=None,
        )
    except VisionConfigurationError as e:
        logger.warning("[Vision] DeepFace 설정 오류: %s", e)
        return _error_response(str(e))
    except VisionAnalysisError as e:
        logger.warning("[Vision] 분석 실패: %s", e)
        return _error_response(str(e))
    except Exception as e:
        logger.error("[Vision] 예상치 못한 오류: %s", e, exc_info=True)
        return _error_response("스냅샷 연령 추정 중 오류가 발생했습니다.")
    finally:
        del image_bytes
