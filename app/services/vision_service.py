"""
app/services/vision_service.py - DeepFace 기반 스냅샷 연령 추정
프론트가 보내는 키오스크 시작 스냅샷을 저장하지 않고 바로 분석한다.
"""
import io
import logging
import threading
from typing import Any

import numpy as np
from starlette.concurrency import run_in_threadpool

from app.config import SENIOR_AGE_THRESHOLD, VISION_MAX_IMAGE_PIXELS

logger = logging.getLogger(__name__)

_deepface = None
_age_model_ready = False
_age_model_error: str | None = None
_age_model_lock = threading.Lock()


class VisionAnalysisError(RuntimeError):
    """스냅샷 이미지 분석 실패"""


class VisionConfigurationError(RuntimeError):
    """DeepFace 실행 환경 설정 오류"""


def _classify_age(age: int) -> tuple[str, bool]:
    if age >= SENIOR_AGE_THRESHOLD:
        return "senior", True
    if age >= 20:
        return "adult", False
    return "child", False


def _build_age_payload(age: int) -> dict:
    age_group, is_senior = _classify_age(age)
    return {
        "estimated_age": age,
        "age_group": age_group,
        "is_senior": is_senior,
    }


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """업로드 이미지를 RGB ndarray로 바꾼다. 원본 파일은 저장하지 않는다."""
    if not image_bytes:
        raise VisionAnalysisError("빈 이미지 파일입니다.")

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as e:
        raise VisionConfigurationError("Pillow 패키지가 설치되어 있지 않습니다.") from e

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise VisionAnalysisError("이미지 크기가 올바르지 않습니다.")

            pixel_count = width * height
            if pixel_count > VISION_MAX_IMAGE_PIXELS:
                raise VisionAnalysisError("이미지 해상도가 너무 큽니다.")

            return np.array(image.convert("RGB"))
    except Image.DecompressionBombError as e:
        raise VisionAnalysisError("이미지 해상도가 너무 큽니다.") from e
    except (UnidentifiedImageError, OSError) as e:
        raise VisionAnalysisError("이미지 파일을 읽을 수 없습니다.") from e


def _get_deepface():
    """DeepFace 모듈과 age 모델을 처음 필요할 때 로드한다."""
    global _deepface, _age_model_ready, _age_model_error

    if _deepface is not None and _age_model_ready:
        return _deepface

    if _age_model_error is not None:
        raise VisionConfigurationError(f"DeepFace age 모델 로딩 이력 실패: {_age_model_error}")

    with _age_model_lock:
        if _deepface is not None and _age_model_ready:
            return _deepface
        if _age_model_error is not None:
            raise VisionConfigurationError(f"DeepFace age 모델 로딩 이력 실패: {_age_model_error}")

        if _deepface is None:
            try:
                from deepface import DeepFace
            except ImportError as e:
                _age_model_error = str(e)
                raise VisionConfigurationError("DeepFace 패키지가 설치되어 있지 않습니다.") from e
            _deepface = DeepFace

        if not _age_model_ready:
            try:
                try:
                    _deepface.build_model(task="facial_attribute", model_name="Age")
                except TypeError:
                    _deepface.build_model("Age")
            except Exception as e:
                logger.error("[Vision] DeepFace age 모델 로딩 실패, 다음 요청에서 다시 시도합니다: %s", e)
                raise VisionConfigurationError("DeepFace age 모델 로딩에 실패했습니다.") from e
            _age_model_ready = True
            _age_model_error = None

    return _deepface


def _extract_age(deepface_result: Any) -> int:
    """DeepFace 결과가 dict/list 어느 쪽으로 오든 age 값만 꺼낸다."""
    result = deepface_result[0] if isinstance(deepface_result, list) else deepface_result
    if not isinstance(result, dict) or "age" not in result:
        raise VisionAnalysisError("DeepFace 응답에서 age 값을 찾지 못했습니다.")

    try:
        age = int(round(float(result["age"])))
    except (TypeError, ValueError) as e:
        raise VisionAnalysisError("DeepFace age 값이 올바르지 않습니다.") from e

    if age < 0 or age > 120:
        raise VisionAnalysisError("DeepFace age 값이 허용 범위를 벗어났습니다.")

    return age


def _analyze_age_image(deepface: Any, image: np.ndarray) -> Any:
    try:
        return deepface.analyze(
            img_path=image,
            actions=["age"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
    except TypeError:
        return deepface.analyze(
            img_path=image,
            actions=["age"],
            enforce_detection=False,
            detector_backend="opencv",
        )


def _run_deepface_age(image_bytes: bytes) -> dict:
    image = _decode_image_bytes(image_bytes)
    deepface = _get_deepface()

    try:
        result = _analyze_age_image(deepface, image)
    except Exception as e:
        logger.warning(f"[Vision] DeepFace 연령 추정 실패: {e}")
        raise VisionAnalysisError("DeepFace 연령 추정에 실패했습니다.") from e
    finally:
        del image

    return _build_age_payload(_extract_age(result))


def warm_up_deepface() -> None:
    """설정이 켜져 있으면 서버 시작 때 DeepFace age 모델을 미리 올린다."""
    logger.info("[Vision] DeepFace age 모델 워밍업 시작")
    deepface = _get_deepface()
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)

    try:
        _analyze_age_image(deepface, dummy)
    except Exception as e:
        logger.info("[Vision] DeepFace 더미 추론은 건너뜀: %s", e)
    finally:
        del dummy

    logger.info("[Vision] DeepFace age 모델 워밍업 완료")


async def analyze_age(image_bytes: bytes) -> dict:
    """DeepFace 추론은 무거워서 threadpool에서 실행한다."""
    return await run_in_threadpool(_run_deepface_age, image_bytes)
