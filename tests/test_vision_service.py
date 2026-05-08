import io
import unittest

from app.services import vision_service
from app.services.vision_service import (
    VisionAnalysisError,
    VisionConfigurationError,
    _build_age_payload,
    _classify_age,
    _decode_image_bytes,
    _extract_age,
    _get_deepface,
)


class VisionServiceTest(unittest.TestCase):
    def test_classify_age_groups(self):
        self.assertEqual(_classify_age(12), ("child", False))
        self.assertEqual(_classify_age(35), ("adult", False))
        self.assertEqual(_classify_age(60), ("senior", True))

    def test_build_age_payload(self):
        payload = _build_age_payload(72)

        self.assertEqual(payload["estimated_age"], 72)
        self.assertEqual(payload["age_group"], "senior")
        self.assertTrue(payload["is_senior"])

    def test_extract_age_from_dict_result(self):
        self.assertEqual(_extract_age({"age": 44.6}), 45)

    def test_extract_age_from_list_result(self):
        self.assertEqual(_extract_age([{"age": "21"}]), 21)

    def test_reject_empty_deepface_list_result(self):
        with self.assertRaises(VisionAnalysisError):
            _extract_age([])

    def test_reject_invalid_age_result(self):
        with self.assertRaises(VisionAnalysisError):
            _extract_age({"age": 150})

    def test_reject_missing_age_result(self):
        with self.assertRaises(VisionAnalysisError):
            _extract_age({"dominant_gender": "Man"})

    def test_reject_image_when_decoded_pixels_exceed_limit(self):
        from PIL import Image

        old_limit = vision_service.VISION_MAX_IMAGE_PIXELS
        vision_service.VISION_MAX_IMAGE_PIXELS = 4
        try:
            buffer = io.BytesIO()
            Image.new("RGB", (3, 2), color="white").save(buffer, format="PNG")

            with self.assertRaises(VisionAnalysisError):
                _decode_image_bytes(buffer.getvalue())
        finally:
            vision_service.VISION_MAX_IMAGE_PIXELS = old_limit

    def test_deepface_model_load_failure_can_retry_next_request(self):
        old_deepface = vision_service._deepface
        old_ready = vision_service._age_model_ready
        old_error = vision_service._age_model_error

        class DummyDeepFace:
            calls = 0

            @classmethod
            def build_model(cls, *args, **kwargs):
                cls.calls += 1
                if cls.calls == 1:
                    raise RuntimeError("temporary download failure")
                return object()

        vision_service._deepface = DummyDeepFace
        vision_service._age_model_ready = False
        vision_service._age_model_error = None

        try:
            with self.assertRaises(VisionConfigurationError):
                _get_deepface()

            self.assertIsNone(vision_service._age_model_error)
            self.assertFalse(vision_service._age_model_ready)
            self.assertIs(_get_deepface(), DummyDeepFace)
            self.assertTrue(vision_service._age_model_ready)
            self.assertEqual(DummyDeepFace.calls, 2)
        finally:
            vision_service._deepface = old_deepface
            vision_service._age_model_ready = old_ready
            vision_service._age_model_error = old_error


if __name__ == "__main__":
    unittest.main()
