import unittest
from types import SimpleNamespace

import numpy as np

from app.services import stt_service


class STTServiceTest(unittest.TestCase):
    def test_transcribe_consumes_segments_under_inference_lock(self):
        old_model = stt_service._whisper_model
        old_error = stt_service._whisper_model_error
        old_lock = stt_service._whisper_transcribe_lock

        class RecordingLock:
            locked = False

            def __enter__(self):
                self.locked = True

            def __exit__(self, exc_type, exc, tb):
                self.locked = False

        lock = RecordingLock()

        class DummyModel:
            def transcribe(self, audio_data, language, beam_size):
                def segments():
                    if not lock.locked:
                        raise AssertionError("STT inference lock was not held")
                    yield SimpleNamespace(text=" 콜라 ")

                info = SimpleNamespace(language="ko", language_probability=0.99)
                return segments(), info

        stt_service._whisper_model = DummyModel()
        stt_service._whisper_model_error = None
        stt_service._whisper_transcribe_lock = lock

        try:
            result = stt_service._transcribe_sync(np.zeros(1600, dtype=np.float32))
        finally:
            stt_service._whisper_model = old_model
            stt_service._whisper_model_error = old_error
            stt_service._whisper_transcribe_lock = old_lock

        self.assertEqual(result, ("콜라", "ko"))


class LanguagePickTest(unittest.TestCase):
    def _info(self, language, prob, all_probs=None):
        return SimpleNamespace(
            language=language,
            language_probability=prob,
            all_language_probs=all_probs,
        )

    def test_supported_language_is_kept(self):
        self.assertEqual(stt_service._pick_language(self._info("ja", 0.92)), "ja")

    def test_low_confidence_foreign_language_falls_back_to_korean(self):
        self.assertEqual(stt_service._pick_language(self._info("th", 0.31)), "ko")

    def test_unsupported_language_uses_best_supported_candidate(self):
        info = self._info("zh", 0.6, [("zh", 0.6), ("ja", 0.3), ("ko", 0.1)])
        self.assertEqual(stt_service._pick_language(info), "ko")
        info = self._info("pt", 0.4, [("pt", 0.4), ("es", 0.55), ("ko", 0.05)])
        self.assertEqual(stt_service._pick_language(info), "es")

    def test_unsupported_language_without_probs_defaults_to_korean(self):
        self.assertEqual(stt_service._pick_language(self._info("en", 0.99)), "ko")

    def test_retranscribes_when_detected_language_is_not_supported(self):
        old_model = stt_service._whisper_model
        calls = []

        class DummyModel:
            def transcribe(self, audio_data, language, beam_size):
                calls.append(language)
                if language is None:
                    info = SimpleNamespace(
                        language="pt",
                        language_probability=0.4,
                        all_language_probs=[("pt", 0.4), ("es", 0.55)],
                    )
                    return iter([SimpleNamespace(text="uma coca")]), info
                info = SimpleNamespace(language=language, language_probability=1.0)
                return iter([SimpleNamespace(text="una coca cola")]), info

        stt_service._whisper_model = DummyModel()
        try:
            result = stt_service._transcribe_sync(np.zeros(1600, dtype=np.float32))
        finally:
            stt_service._whisper_model = old_model

        self.assertEqual(calls, [None, "es"])
        self.assertEqual(result, ("una coca cola", "es"))


if __name__ == "__main__":
    unittest.main()
