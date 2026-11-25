from unittest.mock import patch
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
            def transcribe(self, audio_data, language, beam_size, hotwords=None):
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

    def test_screen_language_wins_over_unsure_detection(self):
        self.assertEqual(stt_service._pick_language(self._info("ko", 0.62), "es"), "es")
        self.assertEqual(stt_service._pick_language(self._info("th", 0.31), "th"), "th")
        self.assertEqual(stt_service._pick_language(self._info("ja", 0.45), "es"), "es")

    def test_clear_other_language_still_switches_screen(self):
        self.assertEqual(stt_service._pick_language(self._info("ko", 0.95), "es"), "ko")
        self.assertEqual(stt_service._pick_language(self._info("ja", 0.92), "ko"), "ja")

    def test_unknown_screen_language_is_ignored(self):
        self.assertEqual(stt_service._pick_language(self._info("th", 0.31), "en"), "ko")

    def test_unsupported_language_without_probs_defaults_to_korean(self):
        self.assertEqual(stt_service._pick_language(self._info("en", 0.99)), "ko")

    def test_retranscribes_when_detected_language_is_not_supported(self):
        old_model = stt_service._whisper_model
        calls = []

        class DummyModel:
            def transcribe(self, audio_data, language, beam_size, hotwords=None):
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

    def test_detected_language_uses_menu_hotwords(self):
        old_model = stt_service._whisper_model
        calls = []

        class DummyModel:
            def transcribe(self, audio_data, language, beam_size, hotwords=None):
                calls.append((language, hotwords))
                info = SimpleNamespace(language="ko", language_probability=0.97, all_language_probs=None)
                return iter([SimpleNamespace(text="콜라 세 개")]), info

        stt_service._whisper_model = DummyModel()
        try:
            result = stt_service._transcribe_sync(np.zeros(1600, dtype=np.float32))
        finally:
            stt_service._whisper_model = old_model

        self.assertEqual(result, ("콜라 세 개", "ko"))
        self.assertEqual(calls[0], (None, None))
        self.assertEqual(calls[1][0], "ko")
        self.assertIn("콜라", calls[1][1])

    def _run_with_screen(self, screen, detected, probability):
        old_model = stt_service._whisper_model
        calls = []

        class DummyModel:
            def transcribe(self, audio_data, language, beam_size, hotwords=None):
                calls.append((language, hotwords))
                info = SimpleNamespace(language=language or detected, language_probability=probability, all_language_probs=None)
                return iter([SimpleNamespace(text="texto")]), info

        stt_service._whisper_model = DummyModel()
        names = {"ko": ("데리버거", "콜라"), "es": ("Hamburguesa Teri", "Cola")}
        try:
            with patch("app.services.llm_service.menu_names_for_language", side_effect=lambda lang: names.get(lang, ())):
                result = stt_service._transcribe_sync(np.zeros(1600, dtype=np.float32), screen)
        finally:
            stt_service._whisper_model = old_model
        return result, calls

    def test_screen_language_hotwords_need_only_one_pass(self):
        result, calls = self._run_with_screen("es", "es", 0.9)
        self.assertEqual(result[1], "es")
        self.assertEqual(len(calls), 1)
        self.assertIn("Hamburguesa Teri", calls[0][1])

    def test_korean_screen_needs_only_one_pass(self):
        result, calls = self._run_with_screen("ko", "ko", 0.97)
        self.assertEqual(len(calls), 1)
        self.assertIn("데리버거", calls[0][1])

    def test_other_language_on_screen_still_retranscribes(self):
        result, calls = self._run_with_screen("es", "ko", 0.97)
        self.assertEqual(result[1], "ko")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], "ko")
        self.assertIn("데리버거", calls[1][1])

    def test_korean_hotwords_drop_brand_prefix(self):
        with patch("app.services.llm_service.menu_names_for_language", return_value=("데리버거", "리아 불고기버거", "리아 새우버거")):
            hotwords = stt_service._hotwords_for("ko")
        self.assertEqual(hotwords, "데리버거, 불고기버거, 새우버거")

    def test_hotwords_are_not_used_for_foreign_languages(self):
        self.assertIsNone(stt_service._hotwords_for("ja"))
        self.assertIsNone(stt_service._hotwords_for("th"))
        self.assertIn("콜라", stt_service._hotwords_for("ko"))


if __name__ == "__main__":
    unittest.main()
