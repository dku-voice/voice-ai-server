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

        self.assertEqual(result, "콜라")


if __name__ == "__main__":
    unittest.main()
