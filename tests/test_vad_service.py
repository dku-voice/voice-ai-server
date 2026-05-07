import unittest
from unittest.mock import patch

import numpy as np

from app.services import vad_service
from app.services.vad_service import _collect_speech_audio


class VADServiceTest(unittest.TestCase):
    def test_collect_speech_audio_keeps_only_timestamp_ranges(self):
        audio = np.arange(20000, dtype=np.float32)
        timestamps = [{"start": 5000, "end": 6000}, {"start": 12000, "end": 13000}]

        speech_audio = _collect_speech_audio(audio, timestamps)

        self.assertLess(len(speech_audio), len(audio))
        self.assertEqual(speech_audio[0], audio[3080])
        self.assertEqual(speech_audio[-1], audio[14919])

    def test_detect_speech_falls_back_to_raw_audio_when_model_load_fails(self):
        audio_bytes = np.zeros(1600, dtype=np.int16).tobytes()

        with patch.object(vad_service, "_load_vad_model", side_effect=RuntimeError("load failed")):
            result = vad_service.detect_speech(audio_bytes)

        self.assertTrue(result["has_speech"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertIsNotNone(result["speech_audio"])
        self.assertEqual(result["speech_audio"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
