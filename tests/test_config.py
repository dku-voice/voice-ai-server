import importlib
import os
import unittest
from unittest.mock import patch

import app.config as config


class ConfigTest(unittest.TestCase):
    def tearDown(self):
        importlib.reload(config)

    def test_server_port_falls_back_when_env_is_not_numeric(self):
        original_port = os.environ.get("SERVER_PORT")

        try:
            with patch.dict(os.environ, {"SERVER_PORT": "not-a-number"}):
                reloaded = importlib.reload(config)
                self.assertEqual(reloaded.SERVER_PORT, 8000)
        finally:
            if original_port is None:
                os.environ.pop("SERVER_PORT", None)
            else:
                os.environ["SERVER_PORT"] = original_port

    def test_numeric_envs_fall_back_when_values_are_invalid(self):
        invalid_env = {
            "VAD_THRESHOLD": "not-a-number",
            "WS_RECEIVE_TIMEOUT_SECONDS": "not-a-number",
            "VISION_MAX_IMAGE_BYTES": "not-a-number",
            "VISION_MAX_IMAGE_PIXELS": "not-a-number",
            "SENIOR_AGE_THRESHOLD": "not-a-number",
            "VISION_MODEL_RETRY_COOLDOWN_SECONDS": "inf",
        }

        with patch.dict(os.environ, invalid_env):
            reloaded = importlib.reload(config)

        self.assertEqual(reloaded.VAD_THRESHOLD, 0.6)
        self.assertEqual(reloaded.WS_RECEIVE_TIMEOUT_SECONDS, 30.0)
        self.assertEqual(reloaded.VISION_MAX_IMAGE_BYTES, 5 * 1024 * 1024)
        self.assertEqual(reloaded.VISION_MAX_IMAGE_PIXELS, 6_000_000)
        self.assertEqual(reloaded.SENIOR_AGE_THRESHOLD, 60)
        self.assertEqual(reloaded.VISION_MODEL_RETRY_COOLDOWN_SECONDS, 10.0)

    def test_numeric_envs_keep_minimum_safe_values(self):
        low_env = {
            "SERVER_PORT": "0",
            "WS_RECEIVE_TIMEOUT_SECONDS": "-5",
            "VISION_MAX_IMAGE_BYTES": "0",
            "VISION_MAX_IMAGE_PIXELS": "0",
            "SENIOR_AGE_THRESHOLD": "0",
            "VISION_MODEL_RETRY_COOLDOWN_SECONDS": "-1",
        }

        with patch.dict(os.environ, low_env):
            reloaded = importlib.reload(config)

        self.assertEqual(reloaded.SERVER_PORT, 1)
        self.assertEqual(reloaded.WS_RECEIVE_TIMEOUT_SECONDS, 1.0)
        self.assertEqual(reloaded.VISION_MAX_IMAGE_BYTES, 1)
        self.assertEqual(reloaded.VISION_MAX_IMAGE_PIXELS, 1)
        self.assertEqual(reloaded.SENIOR_AGE_THRESHOLD, 1)
        self.assertEqual(reloaded.VISION_MODEL_RETRY_COOLDOWN_SECONDS, 0.0)


if __name__ == "__main__":
    unittest.main()
