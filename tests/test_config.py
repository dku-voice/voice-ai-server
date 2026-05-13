import importlib
import os
import unittest
from unittest.mock import patch

import app.config as config


class ConfigTest(unittest.TestCase):
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
            importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
