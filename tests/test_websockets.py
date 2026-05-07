import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.websockets import router


class WebSocketAudioTest(unittest.TestCase):
    def test_empty_end_returns_empty_success_response(self):
        app = FastAPI()
        app.include_router(router)

        with TestClient(app).websocket_connect("/ws/audio") as ws:
            ws.send_text("END")
            result = ws.receive_json()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recognized_text"], "")
        self.assertEqual(result["items"], [])
        self.assertIsNone(result["error_msg"])


if __name__ == "__main__":
    unittest.main()
