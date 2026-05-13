import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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

    def test_audio_over_limit_returns_error_and_closes(self):
        app = FastAPI()
        app.include_router(router)

        with patch("app.api.websockets.MAX_AUDIO_BYTES", 4):
            with TestClient(app).websocket_connect("/ws/audio") as ws:
                ws.send_bytes(b"123456")
                result = ws.receive_json()

                self.assertEqual(result["status"], "error")
                self.assertIn("오디오 데이터가 너무 큽니다", result["error_msg"])
                with self.assertRaises(WebSocketDisconnect) as context:
                    ws.receive_json()

        self.assertEqual(context.exception.code, 1009)


if __name__ == "__main__":
    unittest.main()
