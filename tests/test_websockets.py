import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.websockets import router
from app.schemas import OrderItem


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


class WebSocketAudioTest(unittest.TestCase):
    def test_empty_end_returns_empty_success_response(self):
        with TestClient(_build_app()).websocket_connect("/ws/audio") as ws:
            ws.send_text("END")
            result = ws.receive_json()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recognized_text"], "")
        self.assertEqual(result["items"], [])
        self.assertIsNone(result["error_msg"])

    def test_audio_over_limit_returns_error_and_closes(self):
        with patch("app.api.websockets.MAX_AUDIO_BYTES", 4):
            with TestClient(_build_app()).websocket_connect("/ws/audio") as ws:
                ws.send_bytes(b"123456")
                result = ws.receive_json()

                self.assertEqual(result["status"], "error")
                self.assertIn("오디오 데이터가 너무 큽니다", result["error_msg"])
                with self.assertRaises(WebSocketDisconnect) as context:
                    ws.receive_json()

        self.assertEqual(context.exception.code, 1009)

    def test_unsupported_audio_container_returns_error(self):
        # WAV(RIFF) 헤더가 들어오면 raw PCM이 아니므로 즉시 거절해야 한다.
        wav_like_bytes = b"RIFF" + b"\x00" * 40

        with TestClient(_build_app()).websocket_connect("/ws/audio") as ws:
            ws.send_bytes(wav_like_bytes)
            ws.send_text("END")
            result = ws.receive_json()

        self.assertEqual(result["status"], "error")
        self.assertIn("raw PCM", result["error_msg"])
        self.assertIn("WAV", result["error_msg"])

    def test_vad_no_speech_returns_empty_success(self):
        audio_bytes = np.zeros(1600, dtype=np.int16).tobytes()
        vad_result = {"has_speech": False, "speech_audio": None, "confidence": 0.0}

        with patch(
            "app.api.websockets.detect_speech_async",
            new=AsyncMock(return_value=vad_result),
        ):
            with TestClient(_build_app()).websocket_connect("/ws/audio") as ws:
                ws.send_bytes(audio_bytes)
                ws.send_text("END")
                result = ws.receive_json()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recognized_text"], "")
        self.assertEqual(result["items"], [])

    def test_stt_failure_returns_error_response(self):
        audio_bytes = np.zeros(1600, dtype=np.int16).tobytes()
        vad_result = {
            "has_speech": True,
            "speech_audio": np.zeros(1600, dtype=np.float32),
            "confidence": 0.99,
        }

        with patch(
            "app.api.websockets.detect_speech_async",
            new=AsyncMock(return_value=vad_result),
        ), patch(
            "app.api.websockets.transcribe",
            new=AsyncMock(side_effect=RuntimeError("decoder boom")),
        ):
            with TestClient(_build_app()).websocket_connect("/ws/audio") as ws:
                ws.send_bytes(audio_bytes)
                ws.send_text("END")
                result = ws.receive_json()

        self.assertEqual(result["status"], "error")
        self.assertIn("STT 처리 실패", result["error_msg"])

    def test_full_chain_happy_path_returns_voice_order(self):
        # Week 6 mock-first 통합 테스트: VAD → STT → LLM 정상 흐름.
        audio_bytes = np.zeros(3200, dtype=np.int16).tobytes()
        vad_result = {
            "has_speech": True,
            "speech_audio": np.zeros(3200, dtype=np.float32),
            "confidence": 0.99,
        }
        llm_result = {
            "status": "success",
            "items": [OrderItem(menu_id="cola_01", quantity=1, options=[])],
            "error_msg": None,
        }

        with patch(
            "app.api.websockets.detect_speech_async",
            new=AsyncMock(return_value=vad_result),
        ), patch(
            "app.api.websockets.transcribe",
            new=AsyncMock(return_value="콜라 하나 주세요"),
        ), patch(
            "app.api.websockets.extract_order",
            new=AsyncMock(return_value=llm_result),
        ):
            with TestClient(_build_app()).websocket_connect("/ws/audio") as ws:
                ws.send_bytes(audio_bytes)
                ws.send_text("END")
                result = ws.receive_json()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recognized_text"], "콜라 하나 주세요")
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["menu_id"], "cola_01")
        self.assertEqual(result["items"][0]["quantity"], 1)
        self.assertIsNone(result["error_msg"])


if __name__ == "__main__":
    unittest.main()
