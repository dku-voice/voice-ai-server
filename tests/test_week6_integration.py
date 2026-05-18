"""Week 6 통합 테스트: 헬스 체크가 모델 로딩 없이도 정상 응답하는지 확인."""
import unittest
from unittest.mock import patch

import main


class HealthCheckIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_health_check_does_not_trigger_model_load(self):
        # GET /가 모델 다운로드/워밍업을 부르면 안 된다. 첫 페이지 호출만으로
        # STT/DeepFace 로딩이 따라오면 응답이 수십 초 단위로 늘어질 수 있다.
        with patch("main.load_stt_model") as mock_load_stt, \
             patch("main.warm_up_deepface") as mock_warm_vision, \
             patch("main.run_model_task") as mock_run_model_task:
            result = await main.health_check()

        self.assertEqual(result["version"], "2.0.0")
        self.assertIn("websocket", result["endpoints"])
        self.assertIn("vision_age", result["endpoints"])
        self.assertIn("menu_recommendation", result["endpoints"])
        self.assertIn("stt_ready", result["models"])
        self.assertIn("vision_ready", result["models"])
        mock_load_stt.assert_not_called()
        mock_warm_vision.assert_not_called()
        mock_run_model_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
