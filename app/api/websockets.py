"""
app/api/websockets.py - WebSocket 라우터
프론트엔드에서 마이크 오디오 chunk를 실시간으로 보내면
VAD → STT → LLM 파이프라인을 거쳐 주문 결과를 JSON으로 응답

v1.0: HTTP POST로 파일 통째로 업로드 → 느림 + 서버 블로킹
v2.0: WebSocket으로 스트리밍 → 빠름 + 비동기 처리
"""
import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas import VoiceOrderResponse
from app.services.vad_service import detect_speech_async, bytes_to_float32
from app.services.stt_service import transcribe
from app.services.llm_service import extract_order

logger = logging.getLogger(__name__)

router = APIRouter()

# HIGH-3 fix: 오디오 버퍼 최대 크기 (10MB ≈ 약 5분 16kHz 16bit mono)
MAX_AUDIO_BYTES = 10 * 1024 * 1024


@router.websocket("/ws/audio")
async def audio_websocket(ws: WebSocket):
    """
    WebSocket 음성 주문 처리 엔드포인트

    프론트엔드 흐름:
    1. WebSocket 연결
    2. 마이크 녹음 → PCM 바이너리 chunk 전송
    3. "END" 텍스트 메시지 수신 → 녹음 종료 신호
    4. 서버가 VAD → STT → LLM 파이프라인 실행
    5. JSON 결과 응답
    6. 연결 종료 또는 다음 주문 대기
    """
    await ws.accept()
    client_info = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    print(f"[WS] 클라이언트 연결됨: {client_info}")

    try:
        while True:
            # --- 오디오 chunk 수집 ---
            # 프론트에서 녹음하면서 chunk 단위로 계속 보냄
            # "END" 메시지가 오면 녹음 끝난 거임
            audio_buffer = bytearray()

            while True:
                data = await ws.receive()

                # FATAL-2 fix: 연결 종료 메시지 감지
                # ws.receive()는 disconnect 시 예외를 던지지 않고
                # {"type": "websocket.disconnect"} 를 반환함 → 무한루프 방지
                if data.get("type") == "websocket.disconnect":
                    print(f"[WS] 내부 루프에서 연결 종료 감지")
                    raise WebSocketDisconnect(code=data.get("code", 1000))

                # 텍스트 메시지 체크 ("END" = 녹음 종료)
                if "text" in data:
                    msg = data["text"]
                    if msg.strip().upper() == "END":
                        print(f"[WS] 녹음 종료 신호 수신 (수집된 bytes: {len(audio_buffer)})")
                        break
                    # END 아닌 텍스트는 무시
                    continue

                # 바이너리 데이터 = 오디오 chunk
                if "bytes" in data:
                    chunk = data["bytes"]
                    audio_buffer.extend(chunk)

                    # HIGH-3 fix: 버퍼 크기 제한 (메모리 DoS 방지)
                    if len(audio_buffer) > MAX_AUDIO_BYTES:
                        print(f"[WS] ⚠️ 오디오 버퍼 한도 초과: {len(audio_buffer)} bytes")
                        await ws.send_json(
                            VoiceOrderResponse(
                                status="error",
                                recognized_text="",
                                items=[],
                                error_msg="오디오 데이터가 너무 큽니다 (최대 10MB)",
                            ).model_dump()
                        )
                        audio_buffer.clear()
                        break

                    continue

            # 버퍼가 비었으면 스킵 (overflow clear 포함)
            if len(audio_buffer) == 0:
                # overflow로 clear된 경우는 이미 에러 응답 보냄 → 중복 방지
                # END 직후 빈 버퍼인 경우만 에러 응답
                if not (data.get("type") == "websocket.disconnect"):
                    print("[WS] 빈 오디오 버퍼, 스킵")
                    # overflow clear 후에는 이미 에러 보냈으므로 다시 안 보냄
                continue

            audio_bytes = bytes(audio_buffer)

            # ===== Stage 1: VAD (음성 감지) =====
            print("[WS] Stage 1: VAD 처리 중...")
            vad_result = await detect_speech_async(audio_bytes)  # FATAL-1 fix: 비동기 호출

            if not vad_result["has_speech"]:
                print("[WS] VAD: 음성 없음 → 무시")
                await ws.send_json(
                    VoiceOrderResponse(
                        status="success",
                        recognized_text="",
                        items=[],
                        error_msg=None,
                    ).model_dump()
                )
                continue

            speech_audio = vad_result["speech_audio"]  # float32 ndarray

            # FATAL-3 fix: VAD fallback 시 speech_audio가 None일 수 있음
            if speech_audio is None:
                print("[WS] VAD fallback: speech_audio 없음 → 원본 bytes로 STT 시도")
                try:
                    speech_audio = bytes_to_float32(audio_bytes)
                except Exception:
                    await ws.send_json(
                        VoiceOrderResponse(
                            status="error",
                            recognized_text="",
                            items=[],
                            error_msg="오디오 데이터 변환 실패",
                        ).model_dump()
                    )
                    continue

            # ===== Stage 2: STT (음성 → 텍스트) =====
            print("[WS] Stage 2: STT 처리 중...")
            try:
                recognized_text = await transcribe(speech_audio)
                print(f"[WS] STT 결과: '{recognized_text}'")
            except Exception as e:
                print(f"[WS] STT 에러: {e}")
                await ws.send_json(
                    VoiceOrderResponse(
                        status="error",
                        recognized_text="",
                        items=[],
                        error_msg=f"STT 처리 실패: {str(e)}",
                    ).model_dump()
                )
                continue

            # STT 결과가 비었으면 (무음이었거나 인식 불가)
            if not recognized_text:
                await ws.send_json(
                    VoiceOrderResponse(
                        status="success",
                        recognized_text="",
                        items=[],
                        error_msg=None,
                    ).model_dump()
                )
                continue

            # ===== Stage 3: LLM (텍스트 → 주문 파싱) =====
            print("[WS] Stage 3: LLM 주문 파싱 중...")
            llm_result = await extract_order(recognized_text)

            # 최종 응답 조립
            response = VoiceOrderResponse(
                status=llm_result["status"],
                recognized_text=recognized_text,
                items=llm_result["items"],
                error_msg=llm_result.get("error_msg"),
            )

            await ws.send_json(response.model_dump())
            print(f"[WS] 응답 전송 완료: status={response.status}, items={len(response.items)}건")

    except WebSocketDisconnect:
        # 클라이언트가 연결 끊음 (정상 종료)
        # 브라우저 탭 닫거나 네트워크 끊기면 여기로 옴
        print(f"[WS] 클라이언트 연결 해제: {client_info}")
        logger.info(f"[WS] WebSocket 연결 종료: {client_info}")

    except Exception as e:
        # 예상 못한 에러 (서버 내부 문제)
        print(f"[WS] 예상치 못한 에러: {e}")
        logger.error(f"[WS] 핸들러 에러: {e}", exc_info=True)
        try:
            await ws.send_json(
                VoiceOrderResponse(
                    status="error",
                    recognized_text="",
                    items=[],
                    error_msg=f"서버 내부 에러: {str(e)}",
                ).model_dump()
            )
        except Exception:
            pass  # 이미 연결 끊긴 상태일 수 있음
