"""
app/api/websockets.py - WebSocket 라우터
프론트엔드가 마이크 오디오 chunk를 보내면
VAD → STT → LLM 순서로 처리해서 주문 결과를 JSON으로 보낸다.

v1.0에서는 HTTP로 파일을 한 번에 받았고, 지금은 WebSocket으로 chunk를 받는다.
"""
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import WS_RECEIVE_TIMEOUT_SECONDS
from app.schemas import VoiceOrderResponse
from app.services.vad_service import detect_speech_async, bytes_to_float32
from app.services.stt_service import transcribe
from app.services.llm_service import extract_order

logger = logging.getLogger(__name__)

router = APIRouter()

# 너무 큰 오디오가 들어오면 메모리가 커지므로 10MB에서 끊는다.
MAX_AUDIO_BYTES = 10 * 1024 * 1024

RAW_PCM_FORMAT_ERROR = (
    "지원하지 않는 오디오 형식입니다. "
    "raw PCM 16kHz mono 16-bit little-endian으로 전송하세요."
)

UNSUPPORTED_AUDIO_HEADERS = {
    b"RIFF": "WAV",
    b"OggS": "Ogg/Opus",
    b"\x1A\x45\xDF\xA3": "WebM/Matroska",
    b"ID3": "MP3",
    b"fLaC": "FLAC",
}


def _detect_unsupported_audio_container(audio_bytes: bytes) -> str | None:
    """raw PCM이 아닌 흔한 오디오 컨테이너를 먼저 걸러낸다."""
    for header, format_name in UNSUPPORTED_AUDIO_HEADERS.items():
        if audio_bytes.startswith(header):
            return format_name

    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return "MP4/M4A"

    return None


def _validate_raw_pcm_audio(audio_bytes: bytes) -> str | None:
    """VAD/STT에 넘기기 전에 raw PCM인지 최소한만 확인한다."""
    container = _detect_unsupported_audio_container(audio_bytes)
    if container:
        return f"{RAW_PCM_FORMAT_ERROR} 감지된 형식: {container}"

    if len(audio_bytes) % 2 != 0:
        return "PCM 16-bit 오디오는 바이트 길이가 짝수여야 합니다."

    return None


@router.websocket("/ws/audio")
async def audio_websocket(ws: WebSocket):
    """
    WebSocket 음성 주문 처리 엔드포인트

    프론트엔드 흐름:
    1. WebSocket 연결
    2. 마이크 녹음 → PCM 바이너리 chunk 전송
    3. "END" 텍스트 메시지 수신 → 녹음 종료 신호
    4. 서버가 VAD → STT → LLM 파이프라인 실행
    5. JSON 결과 전송
    6. 연결 종료 또는 다음 주문 대기
    """
    await ws.accept()
    client_info = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    print(f"[WS] 클라이언트 연결됨: {client_info}")

    try:
        while True:
            # 프론트가 녹음 중에는 bytes chunk를 계속 보내고,
            # "END"가 오면 한 주문이 끝난 것으로 본다.
            audio_buffer = bytearray()
            buffer_overflowed = False

            while True:
                try:
                    data = await asyncio.wait_for(
                        ws.receive(),
                        timeout=WS_RECEIVE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    print(f"[WS] 수신 timeout: {WS_RECEIVE_TIMEOUT_SECONDS}초 동안 메시지 없음")
                    await ws.send_json(
                        VoiceOrderResponse(
                            status="error",
                            recognized_text="",
                            items=[],
                            error_msg="오디오 수신 시간이 초과되었습니다. 다시 시도하세요.",
                        ).model_dump()
                    )
                    await ws.close(code=1008)
                    return

                # ws.receive()가 disconnect 메시지를 dict로 줄 때가 있어서
                # 여기서 직접 확인하지 않으면 루프가 계속 돌 수 있다.
                if data.get("type") == "websocket.disconnect":
                    print(f"[WS] 내부 루프에서 연결 종료 감지")
                    raise WebSocketDisconnect(code=data.get("code", 1000))

                # 텍스트 메시지는 지금 "END"만 의미 있게 쓴다.
                if "text" in data:
                    msg = data["text"]
                    if msg.strip().upper() == "END":
                        print(f"[WS] 녹음 종료 신호 수신 (수집된 bytes: {len(audio_buffer)})")
                        break
                    # END가 아닌 텍스트는 일단 무시한다.
                    continue

                # 바이너리 데이터는 오디오 chunk로 본다.
                if "bytes" in data:
                    chunk = data["bytes"]
                    audio_buffer.extend(chunk)

                    # 너무 긴 녹음은 여기서 끊는다.
                    if len(audio_buffer) > MAX_AUDIO_BYTES:
                        print(f"[WS] 오디오 버퍼 한도 초과: {len(audio_buffer)} bytes")
                        await ws.send_json(
                            VoiceOrderResponse(
                                status="error",
                                recognized_text="",
                                items=[],
                                error_msg="오디오 데이터가 너무 큽니다 (최대 10MB)",
                            ).model_dump()
                        )
                        audio_buffer.clear()
                        buffer_overflowed = True
                        break

                    continue

            # 빈 END만 와도 클라이언트가 계속 기다리지 않도록 바로 응답한다.
            if len(audio_buffer) == 0:
                if buffer_overflowed:
                    continue
                print("[WS] 빈 오디오 버퍼, 빈 주문 응답")
                await ws.send_json(
                    VoiceOrderResponse(
                        status="success",
                        recognized_text="",
                        items=[],
                        error_msg=None,
                    ).model_dump()
                )
                continue

            audio_bytes = bytes(audio_buffer)
            audio_error = _validate_raw_pcm_audio(audio_bytes)
            if audio_error:
                print(f"[WS] 지원하지 않는 오디오 입력: {audio_error}")
                await ws.send_json(
                    VoiceOrderResponse(
                        status="error",
                        recognized_text="",
                        items=[],
                        error_msg=audio_error,
                    ).model_dump()
                )
                continue

            # ===== Stage 1: VAD (음성 감지) =====
            print("[WS] Stage 1: VAD 처리 중...")
            vad_result = await detect_speech_async(audio_bytes)

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

            # VAD fallback에서는 speech_audio가 None일 수 있어서 원본 bytes를 다시 변환한다.
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

            # STT 결과가 비면 주문이 없는 것으로 처리한다.
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

            # 프론트로 보낼 최종 응답
            response = VoiceOrderResponse(
                status=llm_result["status"],
                recognized_text=recognized_text,
                items=llm_result["items"],
                error_msg=llm_result.get("error_msg"),
            )

            await ws.send_json(response.model_dump())
            print(f"[WS] 응답 전송 완료: status={response.status}, items={len(response.items)}건")

    except WebSocketDisconnect:
        # 브라우저 탭을 닫거나 네트워크가 끊기면 여기로 온다.
        print(f"[WS] 클라이언트 연결 해제: {client_info}")
        logger.info(f"[WS] WebSocket 연결 종료: {client_info}")

    except Exception as e:
        # 예상 못한 에러가 나도 가능하면 JSON으로 알려준다.
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
            pass  # 이미 연결이 끊겼을 수 있다.
