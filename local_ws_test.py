"""
local_ws_test.py - V2.0 WebSocket 로컬 자체 테스트 스크립트
.wav 파일을 읽어서 chunk 단위로 WebSocket에 보내고 결과 확인

사용법:
    python local_ws_test.py test_audio.wav
    python local_ws_test.py   (기본: test.wav)

서버가 먼저 실행되어 있어야 함:
    uvicorn main:app --reload
"""
import asyncio
import sys
import json
import wave

import websockets

from app.config import configure_console_encoding


configure_console_encoding()


# --- 설정 ---
WS_URL = "ws://localhost:8000/ws/audio"
CHUNK_SIZE = 4096  # bytes 단위 (프론트엔드 마이크 chunk랑 비슷하게)


def read_wav_file(filepath: str) -> tuple[bytes, int]:
    """
    .wav 파일 읽어서 raw PCM bytes + sample rate 반환
    
    # wav 파일 포맷 주의:
    # 16-bit PCM, mono, 16kHz여야 VAD가 제대로 동작함
    # 다른 포맷이면 ffmpeg로 변환 필요:
    # ffmpeg -i input.wav -ar 16000 -ac 1 -f s16le output.wav
    """
    print(f">>> 파일 열기: {filepath}")

    with wave.open(filepath, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    duration = n_frames / framerate
    print(f">>> WAV 정보: {channels}ch, {sample_width*8}bit, {framerate}Hz, {duration:.2f}초")
    print(f">>> 총 바이트: {len(raw_data)} bytes")

    if channels != 1:
        print(f"⚠️  경고: 모노(1ch)가 아님! ({channels}ch) → VAD 결과 이상할 수 있음")
    if framerate != 16000:
        print(f"⚠️  경고: 16kHz가 아님! ({framerate}Hz) → Silero VAD는 16kHz만 지원")

    return raw_data, framerate


def pretty_print_result(result: dict):
    """결과 예쁘게 출력"""
    print()
    print("=" * 60)
    print("📋 V2.0 파이프라인 처리 결과")
    print("=" * 60)

    status = result.get("status", "unknown")
    status_emoji = {"success": "✅", "fallback": "⚠️", "error": "❌", "deprecated": "🚫"}.get(status, "❓")

    print(f"  상태: {status_emoji} {status}")
    print(f"  인식 텍스트: \"{result.get('recognized_text', '')}\"")

    items = result.get("items", [])
    if items:
        print(f"  주문 항목: ({len(items)}건)")
        for i, item in enumerate(items, 1):
            options_str = ", ".join(item.get("options", [])) if item.get("options") else "없음"
            print(f"    [{i}] {item.get('menu_id', '?')} x {item.get('quantity', '?')} (옵션: {options_str})")
    else:
        print("  주문 항목: 없음")

    error_msg = result.get("error_msg")
    if error_msg:
        print(f"  에러: {error_msg}")

    print("=" * 60)
    print()


async def test_websocket(filepath: str):
    """WebSocket 연결 → chunk 전송 → 결과 수신"""
    raw_data, sample_rate = read_wav_file(filepath)

    print(f"\n>>> WebSocket 연결 중: {WS_URL}")

    try:
        async with websockets.connect(WS_URL) as ws:
            print(">>> 연결 성공! ✅")

            # chunk 단위로 전송 (프론트 마이크 스트리밍 시뮬레이션)
            total_chunks = (len(raw_data) + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(f">>> 총 {total_chunks}개 청크로 분할 전송 (chunk size: {CHUNK_SIZE} bytes)")
            print()

            for i in range(0, len(raw_data), CHUNK_SIZE):
                chunk = raw_data[i:i + CHUNK_SIZE]
                await ws.send(chunk)

                chunk_num = i // CHUNK_SIZE + 1
                progress = min(100, int(chunk_num / total_chunks * 100))
                print(f">>> 청크 전송 중... [{chunk_num}/{total_chunks}] ({progress}%) ({len(chunk)} bytes)")

            # 종료 신호
            print("\n>>> 'END' 신호 전송...")
            await ws.send("END")

            # 결과 수신
            print(">>> 서버 응답 대기 중...")
            response = await ws.recv()

            result = json.loads(response)
            pretty_print_result(result)

            # raw JSON도 출력 (디버그용)
            print("📄 Raw JSON:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"❌ WebSocket 연결 끊김: {e}")
    except ConnectionRefusedError:
        print(f"❌ 서버 연결 거부! uvicorn이 실행 중인지 확인하세요.")
        print(f"   → uvicorn main:app --reload")
    except FileNotFoundError:
        print(f"❌ 파일 없음: {filepath}")
        print(f"   → 테스트용 wav 파일을 프로젝트 루트에 넣어주세요")


if __name__ == "__main__":
    # 커맨드라인 인자로 wav 파일 경로 받기
    wav_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

    print()
    print("🎤 Voice AI Server V2.0 - 로컬 테스트")
    print(f"   파일: {wav_file}")
    print(f"   서버: {WS_URL}")
    print()

    asyncio.run(test_websocket(wav_file))
