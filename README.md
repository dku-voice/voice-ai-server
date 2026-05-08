# voice-ai-server

DKU Capstone 음성 주문 Kiosk 프로젝트에서 사용하는 Python/FastAPI AI 서버입니다.
프론트엔드에서 받은 음성과 스냅샷 이미지를 처리해서 주문 JSON과 연령대 결과를 돌려줍니다.

## 현재 범위

- `WebSocket /ws/audio`: raw PCM 음성 chunk를 받아 VAD, STT, LLM 주문 파싱을 돌립니다.
- `POST /api/vision/analyze_age`: 키오스크 시작 스냅샷으로 DeepFace age 분석을 돌립니다.
- `POST /api/recommendations/menu`: 메뉴 문서를 검색한 뒤 LLM으로 추천 결과를 만듭니다.
- `POST /api/voice`: 예전 HTTP 방식입니다. 새 연동은 `/ws/audio`를 씁니다.

현재 브랜치는 팀 계획 3, 4주차 범위인 LLM 3단계 검증, DeepFace 스냅샷 연령 추정, RAG 메뉴 추천 기초 파이프라인을 정리한 상태입니다.
RAG는 아직 메뉴 DB/벡터 DB 최적화 전 단계입니다. YOLO, 감정 인식, 바코드 스캔은 다음 주차 작업이라 아직 넣지 않았습니다.

## 기술 스택

```text
Python: 3.11.x
API: FastAPI / Uvicorn
Audio: Silero VAD, faster-whisper
LLM/RAG: OpenAI API, LangChain Core
Vision: DeepFace, TensorFlow, tf-keras, Pillow
Schema: Pydantic
Container: Docker
```

기본 패키지는 `requirements.txt`에 정리했습니다.
CPU-only로 설치해야 할 때만 `requirements-linux-cpu.txt`를 씁니다.

## 프로젝트 구조

```text
voice-ai-server/
├── app/
│   ├── api/
│   │   ├── recommendations.py
│   │   ├── vision.py
│   │   └── websockets.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── recommendation_service.py
│   │   ├── stt_service.py
│   │   ├── vad_service.py
│   │   └── vision_service.py
│   ├── config.py
│   └── schemas.py
├── scripts/
│   └── warmup_models.py
├── tests/
├── Dockerfile
├── local_ws_test.py
├── main.py
├── requirements.txt
└── requirements-linux-cpu.txt
```

## 개발 환경

팀 기준 개발 환경은 WSL Ubuntu / Linux + Python 3.11입니다.
Ubuntu 24.04에서는 기본 `python3`가 3.12일 수 있어서 가상환경을 만들 때 `python3.11`을 직접 지정합니다.

```bash
cd ~/projects/voice-ai-server
python3.11 --version
python3.11 -m venv --clear .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows에서 쓰던 `venv/`, `venv311/`, 기존 `.venv/`는 WSL로 복사하지 않습니다.
WSL/Linux 안에서 새로 만드는 쪽이 안전합니다.

CUDA wheel을 받고 싶지 않은 CPU-only 환경에서만 아래 파일을 씁니다.

```bash
python -m pip install -r requirements-linux-cpu.txt
```

## 환경변수

로컬 설정은 `.env` 파일에 둡니다.
`.env`는 GitHub에 올리지 않습니다.

```env
OPENAI_API_KEY=[REDACTED]
LLM_MODEL=gpt-4o-mini

SERVER_HOST=0.0.0.0
SERVER_PORT=8000

WHISPER_MODEL_SIZE=base
DISABLE_GPU=false
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

VAD_THRESHOLD=0.6
WS_RECEIVE_TIMEOUT_SECONDS=30

VISION_MAX_IMAGE_BYTES=5242880
VISION_MAX_IMAGE_PIXELS=6000000
VISION_WARMUP_ON_STARTUP=false
SENIOR_AGE_THRESHOLD=60
```

`DISABLE_GPU=true`이면 GPU를 막고 STT를 `cpu/int8`로 고정합니다.
`DISABLE_GPU=false`이면 GPU를 막지 않습니다. GPU 서버에서 STT도 GPU로 돌릴 때는 `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`처럼 지정합니다.

## 실행

```bash
source .venv/bin/activate
python -m uvicorn main:app --reload
```

기본으로 열어둔 API:

- `GET /`: 서버 상태와 모델 설정 확인
- `WebSocket /ws/audio`: 음성 주문 처리
- `POST /api/vision/analyze_age`: 스냅샷 연령 추정
- `POST /api/recommendations/menu`: RAG 메뉴 추천
- `POST /api/voice`: 예전 HTTP 방식

## WebSocket 음성 입력 형식

현재 `/ws/audio`는 브라우저 녹음 파일이 아니라 raw PCM만 받는 구조입니다.

```text
전송 방식: WebSocket binary chunk
오디오 형식: raw PCM
샘플레이트: 16kHz
채널: mono
비트 깊이: 16-bit signed integer
엔디언: little-endian
녹음 종료 신호: text message "END"
```

`webm`, `ogg`, `mp3`, `m4a`, WAV container를 그대로 보내면 지금 서버는 제대로 처리하지 못합니다.
프론트에서 raw PCM으로 바꿔 보내거나, 서버에 디코딩 로직을 따로 추가해야 합니다.

응답 예시:

```json
{
  "status": "success",
  "recognized_text": "콜라 두 개 주세요",
  "items": [
    {
      "menu_id": "cola_01",
      "quantity": 2,
      "options": []
    }
  ],
  "error_msg": null
}
```

현재 menu_id 목록:

```text
burger_01: 햄버거
cola_01: 콜라
fries_01: 감자튀김
pizza_01: 피자
chicken_01: 치킨
```

menu_id는 프론트엔드 UI, Spring Boot 백엔드 DB/주문 API와 같이 맞춰야 합니다.

## RAG 메뉴 추천 API

```text
POST /api/recommendations/menu
Content-Type: application/json
```

요청 예시:

```json
{
  "query": "사이드랑 음료 추천해줘",
  "top_k": 3
}
```

응답 예시:

```json
{
  "status": "success",
  "query": "사이드랑 음료 추천해줘",
  "recommendations": [
    {
      "menu_id": "fries_01",
      "name": "감자튀김",
      "reason": "사이드 메뉴 요청과 잘 맞습니다."
    }
  ],
  "retrieved_menu_ids": ["fries_01", "cola_01"],
  "error_msg": null
}
```

현재 RAG는 4주차 기초 파이프라인입니다.
서버 안의 간단한 메뉴 문서를 LangChain Document 형태로 만들고, 키워드 검색으로 후보를 고른 뒤 LLM에 넘깁니다.
OpenAI API 설정이 없거나 LLM 응답이 깨지면 검색된 메뉴 문서 기준 fallback 추천을 돌려줍니다.
메뉴 DB 연동, embedding 생성, 벡터 DB 검색 최적화는 5주차 작업으로 남겨둡니다.

## Vision API

```text
POST /api/vision/analyze_age
Content-Type: multipart/form-data
field name: file
supported image: jpg, jpeg, png, webp
```

응답 예시:

```json
{
  "status": "success",
  "estimated_age": 72,
  "age_group": "senior",
  "is_senior": true,
  "error_msg": null
}
```

업로드 이미지는 따로 저장하지 않고 요청 안에서 바로 씁니다.
DeepFace 모델은 기본적으로 첫 요청 때 로딩합니다.

## 테스트와 검증

빠른 검증:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
python -m compileall app tests scripts main.py local_ws_test.py
python -c "import main; print('IMPORT_MAIN_OK')"
python -m pip check
```

모델 weight 다운로드와 실제 로딩은 필요할 때 따로 확인합니다.

```bash
python scripts/warmup_models.py
python scripts/warmup_models.py --stt-only
python scripts/warmup_models.py --vision-only
python scripts/warmup_models.py --vision-only --vision-image test.jpg
```

`test.jpg`, `test.wav`는 로컬 테스트 파일이며 GitHub에 올리지 않습니다.

## 로컬 WebSocket 테스트

서버를 먼저 켠 뒤 16kHz mono 16-bit WAV 테스트 파일을 보냅니다.

```bash
source .venv/bin/activate
python local_ws_test.py test.wav
```

오디오 파일 변환 예시:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

## Docker

`.venv`는 Docker build context에 포함하지 않습니다.
컨테이너 안에서는 build 단계에서 의존성을 다시 설치합니다.

기본 빌드는 GPU를 막지 않고 `requirements.txt`를 설치합니다.

```bash
docker build -t voice-ai-server .
docker run --env-file .env -p 8000:8000 voice-ai-server
```

CPU-only wheel로 빌드하고 싶을 때만 `DISABLE_GPU=true`를 씁니다.

```bash
docker build --build-arg DISABLE_GPU=true -t voice-ai-server:cpu .
docker run --env-file .env -e DISABLE_GPU=true -p 8000:8000 voice-ai-server:cpu
```

GPU를 쓰려면 `DISABLE_GPU=false`로 두고, 실행 환경에서 GPU를 연결한 뒤 STT device를 `cuda`로 지정합니다.

```bash
docker build --build-arg DISABLE_GPU=false -t voice-ai-server:gpu .
docker run --gpus all --env-file .env \
  -e DISABLE_GPU=false \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -p 8000:8000 \
  voice-ai-server:gpu
```

GPU 실행은 호스트의 NVIDIA driver와 NVIDIA Container Toolkit 설정이 필요합니다.

컨테이너를 처음 실행하면 STT / DeepFace 모델 weight가 다운로드될 수 있습니다.
반복 다운로드가 귀찮으면 배포 환경에서 HuggingFace / DeepFace 캐시를 volume으로 빼면 됩니다.

## GitHub에 올리면 안 되는 파일

GitHub에 올리지 않는 파일:

- `.env`, `.env.*`
- `.venv/`, `venv/`, `venv311/`, `venv312/`
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- `test.wav`, `test.jpg`, 기타 로컬 테스트 미디어
- `.deepface/`, `.keras/`, `.cache/`, `weights/`, 모델 weight 파일
- `AGENTS.md`

PR 전에 아래 명령으로 ignored 상태와 새 소스 파일을 같이 확인합니다.

```bash
git status --short --ignored
git check-ignore -v .env .venv test.wav test.jpg AGENTS.md
git check-ignore -v Dockerfile app/services/llm_service.py tests/test_llm_service.py requirements-linux-cpu.txt || true
```

마지막 명령에서 소스 파일이 나오지 않으면 됩니다.
