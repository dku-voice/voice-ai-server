# voice-ai-server

DKU Capstone 음성 주문 Kiosk용 AI Server입니다.
FastAPI 기반으로 음성 WebSocket 처리, STT, LLM 주문 파싱, DeepFace 스냅샷 연령 추정을 담당합니다.

## 개발 환경

이 프로젝트의 기준 Python 버전은 **Python 3.11**입니다.

DeepFace가 TensorFlow를 사용하기 때문에 Python 3.14 환경에서는 의존성 설치가 실패할 수 있습니다.
Windows 로컬 개발에서는 Python 3.11로 만든 `venv311` 환경을 사용합니다.

```powershell
E:\Python3.11\python.exe -m venv venv311
.\venv311\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

기존 Python 3.14 환경을 확인해야 할 때는 `venv`를 사용할 수 있습니다.

```powershell
.\venv\Scripts\activate
python --version
```

DeepFace age 모델은 최초 실행 시 사용자 홈 디렉터리 아래에 weight 파일을 내려받습니다.

```text
C:\Users\<사용자>\.deepface\weights
```

기본값은 서버 시작 시 DeepFace를 미리 로딩하지 않는 lazy loading입니다.
첫 vision 요청 지연을 줄여야 할 때만 `.env`에 아래 값을 `true`로 바꿔서 사용합니다.

```text
VISION_WARMUP_ON_STARTUP=false
```

## 실행

```powershell
.\venv311\Scripts\activate
python -m uvicorn main:app --reload
```

기본 엔드포인트:

- `GET /`: 서버 상태 확인
- `WebSocket /ws/audio`: 음성 주문 처리
- `POST /api/vision/analyze_age`: 스냅샷 이미지 연령 추정
- `POST /api/voice`: v1.0 legacy API, deprecated

## 테스트

```powershell
.\venv311\Scripts\python.exe -m unittest discover -s tests
.\venv311\Scripts\python.exe -m compileall app tests
```

DeepFace 실제 이미지 테스트는 루트의 `test.jpg`를 사용할 수 있습니다.
`test.jpg`는 로컬 테스트 파일이므로 Git에 올리지 않습니다.

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\venv311\Scripts\python.exe -c "import json; from pathlib import Path; from app.services.vision_service import _run_deepface_age; print(json.dumps(_run_deepface_age(Path('test.jpg').read_bytes()), ensure_ascii=False))"
```

## Docker

Dockerfile은 팀원 재현과 배포 준비를 위한 용도입니다.
현재 로컬 개발은 `venv311`을 기준으로 진행하고, Docker는 통합 테스트나 배포 준비 단계에서 사용합니다.

```powershell
docker build -t voice-ai-server .
docker run --env-file .env -p 8000:8000 voice-ai-server
```

DeepFace와 STT 모델 weight는 컨테이너 최초 실행 시 다운로드될 수 있습니다.
