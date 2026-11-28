# voice-ai-server

단국대학교 컴퓨터공학과 캡스톤디자인 **보이스(VOICE)** 프로젝트의 AI 서버입니다.
키오스크에서 받은 음성을 텍스트로 바꾸고, 메뉴·수량·옵션을 뽑아 주문 JSON으로 돌려줍니다.
메뉴 추천과 카메라 연령 추정도 이 서버에서 처리합니다.

- 서비스 주소: https://dku-voice.com (2026-11-27 캡스톤 페스티벌 시연 완료)
- 지원 언어: 한국어, 영어, 일본어, 스페인어, 태국어
- 테스트: 219개 통과

## 처리 흐름

```text
키오스크 마이크 (16kHz mono PCM)
  │  WebSocket /ws/audio, 끝나면 "END:<화면 언어>"
  ▼
Silero VAD ── 말소리 구간만 남김
  ▼
faster-whisper (base, CPU int8) ── 텍스트 + 발화 언어 감지
  ▼
주문 파싱
  ├─ 한국어: 키워드 매칭 우선, 문맥이 필요한 말이면 Gemini 재확인
  └─ 외국어: Gemini 우선, 실패하거나 늦으면 키워드 매칭
  ▼
검증 ── 메뉴판에 있는 메뉴인지, 메뉴가 지원하는 옵션인지, (외국어) 발음 근거가 있는지
  ▼
JSON 응답 → 프론트가 장바구니에 반영
```

추천 요청("안 매운 버거 추천해줘")이면 주문 대신 추천 결과를 돌려줍니다.

## 주요 기능

### 음성 인식 (STT)

- faster-whisper `base` 모델을 CPU `int8`로 씁니다. 서버가 vCPU 2개, 메모리 3.7GB라 더 큰 모델은 응답이 너무 느렸습니다.
- 메뉴 이름, 수량 표현("두 잔", "세 개"), 자주 쓰는 말을 STT 힌트(hotword)로 넣어 인식률을 올립니다.
- 힌트는 언어별 A/B 테스트로 켤 언어를 정했습니다 (`STT_HOTWORD_LANGUAGES`).
  - 한국어·영어·일본어·스페인어: 켜면 정확도가 올라가서 켬
  - 태국어: 켜면 없는 메뉴 이름을 만들어내고 STT가 수십 초씩 돌아서 끔
- 서버 시작 때 VAD와 STT를 미리 한 번 돌려서 첫 주문이 느려지지 않게 합니다.

### 언어 감지

- 프론트가 녹음 종료 때 화면 언어를 같이 보냅니다 (`END:ko`, `END:en` 등).
- Whisper가 감지한 언어의 확신도가 기준보다 높을 때만 그 언어를 쓰고, 낮으면 화면 언어를 따릅니다.
  - 외국어 화면에서 다른 언어로 바꾸는 기준: `LANGUAGE_HINT_MIN_PROBABILITY` (기본 0.8)
  - 한국어 화면에서 다른 언어로 바꾸는 기준: `LANGUAGE_MIN_PROBABILITY` (기본 0.5). 외국인이 언어를 바꾸지 않고 말해도 알아듣게 낮게 둡니다.

### 주문 파싱

| 언어 | 1순위 | 2순위 | 이유 |
| --- | --- | --- | --- |
| 한국어 | 키워드 매칭 (약 1ms) | Gemini 재확인 | 메뉴가 정해져 있고 발음 변형까지 직접 다룰 수 있어 키워드가 더 빠르고 정확함 |
| 영어·일본어·스페인어·태국어 | Gemini | 키워드 매칭 | 표현이 다양해서 문맥 이해가 필요함 |

- 한국어는 "취소", "바꿔", "말고", "하나만", "아까" 같은 말이 들어 있을 때만 Gemini가 다시 판단합니다 (`LLM_SKIP_KOREAN`).
- 키워드 매칭이 다루는 것
  - 메뉴 별칭과 STT 오인식: "감튀", "제로콜라", "코랄"(콜라), "모짜렐링"(오징어링) 등
  - 수량: "두 개", "세 잔", "2개", 오인식("새잔"→3), 영어 "two/to"(메뉴 바로 앞일 때만)
  - 수정 표현: "콜라 말고 사이다", "콜라 두 잔 말고 세 잔"
- Gemini 결과는 그대로 쓰지 않고 검증합니다.
  - 메뉴판에 없는 메뉴 ID, 메뉴가 지원하지 않는 옵션은 버립니다.
  - 외국어는 받아 적은 문장에 그 메뉴의 발음 근거가 있는지 확인합니다 (환각 방지).
  - Gemini가 빈 결과를 주면 키워드 결과를 씁니다.
- 전체 응답 시간이 `VOICE_RESPONSE_DEADLINE_SECONDS`(기본 11초)를 넘기면 그때까지의 키워드 결과로 응답합니다.

### 옵션 인식

"재료 + 추가/빼기/적게 표현"을 조합해서 인식합니다. 5개 언어 모두 같은 방식입니다.

| 종류 | 한국어 예 | 외국어 예 |
| --- | --- | --- |
| 추가 | 치즈 추가, 베이컨 듬뿍, 치즈 추기(오인식) | extra cheese, con queso extra, チーズ追加, เพิ่มชีส |
| 빼기 | 양파 빼고, 토마토 넣지 마세요, 노 어니언 | no onions, sin tomate, 玉ねぎ抜き, ไม่ใส่หอม |
| 적게 | 소스 적게, 소스 살살, 소스 젖어(오인식) | light on the sauce, poca salsa, ソース少なめ, ซอสน้อย |

- 여러 재료를 묶어 말해도 모두 적용합니다: "치즈랑 토마토 추가", "add cheese and bacon", "sin cebolla ni tomate"
- STT가 넣은 쉼표("양파, 제거")나 "랑"을 "란"으로 적은 경우도 처리합니다.
- 양념감자 소스(어니언·치즈·칠리)는 "칠리로", "치즈맛", "cheese sauce", "チリソース"처럼 짧게 말해도 인식합니다.
- "치즈스틱", "cheese sticks"처럼 메뉴 이름과 겹치는 단어는 옵션으로 보지 않습니다.

### 메뉴 추천 (RAG)

- 메뉴마다 특징(매콤함, 달콤함 등), 영양 정보, 가격을 담은 문서를 만들고, 손님 조건에 맞는 문서를 검색해 Gemini가 이유와 함께 추천합니다.
- Gemini가 늦거나 실패하면 규칙 기반 추천으로 대신합니다 (`VOICE_RECOMMENDATION_BUDGET_SECONDS`).
- 부정 표현을 반대로 처리합니다: "안 매운", "매운 거 말고", "짜지 않은", "not spicy", "辛くない", "sin picante", "ไม่เผ็ด"
- Gemini가 손님이 원하지 않은 특징의 메뉴를 고르면 결과에서 뺍니다.

### 메뉴 동기화

- 서버 시작 때 백엔드(`MENU_SYNC_URL`)에서 메뉴, 다국어 이름, 메뉴별 옵션을 받아옵니다. 백엔드가 늦게 뜨면 재시도합니다.
- 메뉴를 바꾼 뒤에는 `POST /api/menus/resync`로 다시 받아올 수 있습니다.
- 발음 별칭은 `app/services/llm_service.py`의 별칭 사전에 있습니다.

### 연령 추정

- 키오스크 시작 화면 스냅샷으로 DeepFace 나이 추정을 돌리고, `SENIOR_AGE_THRESHOLD`(기본 60세) 이상이면 시니어(터치) 모드를 권합니다.
- 업로드 이미지는 저장하지 않고 요청 안에서만 씁니다.

## API

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| GET | `/` | 서버 상태, 모델 설정 |
| WebSocket | `/ws/audio` | 음성 주문, 음성 추천 |
| POST | `/api/recommendations/menu` | 문장 기반 메뉴 추천 |
| POST | `/api/vision/analyze_age` | 스냅샷 연령 추정 (multipart, field `file`) |
| GET | `/api/menus` | 현재 메뉴 카탈로그 요약 |
| POST | `/api/menus/resync` | 백엔드 메뉴 다시 받기 |
| POST | `/api/voice` | 예전 HTTP 방식 (새 연동은 `/ws/audio`) |

### WebSocket `/ws/audio`

```text
보내는 것: raw PCM binary chunk (16kHz, mono, 16-bit signed, little-endian)
녹음 종료: 텍스트 "END" 또는 "END:<언어>" (ko | en | ja | es | th)
```

webm, mp3, WAV 컨테이너는 받지 않습니다. 프론트에서 raw PCM으로 바꿔 보냅니다.

주문 응답:

```json
{
  "status": "success",
  "recognized_text": "데리버거 하나 양파 빼고 콜라 두 잔",
  "items": [
    {"menu_id": "1", "quantity": 1, "options": ["양파 제거"]},
    {"menu_id": "14", "quantity": 2, "options": []}
  ],
  "error_msg": null,
  "language": "ko"
}
```

- `status`: `success`, `fallback`(Gemini 실패로 키워드 결과), `error`
- `menu_id`는 백엔드 DB의 메뉴 ID, `options`는 백엔드 옵션의 한국어 이름입니다.

추천 응답:

```json
{
  "type": "recommendation",
  "recognized_text": "안 매운 버거 추천해줘",
  "language": "ko",
  "source": "llm",
  "items": [{"menu_id": "1", "name": "데리버거", "reason": "달콤한 데리야끼 소스로 맵지 않습니다."}]
}
```

### POST `/api/recommendations/menu`

```json
{"query": "사이드랑 음료 추천해줘", "top_k": 3, "language": "ko"}
```

## 환경변수

`.env`에 두고 GitHub에는 올리지 않습니다. 주요 값만 적었고, 전체 목록과 기본값은 `app/config.py`에 있습니다.

```env
GEMINI_API_KEY=
LLM_MODEL=gemini-3.1-flash-lite
LLM_TIMEOUT_SECONDS=6
LLM_SKIP_KOREAN=true

WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
STT_HOTWORD_LANGUAGES=ko,en,ja,es
LANGUAGE_MIN_PROBABILITY=0.5
LANGUAGE_HINT_MIN_PROBABILITY=0.8
VAD_THRESHOLD=0.6

VOICE_RESPONSE_DEADLINE_SECONDS=11
VOICE_RECOMMENDATION_BUDGET_SECONDS=6.5

MENU_SYNC_ENABLED=true
MENU_SYNC_URL=http://backend:8080/api/ai/menus

SENIOR_AGE_THRESHOLD=60
VISION_WARMUP_ON_STARTUP=false
```

- `GEMINI_API_KEY`가 있으면 Gemini(OpenAI 호환 방식), 없으면 `OPENAI_API_KEY`로 OpenAI를 씁니다.
- 둘 다 없으면 LLM 없이 키워드 매칭과 규칙 기반 추천만으로 동작합니다.

## 로컬 실행

Python 3.11 기준입니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

CPU 전용 환경은 `requirements-linux-cpu.txt`를 씁니다.

WebSocket 테스트 (16kHz mono 16-bit WAV):

```bash
python local_ws_test.py test.wav
ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 test.wav
```

## 테스트

```bash
python -m pytest -q tests
```

| 파일 | 다루는 것 |
| --- | --- |
| `test_llm_service.py` | 한국어 키워드 매칭, 수량, 수정 표현, Gemini 재확인 |
| `test_foreign_fallback.py` | 외국어 키워드 매칭, 발음 근거 검증, 외국어 옵션 |
| `test_menu_options.py` | 옵션 조합, 오인식, 양념감자 소스 |
| `test_recommendation_service.py` | RAG 추천, 부정 표현, 영양·가격 조건 |
| `test_stt_service.py` | 언어 선택, 힌트 구성 |
| `test_websockets.py` | END 신호, 응답 형식, 주문·추천 분기, 오류 처리 |
| `test_menu_sync.py` | 백엔드 메뉴 동기화 |

## 배포

- AWS EC2 한 대에서 Docker Compose로 운영했습니다. 서비스 이름은 `ai-server`, 컨테이너 이름은 `voice-ai`입니다.
- Nginx가 `/ws/`를 이 서버로 넘깁니다. Nginx는 시작할 때 컨테이너 주소를 정하므로, AI 서버를 다시 띄운 뒤에는 `docker restart voice-gateway`가 필요합니다.
- main에 머지하면 GitHub Actions(`Deploy to EC2`)가 이 컨테이너만 다시 빌드합니다.
- 운영하면서 쓴 명령어:

```bash
docker logs --since 10m voice-ai 2>&1 | grep -E "STT 완료|키워드 매칭 결과|Phase 1|응답 전송"
docker exec voice-ai env | grep -E "STT_HOTWORD|LLM_MODEL"
```

로그에는 단계별 소요 시간(VAD, STT, LLM)이 남아서 어느 단계가 느린지 바로 볼 수 있습니다.

## 알고 있는 한계

- STT가 CPU를 많이 써서 여러 키오스크가 동시에 쓰면 느려집니다 (혼자 약 3.5초, 동시 사용 시 6초 안팎).
- 긴 문장에 메뉴를 여러 개 말하면 base 모델이 가끔 잘못 듣습니다.
- 주변 목소리가 섞이면 정확도가 떨어집니다. VAD는 무음만 자릅니다.
- 언어별 인식 정확도를 수치로 측정하지는 못했습니다. 로그에서 틀린 문장을 찾아 고치고 테스트로 고정하는 방식으로 개선했습니다.
- 발음 별칭은 코드에 있어서, 바꾸려면 배포가 필요합니다.

## 개선 방향

- GPU 서버나 STT 전용 서버를 두고 더 큰 Whisper 모델 사용
- 언어별 테스트 문장 세트로 인식 정확도 측정
- 발음 별칭을 백엔드 DB에서 관리
- 잡음 제거, 지향성 마이크 적용

## 팀

| 이름 | 역할 |
| --- | --- |
| 김진우 | PM, 인프라, AI 서버 (원담당자 조기졸업 후 인수) |
| 홍정안 | 백엔드 (대표) |
| 정현웅 | 프론트엔드 |
