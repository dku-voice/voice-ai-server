from fastapi import FastAPI, UploadFile, File
import whisper
import os
import uuid

app = FastAPI(title="Voice AI Server")

# Whisper 모델 로드 (CPU 환경 고려 base 모델 사용)
model = whisper.load_model("base")

@app.post("/api/voice")
async def process_voice(file: UploadFile = File(...)):
    # 프론트엔드에서 받은 오디오 파일을 임시 저장
    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    try:
        # Whisper 모델을 통한 음성 인식 (STT)
        result = model.transcribe(temp_filename, language="ko")
        text = result.get("text", "").strip()
        
        # 텍스트에서 주문 메뉴 및 수량 추출 (1학기: 키워드 기반 매칭)
        menus_detected = []
        
        number_map = {"하나": 1, "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
        menu_keywords = ["햄버거", "콜라", "감자튀김", "피자", "치킨"]
        
        for menu in menu_keywords:
            if menu in text:
                quantity = 1 # 기본 수량
                for kor_num, int_num in number_map.items():
                    # 메뉴명 뒤에 숫자가 오는지 패턴 확인
                    if f"{menu} {kor_num}" in text or f"{menu}{kor_num}" in text:
                        quantity = int_num
                        break
                
                menus_detected.append({
                    "menu": menu,
                    "quantity": quantity
                })

        # 백엔드 API 규격에 맞춘 JSON 반환
        return {
            "status": "success",
            "original_text": text,
            "extracted_data": menus_detected
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # 임시 파일 삭제 (디스크 용량 확보)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/")
def root():
    return {"message": "AI Server is running"}
