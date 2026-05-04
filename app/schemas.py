"""
app/schemas.py - API 요청/응답 스키마 정의
아키텍처팀이 정한 계약서 (절대 필드명 변경 금지!!)
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class OrderItem(BaseModel):
    """주문 항목 하나 (메뉴 + 수량 + 옵션)"""
    menu_id: str = Field(..., description="메뉴 항목 ID (예: 'burger_01')")
    quantity: int = Field(..., description="주문 수량 (예: 2)")
    options: Optional[List[str]] = Field(default=[], description="고객의 추가 요청 사항 (옵션)")

class VoiceOrderResponse(BaseModel):
    """음성 주문 처리 결과 - 프론트엔드한테 보내는 최종 응답"""
    status: str = Field(..., description="처리 상태: 'success' | 'fallback' | 'error'")
    recognized_text: str = Field(..., description="STT가 인식한 원본 고객 음성 텍스트")
    items: List[OrderItem] = Field(default=[], description="파싱된 최종 주문 항목 리스트")
    error_msg: Optional[str] = Field(default=None, description="에러 발생 시 상세 메시지")
    