"""
app/schemas.py - API 요청/응답 스키마 정의
프론트/백엔드랑 맞춘 값이라 필드명은 마음대로 바꾸지 않는다.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class OrderItem(BaseModel):
    """주문 항목 하나 (메뉴 + 수량 + 옵션)"""
    menu_id: str = Field(..., description="메뉴 ID (예: 'burger_01')")
    quantity: int = Field(..., description="수량 (예: 2)")
    options: Optional[List[str]] = Field(default=[], description="추가 요청 사항")


class VoiceOrderResponse(BaseModel):
    """음성 주문 처리 후 프론트로 보내는 응답"""
    status: str = Field(..., description="'success' | 'fallback' | 'error'")
    recognized_text: str = Field(..., description="STT 인식 텍스트")
    items: List[OrderItem] = Field(default=[], description="주문 항목")
    error_msg: Optional[str] = Field(default=None, description="에러 메시지")


class AgeAnalysisResponse(BaseModel):
    """스냅샷 이미지 기반 연령 추정 결과"""
    status: str = Field(..., description="'success' | 'error'")
    estimated_age: Optional[int] = Field(default=None, description="DeepFace가 추정한 나이")
    age_group: str = Field(default="unknown", description="'child' | 'adult' | 'senior' | 'unknown'")
    is_senior: bool = Field(default=False, description="고령자 UI로 바꿀지 여부")
    error_msg: Optional[str] = Field(default=None, description="에러 메시지")
