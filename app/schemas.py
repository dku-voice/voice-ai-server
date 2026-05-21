"""
app/schemas.py - API 요청/응답 스키마 정의
프론트/백엔드랑 맞춘 값이라 필드명은 마음대로 바꾸지 않는다.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


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


class MenuRecommendationRequest(BaseModel):
    """메뉴 추천 요청"""
    query: str = Field(..., min_length=1, max_length=500, description="고객 요청 문장")
    top_k: int = Field(default=3, ge=1, le=5, description="추천 후보 개수")


class MenuRecommendationItem(BaseModel):
    """추천 메뉴 하나"""
    menu_id: str = Field(..., description="메뉴 ID")
    name: str = Field(..., description="메뉴 이름")
    reason: str = Field(..., description="추천 이유")


class MenuRecommendationResponse(BaseModel):
    """RAG 메뉴 추천 응답"""
    status: str = Field(..., description="'success' | 'fallback' | 'error'")
    query: str = Field(..., description="추천에 사용한 고객 요청")
    recommendations: List[MenuRecommendationItem] = Field(default=[], description="추천 메뉴 목록")
    retrieved_menu_ids: List[str] = Field(default=[], description="검색된 메뉴 문서 ID")
    error_msg: Optional[str] = Field(default=None, description="에러 메시지")
