"""
app/schemas.py - API 요청/응답 스키마 정의
아키텍처팀이 정한 계약서 (절대 필드명 변경 금지!!)
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class OrderItem(BaseModel):
    """주문 항목 하나 (메뉴 + 수량 + 옵션)"""
    menu_id: str = Field(..., description="菜单项ID，例如 'burger_01'")
    quantity: int = Field(..., description="数量，例如 2")
    options: Optional[List[str]] = Field(default=[], description="顾客的附加要求")


class VoiceOrderResponse(BaseModel):
    """음성 주문 처리 결과 - 프론트엔드한테 보내는 최종 응답"""
    status: str = Field(..., description="处理状态：'success' | 'fallback' | 'error'")
    recognized_text: str = Field(..., description="STT识别出的原始顾客语音文本")
    items: List[OrderItem] = Field(default=[], description="解析出的点餐数组")
    error_msg: Optional[str] = Field(default=None, description="错误信息")
