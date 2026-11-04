"""
Kafka 事件体模型
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional


class KafkaEvent(BaseModel):
    """Kafka 消息体 - 发布/消费的事件"""

    id: str = Field(..., description="事件唯一 ID")
    dataStatus: Literal["ADD", "UPDATE", "DELETE"] = Field(
        ..., description="操作类型：新增 / 更新 / 删除"
    )
    docId: str = Field(..., description="文献唯一标识")
    recordUpdateTime: datetime = Field(
        ..., description="业务更新时间（幂等判定关键字段）"
    )
    subDocType: Optional[str] = Field(default=None, description="文献类型")
    articleId: Optional[str] = Field(default=None, description="原始数据 ID")
