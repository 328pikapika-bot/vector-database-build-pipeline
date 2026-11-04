"""
文献完整 JSON 模型
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from .base.article import Article

class Document(BaseModel):
    """文献完整数据模型 - 匹配 Kafka 外层消息体结构"""

    success: Optional[bool] = Field(default=True, description="是否成功标识")
    subDocType: str = Field(..., description="文献类型")
    dataType: Optional[str] = Field(default=None, description="数据类型")
    data: List[Article] = Field(default_factory=list, description="文献对象数组")
