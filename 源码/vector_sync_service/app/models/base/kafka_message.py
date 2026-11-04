from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from .article import Article

class KafkaMessage(BaseModel):
    """Kafka 接收的顶层数据结构 (包装预处理后的 Article)"""
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    
    success: Optional[bool] = None
    subDocType: Optional[str] = None
    dataType: Optional[str] = None
    data: List[Article] = Field(default_factory=list, description="包含的文章/文献集合")
