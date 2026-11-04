from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional


class RawKafkaMessage(BaseModel):
    """Kafka 原始消息体。

    当前阶段仅做基本的 JSON 解析，不强约束业务字段，
    主要由下游 parser 做深度的结构化处理。
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: Optional[bool] = None
    subDocType: Optional[str] = None
    dataType: Optional[str] = None
    data: List[Dict[str, Any]] = Field(default_factory=list)
