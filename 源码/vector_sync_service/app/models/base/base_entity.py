from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class BaseEntity(BaseModel):
    """所有基础数据实体的基类"""
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    
    id: Optional[str] = Field(default=None, description="唯一标识")
    cls: Optional[str] = Field(default=None, description="对象类型")
