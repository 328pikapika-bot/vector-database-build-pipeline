"""
MongoDB 状态记录模型

两个集合：
- doc_sync_status: 待发布记录表（按 docId 跟踪处理状态）
- doc_sync_failures: 失败记录表（用于补偿重试）
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional, Dict, Any


class SyncStatus(BaseModel):
    """文献同步状态记录（对应 MongoDB doc_sync_status 集合）"""

    docId: str = Field(..., description="文献 ID，唯一标识")
    subDocType: Optional[str] = Field(default=None, description="文献类型")
    articleId: Optional[str] = Field(default=None, description="原始数据 ID")
    dataStatus: Literal["ADD", "UPDATE", "DELETE"] = Field(
        ..., description="事件操作类型"
    )
    status: Literal["PROCESSING", "SUCCESS", "FAILURE"] = Field(
        default="PROCESSING", description="处理状态"
    )
    recordUpdateTime: datetime = Field(..., description="最新的 recordUpdateTime")
    recordCreateTime: Optional[datetime] = Field(default=None, description="记录创建时间")
    chunkCount: int = Field(default=0, description="写入 Milvus 的 chunk 数量")
    errorMsg: Optional[str] = Field(default=None, description="失败时的错误信息")
    updatedAt: datetime = Field(
        default_factory=datetime.utcnow, description="状态最后更新时间"
    )

    class Config:
        # 允许兼容 MongoDB ObjectId
        arbitrary_types_allowed = True


class FailureRecord(BaseModel):
    """失败记录（对应 MongoDB doc_sync_failures 集合）"""

    doc_id: str = Field(..., description="文献 ID")
    type: Literal["ADD", "UPDATE", "DELETE"] = Field(..., description="操作类型")
    data: Dict[str, Any] = Field(..., description="完整原始事件 JSON，用于补偿重试")
    status: int = Field(
        default=0,
        description="处理状态: 0=FAILED, 1=RETRYING, 2=DONE"
    )
    error_type: Literal["RETRYABLE", "NON_RETRYABLE"] = Field(
        ..., description="错误类型"
    )
    msg: str = Field(..., description="失败异常信息")
    retry_count: int = Field(default=0, description="已重试次数")
    create_time: datetime = Field(
        default_factory=datetime.utcnow, description="创建时间"
    )
    update_time: datetime = Field(
        default_factory=datetime.utcnow, description="更新时间"
    )

    class Config:
        arbitrary_types_allowed = True
