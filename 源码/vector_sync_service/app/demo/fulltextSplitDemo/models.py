"""
全文切分 Demo - 数据模型
定义三级切分的 Chunk 数据结构
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


class ChunkType(str, Enum):
    """子块类型枚举"""
    SECTION_HEADER = "section_header"   # 第一级: 章节标题块
    PARAGRAPH = "paragraph"             # 第二级: 段落块
    FIXED_CHUNK = "fixed_chunk"         # 第三级: 固定长度切分块


@dataclass
class FullTextFileInfo:
    """从 hasFullTextFile 解析出的全文文件信息"""
    fileId: str = ""                    # 文件内部 ID
    fileName: str = ""                  # 文件名 (如 xxx.mmd)
    url: str = ""                       # 相对路径 URL
    fileType: str = "mmd"              # 文件类型
    fileSize: int = 0                   # 文件大小(字节)
    sysDataType: str = "file"          # 系统数据类型


@dataclass
class RedundantMeta:
    """冗余元数据 — 附加到每个 Chunk 的信息"""
    sysId: str = ""                     # 文献ID
    title: Optional[str] = None         # 文献标题
    authors: List[str] = field(default_factory=list)   # 作者列表
    year: Optional[str] = None          # 年份
    journal: Optional[str] = None       # 期刊名
    doi: Optional[str] = None           # DOI
    keywords: List[str] = field(default_factory=list)  # 关键词
    sourceFileName: str = ""            # 来源全文文件名


@dataclass
class FullTextChunk:
    """三级切分后的子块数据模型"""
    chunkId: str = ""                   # 子块唯一ID: {sysId}_{chunkIndex}
    sysId: str = ""                     # 所属文献ID
    chunkIndex: int = 0                 # 块序号 (从0自增)
    chunkType: ChunkType = ChunkType.PARAGRAPH

    # 冗余元数据
    redundantMeta: RedundantMeta = field(default_factory=RedundantMeta)

    # 子块内容
    content: str = ""                   # 子块文本内容
    contentVector: List[float] = field(default_factory=list)  # 内容向量化

    # 位置信息
    paraStartIdx: int = 0               # 段落起始字符偏移
    paraEndIdx: int = 0                 # 段落结束字符偏移
    charStart: int = 0                  # 在原文中的起始字符偏移
    charEnd: int = 0                    # 在原文中的结束字符偏移

    # 结构信息
    sectionPath: str = ""               # 章节层级路径
    level: int = 0                      # 章节层级深度
    parentParaId: str = ""              # 上级段落ID (仅 fixed_chunk 使用)
    tokenCount: int = 0                 # 近似 token 数

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的字典"""
        result = {
            "chunkId": self.chunkId,
            "sysId": self.sysId,
            "chunkIndex": self.chunkIndex,
            "chunkType": self.chunkType.value if isinstance(self.chunkType, ChunkType) else self.chunkType,
            "redundantMeta": {
                "sysId": self.redundantMeta.sysId,
                "title": self.redundantMeta.title,
                "authors": self.redundantMeta.authors,
                "year": self.redundantMeta.year,
                "journal": self.redundantMeta.journal,
                "doi": self.redundantMeta.doi,
                "keywords": self.redundantMeta.keywords,
                "sourceFileName": self.redundantMeta.sourceFileName,
            },
            "content": self.content,
            "contentVector": self.contentVector[:10] if len(self.contentVector) > 10 else self.contentVector,  # 摘要显示
            "contentVectorDim": len(self.contentVector),
            "paraStartIdx": self.paraStartIdx,
            "paraEndIdx": self.paraEndIdx,
            "charStart": self.charStart,
            "charEnd": self.charEnd,
            "sectionPath": self.sectionPath,
            "level": self.level,
            "parentParaId": self.parentParaId,
            "tokenCount": self.tokenCount,
        }
        return result


@dataclass
class SplitResult:
    """单篇文献的切分结果"""
    sysId: str = ""
    success: bool = False
    totalChunks: int = 0
    chunksByType: Dict[str, int] = field(default_factory=dict)
    chunks: List[FullTextChunk] = field(default_factory=list)
    errorMessage: str = ""
    sourceFile: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sysId": self.sysId,
            "success": self.success,
            "totalChunks": self.totalChunks,
            "chunksByType": self.chunksByType,
            "chunks": [c.to_dict() for c in self.chunks],
            "errorMessage": self.errorMessage,
            "sourceFile": self.sourceFile,
        }
