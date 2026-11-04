"""Full-text chunk model aligned with the DOCX acceptance contract."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A full-text chunk ready for embedding and Milvus persistence."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(
        ...,
        validation_alias=AliasChoices("id", "child_chunk_id", "chunkId"),
        description="chunk primary key",
    )
    doc_id: str = Field(
        ...,
        validation_alias=AliasChoices("doc_id", "docId"),
        description="document id",
    )
    title: Optional[List[str]] = Field(default=None, description="document title")
    keyword: Optional[List[str]] = Field(default=None, description="document keywords")
    accessType: Optional[str] = Field(default=None, description="access source/type")
    classificationCode: Optional[str] = Field(default=None, description="classification code")
    year: Optional[str] = Field(default=None, description="publication year")
    hasAuthorname: Optional[List[str]] = Field(default=None, description="authors")
    hasCreateOrganizationname: Optional[List[str]] = Field(default=None, description="organizations")
    language: Optional[List[str]] = Field(default=None, description="language")

    chunkIndex: int = Field(..., description="chunk index from 0")
    chunkType: Literal["title_abstract", "section", "paragraph", "table", "reference"] = Field(
        ...,
        description="chunk type",
    )
    text: str = Field(..., description="original chunk text")
    charStart: int = Field(..., description="source start character offset")
    charEnd: int = Field(..., description="source end character offset")
    sectionPath: Optional[str] = Field(
        default=None,
        description="section hierarchy path",
    )
    tokenCount: Optional[int] = Field(default=None, description="approximate token count")
    text_embed: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("text_embed", "embedding"),
        description="chunk text embedding",
    )
    dynamic_fields: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def chunkId(self) -> str:
        return self.id

    @chunkId.setter
    def chunkId(self, value: str) -> None:
        self.id = value

    @property
    def child_chunk_id(self) -> str:
        """Compatibility alias; V2 persists this value as the `id` field."""
        return self.id

    @child_chunk_id.setter
    def child_chunk_id(self, value: str) -> None:
        self.id = value

    @property
    def docId(self) -> str:
        return self.doc_id

    @docId.setter
    def docId(self, value: str) -> None:
        self.doc_id = value

    @property
    def embedding(self) -> Optional[List[float]]:
        return self.text_embed

    @embedding.setter
    def embedding(self, value: Optional[List[float]]) -> None:
        self.text_embed = value
