from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, Field

from .base_entity import BaseEntity


class Article(BaseEntity):
    """Minimal literature/article entity used by the vector sync pipeline."""

    ID: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ID", "sysId", "docId", "doc_id", "id"),
        description="document id, canonical old sysId",
    )
    title: Optional[List[str]] = Field(default=None, description="title")
    abstract: Optional[List[str]] = Field(default=None, description="abstract")
    year: Optional[str] = Field(default=None, description="year")
    sysSubType: Optional[str] = Field(default=None, description="document subtype")
    keyword: Optional[List[str]] = Field(default=None, description="keywords")

    title_embed: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("title_embed", "title_vector"),
        description="title embedding",
    )
    abstract_embed: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("abstract_embed", "abstract_vector"),
        description="abstract embedding",
    )
    keyword_embed: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("keyword_embed", "keyword_vector"),
        description="keyword embedding",
    )

    dynamic_fields: Dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="legacy in-memory metadata carrier; not written to Milvus",
    )

    @property
    def sysId(self) -> Optional[str]:
        return self.ID

    @sysId.setter
    def sysId(self, value: Optional[str]) -> None:
        self.ID = value

    @property
    def docId(self) -> Optional[str]:
        return self.ID

    @docId.setter
    def docId(self, value: Optional[str]) -> None:
        self.ID = value

    @property
    def title_vector(self) -> Optional[List[float]]:
        return self.title_embed

    @title_vector.setter
    def title_vector(self, value: Optional[List[float]]) -> None:
        self.title_embed = value

    @property
    def abstract_vector(self) -> Optional[List[float]]:
        return self.abstract_embed

    @abstract_vector.setter
    def abstract_vector(self, value: Optional[List[float]]) -> None:
        self.abstract_embed = value

    @property
    def keyword_vector(self) -> Optional[List[float]]:
        return self.keyword_embed

    @keyword_vector.setter
    def keyword_vector(self, value: Optional[List[float]]) -> None:
        self.keyword_embed = value
