"""Image record model with DOCX-aligned persistence fields."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ImageRecord(BaseModel):
    """Image vectorization record.

    `img_id`, `docId`, URL, dimensions, and file metadata remain available for
    internal task identity and download processing. Milvus writers emit only the
    DOCX acceptance fields.
    """

    model_config = ConfigDict(populate_by_name=True)

    img_id: str = Field(
        ...,
        validation_alias=AliasChoices("img_id", "imageId"),
        description="upstream original image id used directly as the Milvus primary key",
    )
    docId: str = Field(
        ...,
        validation_alias=AliasChoices("docId", "doc_id", "sysId", "ID"),
        description="document id",
    )

    title: Any = Field(default=None, description="document title or image caption")
    keyword: Any = Field(default=None, description="document keywords")
    accessType: Optional[str] = Field(default=None)
    classificationCode: Optional[str] = Field(default=None)
    year: Optional[str] = Field(default=None)
    hasAuthorname: Optional[List[str]] = Field(default=None)
    hasCreateOrganizationname: Optional[List[str]] = Field(default=None)
    language: Optional[List[str]] = Field(default=None)
    sysDataStatus: Optional[str] = Field(default=None)
    hasSourcesysSourceId: Optional[List[str]] = Field(default=None)
    hasSourceidSet: Optional[List[str]] = Field(default=None)

    url: str = Field(default="", description="image URL or relative path")
    name: str = Field(default="", description="image file name")
    width: int = Field(default=0, description="image width in pixels")
    height: int = Field(default=0, description="image height in pixels")
    sequence: int = Field(default=0, description="image sequence in document")
    fileType: str = Field(default="", description="image file type")
    size: int = Field(default=0, description="image file size in bytes")

    raw_img_meta: Dict[str, Any] = Field(default_factory=dict, description="raw image object metadata")
    img_meta: Dict[str, Any] = Field(default_factory=dict, description="image bbox/page metadata")
    context_before: Optional[str] = Field(default=None)
    context_after: Optional[str] = Field(default=None)

    p_hash: Optional[bytes] = Field(default=None, description="64-bit perceptual hash")
    visual_embed: Optional[List[float]] = Field(default=None, description="4096-dimensional visual vector")
    caption_embed: Optional[List[float]] = Field(default=None, description="caption text vector")
    context_embed: Optional[List[float]] = Field(default=None, description="context text vector")
    text: Optional[str] = Field(default=None, description="reserved interpretation text")
    text_embed: Optional[List[float]] = Field(default=None, description="reserved text vector")
    visual_text_embed: Optional[List[float]] = Field(default=None, description="reserved visual-text vector")
    text_sparse_vec: Optional[Dict[Any, float]] = Field(default=None, description="reserved sparse vector")

    doc_meta: Dict[str, Any] = Field(default_factory=dict, exclude=True, description="legacy metadata carrier")
    imageBytes: Optional[bytes] = Field(default=None, exclude=True, description="processing-only image bytes")

    @property
    def imageId(self) -> str:
        return self.img_id

    @property
    def sysId(self) -> str:
        return self.docId

    @sysId.setter
    def sysId(self, value: str) -> None:
        self.docId = value
