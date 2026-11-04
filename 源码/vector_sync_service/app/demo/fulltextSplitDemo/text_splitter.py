"""
三级文本切分器
第一级: 章节切分 (Section Split)
第二级: 段落切分 (Paragraph Split)
第三级: 固定长度切分 (Fixed-length Split, 含重叠)
"""
from typing import List, Optional

from .models import FullTextChunk, RedundantMeta, ChunkType
from .mmd_parser import MmdParser, ParsedDocument, Paragraph, Section


# 默认配置
DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 200
DEFAULT_VECTOR_DIM = 2560


class TextSplitter:
    """三级文本切分器"""

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        vector_dim: int = DEFAULT_VECTOR_DIM,
    ):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.vector_dim = vector_dim

    def split_document(
        self,
        mmd_text: str,
        sys_id: str,
        meta: RedundantMeta,
    ) -> List[FullTextChunk]:
        """
        对 MMD 全文执行三级切分

        Args:
            mmd_text: MMD 文件原文
            sys_id: 文献 sysId
            meta: 冗余元数据

        Returns:
            切分后的子块列表
        """
        parsed_doc = MmdParser.parse(mmd_text)
        return self.split_parsed_document(parsed_doc, sys_id, meta)

    def split_parsed_document(
        self,
        parsed_doc: ParsedDocument,
        sys_id: str,
        meta: RedundantMeta,
    ) -> List[FullTextChunk]:
        """
        对已解析的 ParsedDocument 执行三级切分

        Args:
            parsed_doc: 已解析的文档对象
            sys_id: 文献 sysId
            meta: 冗余元数据

        Returns:
            切分后的子块列表
        """
        all_chunks: List[FullTextChunk] = []
        chunk_idx = 0

        # 第一级: 为每个章节创建 section_header chunk
        section_chunks = self._split_by_sections(
            parsed_doc.sections, sys_id, meta, chunk_idx
        )
        all_chunks.extend(section_chunks)
        chunk_idx += len(section_chunks)

        # 第二级: 将每个章节的段落转为 paragraph chunk
        para_chunks = self._split_by_paragraphs(
            parsed_doc.paragraphs, sys_id, meta, chunk_idx
        )
        all_chunks.extend(para_chunks)
        chunk_idx += len(para_chunks)

        # 第三级: 对超长段落进行固定长度切分
        fixed_chunks = self._split_by_fixed_length(
            all_chunks, sys_id, meta, chunk_idx
        )
        all_chunks.extend(fixed_chunks)

        # 重新分配全局 chunkIndex
        for i, c in enumerate(all_chunks):
            c.chunkIndex = i
            c.chunkId = f"{sys_id}_{i}"

        return all_chunks

    def _split_by_sections(
        self,
        sections: List[Section],
        sys_id: str,
        meta: RedundantMeta,
        start_idx: int,
    ) -> List[FullTextChunk]:
        """第一级: 章节切分"""
        chunks = []

        for i, section in enumerate(sections):
            # 构建 sectionPath
            chunk = FullTextChunk(
                chunkId=f"{sys_id}_{start_idx + i}",
                sysId=sys_id,
                chunkIndex=start_idx + i,
                chunkType=ChunkType.SECTION_HEADER,
                redundantMeta=meta,
                content=section.title,
                contentVector=[0.0] * self.vector_dim,
                charStart=section.startPos,
                charEnd=section.endPos,
                sectionPath=section.title,
                level=section.level,
                tokenCount=self._estimate_tokens(section.title),
            )
            chunks.append(chunk)

        return chunks

    def _split_by_paragraphs(
        self,
        paragraphs: List[Paragraph],
        sys_id: str,
        meta: RedundantMeta,
        start_idx: int,
    ) -> List[FullTextChunk]:
        """第二级: 段落切分"""
        chunks = []

        for i, para in enumerate(paragraphs):
            chunk = FullTextChunk(
                chunkId=f"{sys_id}_{start_idx + i}",
                sysId=sys_id,
                chunkIndex=start_idx + i,
                chunkType=ChunkType.PARAGRAPH,
                redundantMeta=meta,
                content=para.text,
                contentVector=[0.0] * self.vector_dim,
                paraStartIdx=para.startPos,
                paraEndIdx=para.endPos,
                charStart=para.startPos,
                charEnd=para.endPos,
                sectionPath=para.sectionPath,
                level=para.sectionLevel,
                tokenCount=self._estimate_tokens(para.text),
            )
            chunks.append(chunk)

        return chunks

    def _split_by_fixed_length(
        self,
        existing_chunks: List[FullTextChunk],
        sys_id: str,
        meta: RedundantMeta,
        start_idx: int,
    ) -> List[FullTextChunk]:
        """第三级: 对超长段落进行固定长度切分 (语义化)"""
        chunks = []
        chunk_idx = start_idx

        for para_chunk in existing_chunks:
            if para_chunk.chunkType != ChunkType.PARAGRAPH:
                continue

            text = para_chunk.content
            if len(text) <= self.max_chars:
                continue

            sub_chunks = self._semantic_split(
                text,
                sys_id,
                meta,
                chunk_idx,
                para_chunk.chunkId,
                para_chunk.sectionPath,
                para_chunk.level,
                para_chunk.paraStartIdx,
            )
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)

        return chunks

    def _semantic_split(
        self,
        text: str,
        sys_id: str,
        meta: RedundantMeta,
        start_idx: int,
        parent_para_id: str,
        section_path: str,
        level: int,
        base_offset: int,
    ) -> List[FullTextChunk]:
        """对单个长文本执行语义化切分 (句子/从句边界 + LaTeX 保护)"""
        from .semantic_splitter import semantic_split as do_semantic_split

        raw_chunks = do_semantic_split(
            text,
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
            search_window=200,
        )

        chunks = []
        for i, rc in enumerate(raw_chunks):
            chunk = FullTextChunk(
                chunkId=f"{sys_id}_{start_idx + i}",
                sysId=sys_id,
                chunkIndex=start_idx + i,
                chunkType=ChunkType.FIXED_CHUNK,
                redundantMeta=meta,
                content=rc["content"],
                contentVector=[0.0] * self.vector_dim,
                paraStartIdx=rc["para_start"],
                paraEndIdx=rc["para_end"],
                charStart=base_offset + rc["para_start"],
                charEnd=base_offset + rc["para_end"],
                sectionPath=section_path,
                level=level,
                parentParaId=parent_para_id,
                tokenCount=self._estimate_tokens(rc["content"]),
            )
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数量（中文字符约 0.5 token/char, 英文约 0.25 token/char）"""
        if not text:
            return 0
        # 简单估算: 英文按空格分词+中文字符计数
        import re
        # 中文字符
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # 英文单词
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        return chinese_chars + english_words
