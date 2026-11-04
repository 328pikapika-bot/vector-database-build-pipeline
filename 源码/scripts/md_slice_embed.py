#!/usr/bin/env python3
"""
MD 文档切片入库脚本 (v2.0)
================================
按照《全文切分规则及设计 v2.0》对 Markdown 文档执行三级切分、向量化、写入 Milvus。

用法:
    # 单文件切片入库
    python scripts/md_slice_embed.py --input ./docs/paper.md

    # 目录批量处理
    python scripts/md_slice_embed.py --input ./md_docs/ --recursive

    # 干跑模式 (只切片，不向量化不入库)
    python scripts/md_slice_embed.py --input ./docs/paper.md --dry-run

    # 指定集合名称
    python scripts/md_slice_embed.py --input ./docs/paper.md --collection my_chunks

    # 指定自定义配置 (PDF Memories 向量库)
    python scripts/md_slice_embed.py --input ./docs/paper.md \
        --milvus-host 127.0.0.1 --milvus-port 19532 \
        --embedding-url http://127.0.0.1:8080/v1/embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 将项目根目录和 vector_sync_service 加入 sys.path，以便导入现有模块
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VECTOR_SERVICE = _PROJECT_ROOT / "vector_sync_service"
for _p in (_PROJECT_ROOT, _VECTOR_SERVICE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# 尝试加载 .env 配置
# ---------------------------------------------------------------------------
def _load_dotenv() -> Dict[str, str]:
    """从 vector_sync_service/.env 读取配置，返回键值对字典。"""
    env_vars: Dict[str, str] = {}
    env_paths = [
        _VECTOR_SERVICE / ".env",
        _PROJECT_ROOT / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        env_vars[key.strip()] = value.strip()
            break
    return env_vars


_ENV = _load_dotenv()


def _env(key: str, default: str = "") -> str:
    return _ENV.get(key, default)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class ChunkType(str, Enum):
    TITLE_SECTION = "title_section"
    PARAGRAPH = "paragraph"
    FIXED_CHUNK = "fixed_chunk"
    TABLE = "table"
    REFERENCE = "reference"


class OverlapType(str, Enum):
    NONE = "none"
    CONTEXT = "context"
    FIXED_WINDOW = "fixed_window"


@dataclass
class V2Chunk:
    """v2.0 切片数据模型 — 22 个字段对齐 Milvus Schema。"""

    # === 核心标识 (前4字段) ===
    id: str = ""                        # PDF/MD 文件名
    text_embed: Optional[List[float]] = None  # 2560维向量
    chunk_index: int = 0                # 切片顺序，从0开始
    child_chunk_id: str = ""            # 主键: {id}_{chunk_index}

    # === 文档级元数据 ===
    title: Optional[List[str]] = None
    keyword: Optional[List[str]] = None
    accessType: Optional[str] = None
    classificationCode: Optional[str] = None
    year: Optional[str] = None
    hasAuthorname: Optional[List[str]] = None
    hasCreateOrganizationname: Optional[List[str]] = None
    language: Optional[List[str]] = None

    # === 切片内容 ===
    chunkType: str = "paragraph"
    text: str = ""

    # === 位置信息 (v2.0 修正: char偏移为真实字符偏移) ===
    char_start: int = 0
    char_end: int = 0
    line_start: int = 0
    line_end: int = 0

    # === 结构信息 ===
    sectionPath: str = ""
    tokenCount: int = 0

    # === v2.0 新增 ===
    overlap_type: str = "none"
    parent_chunk_id: str = ""

    def to_milvus_dict(self) -> Dict[str, Any]:
        """转为 Milvus insert 所需的扁平字典。"""
        return {
            "id": self.id,
            "text_embed": self.text_embed or [],
            "chunk_index": self.chunk_index,
            "child_chunk_id": self.child_chunk_id,
            "title": self.title or [],
            "keyword": self.keyword or [],
            "accessType": self.accessType or "",
            "classificationCode": self.classificationCode or "",
            "year": self.year or "",
            "hasAuthorname": self.hasAuthorname or [],
            "hasCreateOrganizationname": self.hasCreateOrganizationname or [],
            "language": self.language or ["unknown"],
            "chunkType": self.chunkType,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "sectionPath": self.sectionPath,
            "tokenCount": self.tokenCount,
            "overlap_type": self.overlap_type,
            "parent_chunk_id": self.parent_chunk_id,
        }


# ---------------------------------------------------------------------------
# 字符偏移索引工具
# ---------------------------------------------------------------------------

def build_char_offset_index(raw_text: str) -> List[int]:
    """构建 行号 → 累积字符偏移 的索引。

    index[0] = 0
    index[1] = len(line_0) + 1
    index[n] = 前 n 行的总字符数 (含换行符)
    """
    lines = raw_text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)  # +1 for \n
    return offsets


# ---------------------------------------------------------------------------
# v2.0 文本切分器
# ---------------------------------------------------------------------------

class V2TextSplitter:
    """v2.0 三级文本切分器。

    实现《全文切分规则及设计 v2.0》第三章的全部算法：
      - 第一级: 章节标题 + 首段合并 → title_section
      - 第二级: 段落切分 → paragraph (排除已被合并的首段)
      - 第三级: 固定长度语义切分 → fixed_chunk
      - 后处理 A: 短段落合并
      - 后处理 B: 段落上下文微重叠
      - 后处理 C: 全局重新索引 + char偏移修正
    """

    def __init__(
        self,
        max_chars: int = 5000,
        overlap_chars: int = 200,
        min_chars: int = 300,
        context_overlap_chars: int = 50,
        vector_dim: int = 2560,
    ):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars
        self.context_overlap_chars = context_overlap_chars
        self.vector_dim = vector_dim

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def split_document(
        self,
        raw_text: str,
        doc_id: str,
        doc_meta: Optional[Dict[str, Any]] = None,
    ) -> List[V2Chunk]:
        """对 MD 全文执行完整的 v2.0 切分流程。

        Args:
            raw_text: MMD/Markdown 原始文本
            doc_id: 文档标识 (文件名，如 paper.md)
            doc_meta: 可选的文档元数据 (title, keyword, year, ...)

        Returns:
            按 chunk_index 排序的切片列表
        """
        from app.demo.fulltextSplitDemo.mmd_parser import MmdParser

        # 构建字符偏移索引 (用于 v2.0 修正)
        self._char_index = build_char_offset_index(raw_text)
        self._raw_text = raw_text

        # 解析 MMD → ParsedDocument
        parsed_doc = MmdParser.parse(raw_text)

        all_chunks: List[V2Chunk] = []

        # ---- 第一级: 章节切分 + 首段合并 (v2.0 优化) ----
        merged_first_para_indices: set = set()
        section_chunks, merged_first_para_indices = self._split_by_sections_v2(
            parsed_doc, doc_id, doc_meta or {}
        )
        all_chunks.extend(section_chunks)

        # ---- 第二级: 段落切分 (排除已被合并的首段) ----
        para_chunks = self._split_by_paragraphs_v2(
            parsed_doc, doc_id, doc_meta or {}, merged_first_para_indices
        )
        all_chunks.extend(para_chunks)

        # ---- 第三级: 固定长度语义切分 ----
        fixed_chunks = self._split_by_fixed_length_v2(all_chunks)
        all_chunks.extend(fixed_chunks)

        # ---- v2.0 后处理 A: 短段落合并 ----
        all_chunks = self._merge_short_chunks(all_chunks)

        # ---- v2.0 后处理 B: 段落上下文微重叠 ----
        all_chunks = self._add_context_overlap(all_chunks)

        # ---- v2.0 后处理 C: 全局重新索引 + char偏移修正 + line偏移填充 ----
        all_chunks = self._reindex_and_fix_offsets(all_chunks, doc_id)

        return all_chunks

    # ------------------------------------------------------------------
    # 第一级: 章节切分 + 首段合并
    # ------------------------------------------------------------------

    def _split_by_sections_v2(
        self, parsed_doc, doc_id: str, doc_meta: Dict[str, Any]
    ) -> Tuple[List[V2Chunk], set]:
        """v2.0 章节切分: 标题 + 首段合并为 title_section chunk。

        Returns:
            (section_chunks, merged_paragraph_indices)
            merged_paragraph_indices: 已被合并的首段在 parsed_doc.paragraphs 中的索引
        """
        chunks: List[V2Chunk] = []
        merged: set = set()

        # 建立 section.startPos → section 的映射，用于匹配首段
        sections = parsed_doc.sections

        for i, section in enumerate(sections):
            # 找到该章节下的第一个段落
            first_para = None
            first_para_idx = -1
            for pi, para in enumerate(parsed_doc.paragraphs):
                if pi in merged:
                    continue
                # 段落属于该章节: startPos 在章节范围内
                if section.startPos <= para.startPos < section.endPos:
                    first_para = para
                    first_para_idx = pi
                    break

            # 构建切片文本: 标题 + 首段
            if first_para and first_para.text.strip():
                content = section.title + "\n\n" + first_para.text
                char_start = self._line_to_char(section.startPos)
                char_end = self._line_to_char(first_para.endPos)
                line_start = section.startPos
                line_end = first_para.endPos
                merged.add(first_para_idx)
            else:
                # 无正文的章节: 仅标题
                content = section.title
                char_start = self._line_to_char(section.startPos)
                char_end = self._line_to_char(section.startPos + 1)
                line_start = section.startPos
                line_end = section.startPos + 1

            chunk = V2Chunk(
                id=doc_id,
                chunk_index=0,  # 后续重新编排
                child_chunk_id="",
                title=doc_meta.get("title"),
                keyword=doc_meta.get("keyword"),
                accessType=doc_meta.get("accessType"),
                classificationCode=doc_meta.get("classificationCode"),
                year=doc_meta.get("year"),
                hasAuthorname=doc_meta.get("hasAuthorname"),
                hasCreateOrganizationname=doc_meta.get("hasCreateOrganizationname"),
                language=doc_meta.get("language", ["unknown"]),
                chunkType=ChunkType.TITLE_SECTION.value,
                text=content,
                char_start=char_start,
                char_end=char_end,
                line_start=line_start,
                line_end=line_end,
                sectionPath=section.title,
                tokenCount=self._estimate_tokens(content),
                overlap_type=OverlapType.NONE.value,
                parent_chunk_id="",
            )
            chunks.append(chunk)

        return chunks, merged

    # ------------------------------------------------------------------
    # 第二级: 段落切分
    # ------------------------------------------------------------------

    def _split_by_paragraphs_v2(
        self, parsed_doc, doc_id: str, doc_meta: Dict[str, Any],
        skip_indices: set
    ) -> List[V2Chunk]:
        """v2.0 段落切分: 为未被合并的段落创建 paragraph chunk。"""
        chunks: List[V2Chunk] = []

        for pi, para in enumerate(parsed_doc.paragraphs):
            if pi in skip_indices:
                continue
            if not para.text.strip():
                continue

            char_start = self._line_to_char(para.startPos)
            char_end = self._line_to_char(para.endPos)

            chunk = V2Chunk(
                id=doc_id,
                chunk_index=0,
                child_chunk_id="",
                title=doc_meta.get("title"),
                keyword=doc_meta.get("keyword"),
                accessType=doc_meta.get("accessType"),
                classificationCode=doc_meta.get("classificationCode"),
                year=doc_meta.get("year"),
                hasAuthorname=doc_meta.get("hasAuthorname"),
                hasCreateOrganizationname=doc_meta.get("hasCreateOrganizationname"),
                language=doc_meta.get("language", ["unknown"]),
                chunkType=ChunkType.PARAGRAPH.value,
                text=para.text,
                char_start=char_start,
                char_end=char_end,
                line_start=para.startPos,
                line_end=para.endPos,
                sectionPath=para.sectionPath,
                tokenCount=self._estimate_tokens(para.text),
                overlap_type=OverlapType.NONE.value,
                parent_chunk_id="",
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # 第三级: 固定长度语义切分
    # ------------------------------------------------------------------

    def _split_by_fixed_length_v2(
        self, existing_chunks: List[V2Chunk]
    ) -> List[V2Chunk]:
        """对超长 paragraph 执行语义化固定长度切分。"""
        from app.demo.fulltextSplitDemo.semantic_splitter import semantic_split as do_semantic_split

        new_chunks: List[V2Chunk] = []
        chunks_to_remove: List[int] = []

        for i, chunk in enumerate(existing_chunks):
            if chunk.chunkType != ChunkType.PARAGRAPH.value:
                continue
            if len(chunk.text) <= self.max_chars:
                continue

            # 执行语义切分
            sub_results = do_semantic_split(
                chunk.text,
                max_chars=self.max_chars,
                overlap_chars=self.overlap_chars,
                search_window=200,
            )

            if len(sub_results) <= 1:
                continue  # 无法再切

            # 标记原 paragraph 待移除
            chunks_to_remove.append(i)

            parent_id = chunk.child_chunk_id or f"{chunk.id}_para_{i}"

            for si, sr in enumerate(sub_results):
                # fixed_chunk 的 char 偏移 = 原段落在原文中的偏移 + 子块在段落内的偏移
                sub_char_start = chunk.char_start + sr["para_start"]
                sub_char_end = chunk.char_start + sr["para_end"]

                # 行号估算: 按比例
                line_ratio_start = sr["para_start"] / max(len(chunk.text), 1)
                line_ratio_end = sr["para_end"] / max(len(chunk.text), 1)
                sub_line_start = chunk.line_start + int(
                    (chunk.line_end - chunk.line_start) * line_ratio_start
                )
                sub_line_end = chunk.line_start + int(
                    (chunk.line_end - chunk.line_start) * line_ratio_end
                )

                new_chunk = V2Chunk(
                    id=chunk.id,
                    chunk_index=0,
                    child_chunk_id="",
                    title=chunk.title,
                    keyword=chunk.keyword,
                    accessType=chunk.accessType,
                    classificationCode=chunk.classificationCode,
                    year=chunk.year,
                    hasAuthorname=chunk.hasAuthorname,
                    hasCreateOrganizationname=chunk.hasCreateOrganizationname,
                    language=chunk.language,
                    chunkType=ChunkType.FIXED_CHUNK.value,
                    text=sr["content"],
                    char_start=sub_char_start,
                    char_end=sub_char_end,
                    line_start=sub_line_start,
                    line_end=sub_line_end,
                    sectionPath=chunk.sectionPath,
                    tokenCount=self._estimate_tokens(sr["content"]),
                    overlap_type=OverlapType.FIXED_WINDOW.value,
                    parent_chunk_id=parent_id,
                )
                new_chunks.append(new_chunk)

        # 移除原超长 paragraph 切片
        for idx in reversed(chunks_to_remove):
            existing_chunks.pop(idx)

        return new_chunks

    # ------------------------------------------------------------------
    # 后处理 A: 短段落合并
    # ------------------------------------------------------------------

    def _merge_short_chunks(self, chunks: List[V2Chunk]) -> List[V2Chunk]:
        """将相邻的短 paragraph 切片合并，确保每个切片 ≥ min_chars。

        采用一趟扫描策略: 对每个 paragraph，如果它短且能和后面一个合并，就合并；
        否则尝试和前面已保留的合并。
        """
        result: List[V2Chunk] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]

            # 非 paragraph 或已足够长的 paragraph 直接保留
            if chunk.chunkType != ChunkType.PARAGRAPH.value or len(chunk.text) >= self.min_chars:
                result.append(chunk)
                i += 1
                continue

            # 短 paragraph: 尝试向前合并 (合并到 result 的最后一个 paragraph)
            if result and result[-1].chunkType == ChunkType.PARAGRAPH.value:
                prev = result[-1]
                if len(prev.text) + len(chunk.text) + 2 <= self.max_chars:
                    prev.text = prev.text + "\n\n" + chunk.text
                    prev.char_end = chunk.char_end
                    prev.line_end = chunk.line_end
                    prev.tokenCount = self._estimate_tokens(prev.text)
                    prev.overlap_type = OverlapType.NONE.value
                    i += 1
                    continue

            # 尝试向后合并 (合并下一个 paragraph 到当前)
            if i + 1 < len(chunks) and chunks[i + 1].chunkType == ChunkType.PARAGRAPH.value:
                next_chunk = chunks[i + 1]
                if len(chunk.text) + len(next_chunk.text) + 2 <= self.max_chars:
                    chunk.text = chunk.text + "\n\n" + next_chunk.text
                    chunk.char_end = next_chunk.char_end
                    chunk.line_end = next_chunk.line_end
                    chunk.tokenCount = self._estimate_tokens(chunk.text)
                    chunk.overlap_type = OverlapType.NONE.value
                    result.append(chunk)
                    i += 2
                    continue

            # 无法合并，保持原样
            result.append(chunk)
            i += 1

        return result

    # ------------------------------------------------------------------
    # 后处理 B: 段落上下文微重叠
    # ------------------------------------------------------------------

    def _add_context_overlap(self, chunks: List[V2Chunk]) -> List[V2Chunk]:
        """为 paragraph 切片添加前后文微重叠 (50 chars)。"""
        para_indices = [
            i for i, c in enumerate(chunks)
            if c.chunkType == ChunkType.PARAGRAPH.value and c.overlap_type == OverlapType.NONE.value
        ]

        for pos, idx in enumerate(para_indices):
            chunk = chunks[idx]
            prefix = ""
            suffix = ""

            # 前文重叠: 取前一个 paragraph 的最后 N 字符
            if pos > 0:
                prev_idx = para_indices[pos - 1]
                prev_text = chunks[prev_idx].text
                if len(prev_text) > self.context_overlap_chars:
                    prefix = prev_text[-self.context_overlap_chars:]
                elif prev_text:
                    prefix = prev_text

            # 后文重叠: 取后一个 paragraph 的前 N 字符
            if pos + 1 < len(para_indices):
                next_idx = para_indices[pos + 1]
                next_text = chunks[next_idx].text
                if len(next_text) > self.context_overlap_chars:
                    suffix = next_text[:self.context_overlap_chars]
                elif next_text:
                    suffix = next_text

            if prefix or suffix:
                # 在文本前后添加上下文 (用特殊分隔标记)
                parts = []
                if prefix:
                    parts.append(prefix)
                parts.append(chunk.text)
                if suffix:
                    parts.append(suffix)
                chunk.text = "\n...\n".join(parts) if prefix and suffix else (
                    prefix + "\n...\n" + chunk.text if prefix else chunk.text + "\n...\n" + suffix
                )
                chunk.overlap_type = OverlapType.CONTEXT.value

        return chunks

    # ------------------------------------------------------------------
    # 后处理 C: 全局重新索引 + 偏移修正
    # ------------------------------------------------------------------

    def _reindex_and_fix_offsets(
        self, chunks: List[V2Chunk], doc_id: str
    ) -> List[V2Chunk]:
        """按 char_start 排序，重新分配 chunk_index，更新 child_chunk_id。"""
        # 按 char_start 排序
        chunks.sort(key=lambda c: (c.char_start, c.chunkType != ChunkType.TITLE_SECTION.value))

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.child_chunk_id = f"{doc_id}_{i}"

        return chunks

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _line_to_char(self, line_no: int) -> int:
        """将行号转换为真实字符偏移。"""
        if 0 <= line_no < len(self._char_index):
            return self._char_index[line_no]
        # 越界回退
        return self._char_index[-1] if self._char_index else 0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数量 (v2.0 标注为估算值)。"""
        if not text:
            return 0
        chinese_chars = len(re.findall(r"[一-鿿]", text))
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        return chinese_chars + english_words


# ---------------------------------------------------------------------------
# 同步 Embedding 客户端
# ---------------------------------------------------------------------------

class SyncEmbedder:
    """同步 HTTP Embedding 客户端。

    封装对 Embedding API 的同步调用，支持重试和维度校验。
    """

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        api_key: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        vector_dim: int = 2560,
    ):
        import requests as req_lib

        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.vector_dim = vector_dim
        self._session = req_lib.Session()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers.update(headers)

    def embed_single(self, text: str) -> Optional[List[float]]:
        """对单个文本向量化，失败返回 None。"""
        if not text or not text.strip():
            return [0.0] * self.vector_dim

        payload = {"model": self.model, "input": text}

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    self.base_url, json=payload, timeout=self.timeout
                )
                if resp.status_code == 429:
                    print(f"  [Embed] 限流 (429), 重试 {attempt + 1}/{self.max_retries}")
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code >= 500:
                    print(f"  [Embed] 服务端错误 ({resp.status_code}), 重试 {attempt + 1}/{self.max_retries}")
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    print(f"  [Embed] HTTP {resp.status_code}: {resp.text[:200]}")
                    return None

                data = resp.json()
                embeddings = data.get("data", [])
                if not embeddings:
                    print(f"  [Embed] 空响应: {data}")
                    return None

                vector = embeddings[0].get("embedding", [])
                if not isinstance(vector, list) or not vector:
                    print(f"  [Embed] 无效向量: {data}")
                    return None

                if len(vector) != self.vector_dim:
                    print(f"  [Embed] 维度不匹配: 期望 {self.vector_dim}, 实际 {len(vector)}")
                    return None

                return vector

            except Exception as e:
                print(f"  [Embed] 网络错误: {e}, 重试 {attempt + 1}/{self.max_retries}")
                time.sleep(2 ** attempt)

        return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量向量化，逐个调用 (Embedding API 通常不支持批量)。"""
        results: List[Optional[List[float]]] = []
        total = len(texts)
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  [Embed] 进度: {i}/{total}")
            vector = self.embed_single(text)
            results.append(vector)
        return results

    def close(self) -> None:
        self._session.close()


# ---------------------------------------------------------------------------
# Milvus 写入器
# ---------------------------------------------------------------------------

class MilvusWriter:
    """Milvus v2.0 集合写入器。

    负责创建 v2.0 schema 集合、删除旧切片、插入新切片。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
    ):
        from pymilvus import MilvusClient

        uri = f"http://{host}:{port}"
        self.client = MilvusClient(uri=uri, user=user or "", password=password or "")
        self._uri = uri

    def init_collection(
        self,
        collection_name: str,
        vector_dim: int = 2560,
    ) -> bool:
        """初始化 v2.0 集合 (如果不存在则创建)。

        Returns:
            True 表示是新创建的，False 表示已存在。
        """
        from pymilvus import DataType

        if self.client.has_collection(collection_name):
            self.client.load_collection(collection_name)
            print(f"[Milvus] 集合 '{collection_name}' 已存在，使用现有集合")
            return False

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)

        # v2.0 Schema: 22 个字段 (参考《全文切分规则及设计 v2.0》第二章)
        # 前4个核心字段
        schema.add_field("id", DataType.VARCHAR, max_length=256)
        schema.add_field("text_embed", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("child_chunk_id", DataType.VARCHAR, is_primary=True, max_length=512)

        # 文档元数据 (ARRAY 字段)
        _add_array_field(schema, "title")
        _add_array_field(schema, "keyword")
        schema.add_field("accessType", DataType.VARCHAR, max_length=128, nullable=True)
        schema.add_field("classificationCode", DataType.VARCHAR, max_length=128, nullable=True)
        schema.add_field("year", DataType.VARCHAR, max_length=64, nullable=True)
        _add_array_field(schema, "hasAuthorname")
        _add_array_field(schema, "hasCreateOrganizationname")
        _add_array_field(schema, "language")

        # 切片内容
        schema.add_field("chunkType", DataType.VARCHAR, max_length=32)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)

        # 位置信息
        schema.add_field("char_start", DataType.INT64)
        schema.add_field("char_end", DataType.INT64)
        schema.add_field("line_start", DataType.INT64)
        schema.add_field("line_end", DataType.INT64)

        # 结构信息
        schema.add_field("sectionPath", DataType.VARCHAR, max_length=1024)
        schema.add_field("tokenCount", DataType.INT64)

        # v2.0 新增
        schema.add_field("overlap_type", DataType.VARCHAR, max_length=32, nullable=True)
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512, nullable=True)

        # 生产统一使用 IVF_PQ；2560 维可被 m=64 整除。
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="text_embed",
            index_type="IVF_PQ",
            metric_type="COSINE",
            params={"nlist": 4096, "m": 64, "nbits": 8},
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name)
        print(f"[Milvus] 集合 '{collection_name}' 创建成功 (v2.0 schema, {vector_dim}维)")
        return True

    def delete_by_doc(self, collection_name: str, doc_id: str) -> int:
        """删除指定文档的全部旧切片。"""
        if not self.client.has_collection(collection_name):
            return 0
        escaped = doc_id.replace("\\", "\\\\").replace('"', '\\"')
        filter_expr = f'id == "{escaped}"'
        try:
            result = self.client.delete(collection_name=collection_name, filter=filter_expr)
            deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
            if deleted:
                print(f"[Milvus] 已删除 {deleted} 条旧切片 (id={doc_id})")
            return deleted
        except Exception as e:
            print(f"[Milvus] 删除旧切片失败: {e}")
            return 0

    def insert_chunks(
        self, collection_name: str, chunks: List[V2Chunk]
    ) -> int:
        """批量插入切片。

        Returns:
            成功插入的条数。
        """
        if not chunks:
            return 0

        records = [c.to_milvus_dict() for c in chunks]

        try:
            result = self.client.insert(
                collection_name=collection_name, data=records
            )
            count = result.get("insert_count", len(records))
            print(f"[Milvus] 成功写入 {count} 条切片")
            return count
        except Exception as e:
            print(f"[Milvus] 写入失败: {e}")
            traceback.print_exc()
            return 0

    def close(self) -> None:
        if self.client:
            self.client.close()


def _add_array_field(schema: Any, field_name: str) -> None:
    """添加 ARRAY[VARCHAR] 类型字段。"""
    from pymilvus import DataType
    schema.add_field(
        field_name,
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=4096,
        max_length=65535,
        nullable=True,
    )


# ---------------------------------------------------------------------------
# 文档元数据提取
# ---------------------------------------------------------------------------

def extract_meta_from_md(raw_text: str, file_path: str) -> Dict[str, Any]:
    """从 MD 文件中提取基本元数据 (标题、文件名等)。"""
    from app.demo.fulltextSplitDemo.mmd_parser import MmdParser

    title = MmdParser.get_heading_title(raw_text)

    filename = os.path.basename(file_path)

    return {
        "title": [title] if title else [filename],
        "keyword": None,
        "accessType": None,
        "classificationCode": None,
        "year": None,
        "hasAuthorname": None,
        "hasCreateOrganizationname": None,
        "language": ["unknown"],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_single_file(
    file_path: str,
    splitter: V2TextSplitter,
    embedder: Optional[SyncEmbedder],
    writer: Optional[MilvusWriter],
    collection_name: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """处理单个 MD 文件: 切分 → 向量化 → 入库。"""
    file_path = os.path.abspath(file_path)
    filename = os.path.basename(file_path)
    doc_id = filename  # v2.0: id 即文件名

    print(f"\n{'=' * 60}")
    print(f"处理文件: {file_path}")
    print(f"文档 ID: {doc_id}")

    # 1. 读取文件
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if not raw_text.strip():
        print("  [跳过] 文件为空")
        return {"file": file_path, "status": "empty", "chunks": 0}

    print(f"  原文长度: {len(raw_text)} 字符")

    # 2. 提取元数据
    doc_meta = extract_meta_from_md(raw_text, file_path)
    print(f"  标题: {doc_meta['title']}")

    # 3. 切分
    print(f"  正在切分 (max_chars={splitter.max_chars}, overlap={splitter.overlap_chars})...")
    chunks = splitter.split_document(raw_text, doc_id, doc_meta)
    print(f"  切分完成: {len(chunks)} 个切片")

    # 打印切分统计
    type_counts: Dict[str, int] = {}
    for c in chunks:
        t = c.chunkType
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, cnt in sorted(type_counts.items()):
        print(f"    {t}: {cnt}")

    if dry_run:
        print("\n  [干跑模式] 跳过向量化和入库")
        # 打印前几个切片预览
        for i, c in enumerate(chunks[:5]):
            preview = c.text[:120].replace("\n", "\\n")
            print(f"  --- Chunk {i} [{c.chunkType}] ({len(c.text)} chars) ---")
            print(f"      overlap={c.overlap_type}, char=[{c.char_start},{c.char_end}), line=[{c.line_start},{c.line_end})")
            print(f"      text: {preview}...")
        if len(chunks) > 5:
            print(f"  ... (还有 {len(chunks) - 5} 个切片)")
        return {"file": file_path, "status": "dry_run", "chunks": len(chunks), "chunks_by_type": type_counts}

    # 4. 向量化
    if embedder is None:
        print("  [错误] Embedder 未初始化")
        return {"file": file_path, "status": "error", "chunks": 0, "error": "no embedder"}

    print(f"  正在向量化 {len(chunks)} 个切片...")
    texts = [c.text for c in chunks]
    vectors = embedder.embed_batch(texts)

    success_count = 0
    for chunk, vector in zip(chunks, vectors):
        if vector is not None:
            chunk.text_embed = vector
            success_count += 1

    print(f"  向量化完成: {success_count}/{len(chunks)} 成功")

    if success_count == 0:
        print("  [错误] 所有切片向量化失败")
        return {"file": file_path, "status": "embed_failed", "chunks": 0}

    # 只保留向量化成功的切片
    valid_chunks = [c for c in chunks if c.text_embed is not None]

    # 5. 写入 Milvus
    if writer is None:
        print("  [错误] Milvus writer 未初始化")
        return {"file": file_path, "status": "error", "chunks": 0, "error": "no milvus writer"}

    # 初始化集合
    writer.init_collection(collection_name, vector_dim=splitter.vector_dim)

    # 先删后插
    writer.delete_by_doc(collection_name, doc_id)
    inserted = writer.insert_chunks(collection_name, valid_chunks)

    return {
        "file": file_path,
        "status": "success",
        "chunks": len(valid_chunks),
        "embedded": success_count,
        "inserted": inserted,
        "chunks_by_type": type_counts,
    }


SUPPORTED_EXTENSIONS = (".md", ".mmd")


def collect_files(path: str, recursive: bool = False) -> List[str]:
    """收集待处理的 .md / .mmd 文件列表。"""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if path.endswith(SUPPORTED_EXTENSIONS):
            return [path]
        else:
            print(f"[警告] {path} 不是 .md/.mmd 文件，跳过")
            return []

    if not os.path.isdir(path):
        print(f"[错误] 路径不存在: {path}")
        return []

    files: List[str] = []
    if recursive:
        for root, _dirs, filenames in os.walk(path):
            for fname in filenames:
                if fname.endswith(SUPPORTED_EXTENSIONS):
                    files.append(os.path.join(root, fname))
    else:
        for fname in sorted(os.listdir(path)):
            if fname.endswith(SUPPORTED_EXTENSIONS):
                files.append(os.path.join(path, fname))

    return sorted(files)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MD 文档切片入库脚本 v2.0 — 按《全文切分规则及设计 v2.0》切分并写入 Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/md_slice_embed.py --input ./paper.md
  python scripts/md_slice_embed.py --input ./md_docs/ --recursive --dry-run
  python scripts/md_slice_embed.py --input ./paper.md --collection my_test_chunks
        """,
    )

    # 输入
    parser.add_argument("--input", "-i", required=True,
                        help="输入 .md 文件或目录路径")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="递归处理子目录中的 .md 文件")

    # 运行模式
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑模式: 只切分不向量化不入库")

    # Milvus
    parser.add_argument("--milvus-host", default=_env("MILVUS_HOST", "localhost"),
                        help=f"Milvus 主机地址 (默认: {_env('MILVUS_HOST', 'localhost')})")
    parser.add_argument("--milvus-port", type=int, default=int(_env("PDF_MEMORIES_MILVUS_PORT",
                        _env("MILVUS_PORT", "19532"))),
                        help="Milvus 端口 (默认: 19532 PDF Memories)")
    parser.add_argument("--milvus-user", default=_env("MILVUS_USER", ""),
                        help="Milvus 用户名")
    parser.add_argument("--milvus-password", default=_env("MILVUS_PASSWORD", ""),
                        help="Milvus 密码")
    parser.add_argument("--collection", "-c",
                        default=_env("FULLTEXT_CHUNK_COLLECTION", "pdf_fulltext_chunks"),
                        help="Milvus 集合名称 (默认: pdf_fulltext_chunks)")

    # Embedding
    parser.add_argument("--embedding-url", default=_env("EMBEDDING_API_URL", "http://localhost:8080/v1/embeddings"),
                        help="Embedding API 地址")
    parser.add_argument("--embedding-model", default=_env("EMBEDDING_MODEL", "medium"),
                        help="Embedding 模型名称")
    parser.add_argument("--embedding-key", default=_env("EMBEDDING_API_KEY", ""),
                        help="Embedding API Key")
    parser.add_argument("--vector-dim", type=int, default=int(_env("VECTOR_DIM", "2560")),
                        help="向量维度 (默认: 2560)")

    # 切分参数
    parser.add_argument("--max-chars", type=int, default=int(_env("CHUNK_MAX_CHARS", "5000")),
                        help="单切片最大字符数 (默认: 5000)")
    parser.add_argument("--overlap-chars", type=int, default=int(_env("CHUNK_OVERLAP_CHARS", "200")),
                        help="固定窗口重叠字符数 (默认: 200)")
    parser.add_argument("--min-chars", type=int, default=int(_env("CHUNK_MIN_CHARS", "300")),
                        help="单切片最小字符数 (默认: 300)")
    parser.add_argument("--context-overlap", type=int, default=50,
                        help="段落上下文微重叠字符数 (默认: 50)")

    # 其他
    parser.add_argument("--timeout", type=int, default=30,
                        help="Embedding 请求超时秒数 (默认: 30)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Embedding 最大重试次数 (默认: 3)")

    args = parser.parse_args()

    # 收集文件
    files = collect_files(args.input, args.recursive)
    if not files:
        print("未找到 .md 文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个 .md 文件待处理")
    for f in files:
        print(f"  - {f}")

    # 初始化组件
    splitter = V2TextSplitter(
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        min_chars=args.min_chars,
        context_overlap_chars=args.context_overlap,
        vector_dim=args.vector_dim,
    )

    embedder: Optional[SyncEmbedder] = None
    writer: Optional[MilvusWriter] = None

    if not args.dry_run:
        embedder = SyncEmbedder(
            base_url=args.embedding_url,
            model=args.embedding_model,
            api_key=args.embedding_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
            vector_dim=args.vector_dim,
        )
        writer = MilvusWriter(
            host=args.milvus_host,
            port=args.milvus_port,
            user=args.milvus_user,
            password=args.milvus_password,
        )

    # 逐个处理
    results = []
    for f in files:
        try:
            result = process_single_file(
                file_path=f,
                splitter=splitter,
                embedder=embedder,
                writer=writer,
                collection_name=args.collection,
                dry_run=args.dry_run,
            )
            results.append(result)
        except Exception as e:
            print(f"\n[错误] 处理 {f} 时出错: {e}")
            traceback.print_exc()
            results.append({"file": f, "status": "error", "chunks": 0, "error": str(e)})

    # 清理
    if embedder:
        embedder.close()
    if writer:
        writer.close()

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"处理完成: {len(results)} 个文件")
    total_chunks = sum(r.get("chunks", 0) for r in results)
    success = sum(1 for r in results if r.get("status") == "success")
    dry = sum(1 for r in results if r.get("status") == "dry_run")
    errors = sum(1 for r in results if r.get("status") in ("error", "embed_failed"))
    print(f"  成功: {success}, 干跑: {dry}, 失败: {errors}")
    print(f"  总切片数: {total_chunks}")

    # 保存结果摘要
    summary_path = os.path.join(os.path.dirname(__file__) or ".", "slice_result.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  结果摘要: {summary_path}")


if __name__ == "__main__":
    main()
