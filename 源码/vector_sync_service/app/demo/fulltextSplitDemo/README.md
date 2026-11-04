# 全文数据三级切分 Demo 技术文档

## 目录

1. [概述](#1-概述)
2. [整体逻辑 (Pipeline)](#2-整体逻辑-pipeline)
3. [MMD 文档解析 (AST-based)](#3-mmd-文档解析-ast-based)
4. [三级文本切分 (TextSplitter)](#4-三级文本切分-textsplitter)
5. [子块数据结构 (Data Models)](#5-子块数据结构-data-models)
6. [输出格式 (ChunkWriter)](#6-输出格式-chunkwriter)
7. [运行方式 (run_demo.py)](#7-运行方式-run_demopy)

---

## 1. 概述

### 1.1 Demo 目标

将科学文献的 `.mmd` (Markdown 变体) 全文按 **章节 → 段落 → 定长块** 三级策略切分为可供向量检索的独立子块 (Chunk)。

### 1.2 核心设计理念

每个子块携带**完整冗余元数据**（标题、作者、年份、关键词、章节路径等），使得每个块都可以作为**独立检索单元**，不依赖原文上下文即可理解其来源和语义范围。

### 1.3 模块架构

```
run_demo.py          ← 入口，探测 Milvus + 驱动 Pipeline
    │
    ▼
pipeline.py          ← 核心编排：数据获取 → 文件加载 → 切分 → 写入
    │
    ├── models.py          ← 数据模型定义 (FullTextChunk, RedundantMeta, SplitResult...)
    ├── mmd_parser.py      ← AST-based MMD 解析器 (Lexer → Parser → AST)
    ├── text_splitter.py   ← 三级切分器 (Section → Paragraph → Fixed)
    └── chunk_writer.py    ← JSONL 写入器
```

**文件清单：**

| 文件 | 职责 |
|---|---|
| `__init__.py` | 模块导出，暴露所有公共接口 |
| `models.py` | ChunkType 枚举、FullTextChunk、RedundantMeta、SplitResult 等数据模型 |
| `mmd_parser.py` | Lexer + Parser + AST → ParsedDocument 兼容层 |
| `text_splitter.py` | 三级文本切分引擎 |
| `pipeline.py` | 流水线编排，兼容 Milvus / Kafka 两种数据格式 |
| `chunk_writer.py` | JSONL 格式输出 |
| `run_demo.py` | 测试入口 |

---

## 2. 整体逻辑 (Pipeline)

### 2.1 数据源优先级

Pipeline 获取文献元数据按 **三级回退** 策略：

```
本地 JSON (Kafka 格式 / Milvus 扁平格式)
    ↓ 未命中
Milvus 向量库 (pymilvus.query)
    ↓ 未命中 / 连接失败
本地 .mmd 文件全局扫描
```

### 2.2 主流程 `process_by_sys_id(sys_id)`

```
┌─────────────────────────────────────────────────────────────────┐
│                    process_by_sys_id(sys_id)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: 获取文献元数据                                          │
│    ├── _fetch_metadata()      → 本地 sample.json                │
│    ├── _fetch_from_milvus()   → pymilvus query (HTTP 连接)      │
│    └── _process_local_only()  → 全局扫描 .mmd (最终回退)        │
│                                                                  │
│  Step 2: 提取全文文件列表                                        │
│    └── _extract_fulltext_files()                                 │
│        兼容两种格式:                                             │
│        · Kafka: article["op"]["hasFullTextFile"]["value"]        │
│        · Milvus: article["hasFullTextFile"]                      │
│        输出: List[FullTextFileInfo]                              │
│                                                                  │
│  Step 3: 加载 .mmd 内容                                          │
│    └── _load_mmd_file()                                          │
│        ├── _load_mmd_local()   → mmd_dir → base_dir → 全局扫描  │
│        └── _download_mmd()     → https://static.nstl.gov.cn/ + url│
│                                                                  │
│  Step 4: 执行三级文本切分                                        │
│    └── self.splitter.split_document(mmd_text, sys_id, meta)      │
│        输出: List[FullTextChunk]                                 │
│                                                                  │
│  Step 5: 输出结果                                                │
│    ├── self.writer.save_chunks(result)  → JSONL 文件             │
│    └── 返回 SplitResult                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 数据格式兼容

Pipeline 兼容两种元数据格式：

**格式 A — Kafka 嵌套格式：**

```json
{
  "data": [{
    "dp": { "sysId": {"value": ["abc123"]}, "title": {"value": ["Title"]} },
    "op": { "hasFullTextFile": { "value": [{ ... }] } }
  }]
}
```

**格式 B — Milvus 扁平格式：**

```json
{
  "sysId": "abc123",
  "title": ["Title"],
  "year": "2023",
  "hasFullTextFile": [
    { "id": "...", "name": "xxx.mmd", "url": ["/fulltext/..."], "size": 4538 }
  ]
}
```

**`hasFullTextFile` 结构 (扁平格式)：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | 文件内部唯一 ID |
| `name` | `str` | 文件名（如 `2de741fe-.....mmd`） |
| `url` | `list[str]` | 相对路径 URL 数组 |
| `fileType` | `str` | 固定值 `"mmd"` |
| `size` | `int` | 文件大小（字节） |

### 2.4 冗余元数据提取

`_extract_redundant_meta(article)` 从文献记录中提取以下字段，**附加到每个子块的 `redundantMeta`**：

| 字段 | 来源 (Milvus) | 来源 (Kafka) |
|---|---|---|
| `sysId` | `article["sysId"]` | `dp["sysId"]["value"][0]` |
| `title` | `article["title"]` | `dp["title"]["value"][0]` |
| `authors` | `article["hasAuthor"]` | `op["hasAuthor"]["value"][*]["name"]` |
| `year` | `article["year"]` | `dp["year"]["value"][0]` |
| `keywords` | `article["keyword"]` | `dp["keyword"]["value"]` |
| `journal` | — | `op["hasSource"]["value"][0]["name"]` |
| `doi` | `article["doi"]` | `dp["doi"]["value"][0]` |
| `sourceFileName` | `article["hasFullTextFilename"]` | — |

---

## 3. MMD 文档解析 (AST-based)

### 3.1 架构总览

```
raw_text (.mmd)
    │
    ▼
┌──────────────────────────────────────┐
│ Lexer (词法分析)                      │
│   按行扫描 → Token 流                 │
│   Token 类型: HEADING / TEXT / BLANK │
│──────────────────────────────────────│
│ Parser (递归下降解析)                 │
│   Token 流 → AST 文档树               │
│   5 阶段流水线                        │
│──────────────────────────────────────│
│ DocumentNode (AST 根)                │
│   ├── SectionNode* (章节, 含子树)     │
│   ├── ParagraphNode* (段落)           │
│   └── references (引用文本)           │
│──────────────────────────────────────│
│ MmdParser (兼容层)                    │
│   AST → ParsedDocument               │
└──────────────────────────────────────┘
    │
    ▼
ParsedDocument (向下兼容旧接口)
```

### 3.2 Token 类型

```python
class TokenType(Enum):
    HEADING = auto()   # Markdown 标题行 (# 到 ######)
    TEXT    = auto()   # 正文行
    BLANK   = auto()   # 空行
    EOF     = auto()   # 结束标记
```

**Token 结构：**

```
Token
├── type:    TokenType   # 类型
├── value:   str         # 内容 (标题名 / 行文本)
├── line:    int         # 行号 (0-based)
└── level:   int         # 标题层级 (1-6, 仅 HEADING)
```

### 3.3 AST 节点树

```
DocumentNode (根节点)
├── raw_text: str                    # 原始全文
├── title: str                       # 文档标题 (第一个 H1)
├── sections: List[SectionNode]      # 章节列表 (含树结构)
├── paragraphs: List[ParagraphNode]  # 段落列表 (扁平)
└── references: str                  # 参考文献文本

SectionNode (章节节点)
├── title: str                       # 章节标题 (去除 #)
├── level: int                       # 层级 (1-6)
├── start_pos / end_pos: int         # 行号范围
├── children: List[SectionNode]      # 子章节
├── paragraphs: List[ParagraphNode]  # 段落
└── content_only: str                # 自身纯文本 (不含子章节)

ParagraphNode (段落节点)
├── text: str                        # 段落内容
├── start_pos / end_pos: int         # 行号范围
├── section_path: str                # 章节层级路径 (如 "H1/H2/H3")
└── section_level: int               # 所属章节层级
```

### 3.4 Parser 五阶段解析流程

**阶段一：扁平解析所有章节**

遍历 Token 流，遇到 HEADING 就创建 `SectionNode`，计算结束位置（下一个同级/更高级标题之前）。

**阶段二：用栈构建章节树**

```
示例：  # H1              遍历: H1 → 栈=[root, H1]
        ## H2             遍历: H2(≥2 出栈) → 栈=[root, H1, H2]
        ### H3            遍历: H3(≥3 出栈) → 栈=[root, H1, H2, H3]
        ## H2             遍历: H2(≥2 出栈 H3,H2) → 栈=[root, H1] → 出栈后 append H2
```

**阶段三：提取纯文本内容**

每个章节取其行范围内的文本，跳过自身标题行，遇子标题行 (`level > 自身 level`) 即停止。

**阶段四：按空行分割段落**

用正则 `\n\s*\n` 将每个章节的 `content_only` 分割为 `ParagraphNode` 列表。

**阶段五：检测并分离参考文献**

匹配 `# References` / `## References` / `### References` / `# Bibliography` 等标题，将之后的内容隔离为 `doc.references`。

### 3.5 边界情况

| 场景 | 处理方式 |
|---|---|
| 空文档 `""` | 返回空 `ParsedDocument` |
| 无标题纯文本 | 整个文本作为一个 `ParagraphNode` (level=0) |
| 仅有标题无内容 | 章节被解析但段落为空 |

---

## 4. 三级文本切分 (TextSplitter)

### 4.1 切分配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_chars` | 1500 | 第三级固定切分的最大字符数 |
| `overlap_chars` | 200 | 相邻块之间的重叠字符数 |
| `vector_dim` | 2560 | 预留向量维度（当前填充零向量） |

### 4.2 第一级 — 章节标题切分

```
ChunkType: section_header
来源:      parsed_doc.sections[*]
内容:      section.title (章节标题文本)
```

每个 Markdown 标题（`#` ~ `######`）生成一个 `section_header` 块，记录 `sectionPath=标题本身`、`level=层级`。

### 4.3 第二级 — 段落切分

```
ChunkType: paragraph
来源:      parsed_doc.paragraphs[*]
内容:      paragraph.text (段落全文)
```

每个段落生成一个 `paragraph` 块，关联所属的 `sectionPath`（如 `H1/H2`）和 `sectionLevel`。

### 4.4 第三级 — 固定长度 + 重叠切分

```
触发条件: paragraph 块的 content 长度 > max_chars (1500)

滑动窗口算法:
  step = max_chars - overlap_chars  (= 1500 - 200 = 1300)
  pos = 0
  while pos < text_len:
      chunk = text[pos : pos + max_chars]
      pos += step
      if 剩余部分 < max_chars:  直接取到最后一块 (不做过度切割)
```

```
示例: text_len = 4000, max_chars = 1500, overlap_chars = 200

  [0────1500]                  chunk_0
      [1300──────2800]         chunk_1 (与前一块重叠 200 chars)
            [2600──────4100]   chunk_2 (与前一块重叠 200 chars)
                  [3900─4000]  ← 剩余不足一个窗口，直接取完
```

生成的 `fixed_chunk` 块：
- `chunkType = FIXED_CHUNK`
- `parentParaId` 指向父 `paragraph` 块的 `chunkId`
- 保留父块的 `sectionPath` 和 `level`

### 4.5 Token 估算

```python
def _estimate_tokens(text):
    chinese_chars = len(中文)      # 每个中文字符 ≈ 1 token
    english_words = len(英文单词)  # 每个英文单词 ≈ 1 token
    return chinese_chars + english_words
```

---

## 5. 子块数据结构 (Data Models)

### 5.1 ChunkType 枚举

```python
class ChunkType(str, Enum):
    SECTION_HEADER = "section_header"   # 第一级: 章节标题块
    PARAGRAPH      = "paragraph"        # 第二级: 段落块
    FIXED_CHUNK    = "fixed_chunk"      # 第三级: 固定长度切分块
```

### 5.2 FullTextChunk — 子块核心数据

#### 标识字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunkId` | `str` | 全局唯一 ID，格式 `{sysId}_{chunkIndex}` |
| `sysId` | `str` | 所属文献 ID |
| `chunkIndex` | `int` | 块序号，从 0 自增 |
| `chunkType` | `ChunkType` | 块类型 |

#### 冗余元数据

| 字段 | 类型 | 说明 |
|---|---|---|
| `redundantMeta` | `RedundantMeta` | 附加元数据对象，使每个块可独立检索 |

#### 内容字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `content` | `str` | 子块文本内容 |
| `contentVector` | `List[float]` | 内容向量（当前全零占位，维度 = `vector_dim`） |

#### 位置信息

| 字段 | 类型 | 说明 |
|---|---|---|
| `paraStartIdx` | `int` | 段落内起始偏移 |
| `paraEndIdx` | `int` | 段落内结束偏移 |
| `charStart` | `int` | 原文中起始行号 |
| `charEnd` | `int` | 原文中结束行号 |

#### 结构信息

| 字段 | 类型 | 说明 |
|---|---|---|
| `sectionPath` | `str` | 章节层级路径，如 `Introduction/Background` |
| `level` | `int` | 章节层级深度 (1-6) |
| `parentParaId` | `str` | 父段落 chunkId（仅 `fixed_chunk` 使用） |
| `tokenCount` | `int` | 近似 token 数量 |

### 5.3 RedundantMeta

```python
@dataclass
class RedundantMeta:
    sysId: str = ""              # 文献 ID
    title: Optional[str] = None  # 文献标题
    authors: List[str] = []      # 作者列表
    year: Optional[str] = None   # 发表年份
    journal: Optional[str] = None# 期刊名
    doi: Optional[str] = None    # DOI
    keywords: List[str] = []     # 关键词列表
    sourceFileName: str = ""     # 来源 .mmd 文件名
```

### 5.4 SplitResult — 切分结果

```python
@dataclass
class SplitResult:
    sysId: str                         # 文献 ID
    success: bool                      # 是否成功
    totalChunks: int                   # 总子块数
    chunksByType: Dict[str, int]       # 按类型分布 {"section_header": N, "paragraph": M, ...}
    chunks: List[FullTextChunk]        # 子块列表
    errorMessage: str                  # 错误信息（失败时）
    sourceFile: str                    # 来源 .mmd 文件名
```

### 5.5 FullTextFileInfo — 全文文件描述

```python
@dataclass
class FullTextFileInfo:
    fileId: str         # 文件内部 ID
    fileName: str       # 文件名 (如 xxx.mmd)
    url: str            # 相对路径 URL
    fileType: str       # 文件类型 (默认 "mmd")
    fileSize: int       # 文件大小 (字节)
    sysDataType: str    # 系统数据类型 (默认 "file")
```

---

## 6. 输出格式 (ChunkWriter)

### 6.1 JSONL 格式

每个子块作为**独立一行完整 JSON**，可直接逐行读取处理：

```
{"chunkId":"...", "sysId":"...", ...}
{"chunkId":"...", "sysId":"...", ...}
{"chunkId":"...", "sysId":"...", ...}
```

### 6.2 文件命名规则

| 场景 | 文件名 | 说明 |
|---|---|---|
| 单篇处理 | `chunks_{sysId}.jsonl` | 一篇文献的所有子块 |
| 批量处理 | `all_chunks.jsonl` | 多篇文献的所有子块合并 |
| 汇总统计 | `summary.json` | 批量处理的统计摘要 |

### 6.3 输出示例

**单行 JSONL 对象结构：**

```json
{
  "chunkId": "0d9a89249ec5284dcd93fecf1f963262_0",
  "sysId": "0d9a89249ec5284dcd93fecf1f963262",
  "chunkIndex": 0,
  "chunkType": "section_header",
  "redundantMeta": {
    "sysId": "0d9a89249ec5284dcd93fecf1f963262",
    "title": "The h-index in Australian Astronomy",
    "authors": ["Kevin A. Pimbblet"],
    "year": "2013",
    "journal": null,
    "doi": null,
    "keywords": [],
    "sourceFileName": "2de741fe-....mmd"
  },
  "content": "The Young Innovators Program at the Eshelman Institute...",
  "contentVector": [0.0, ...],
  "contentVectorDim": 2560,
  "paraStartIdx": 0,
  "paraEndIdx": 0,
  "charStart": 0,
  "charEnd": 4,
  "sectionPath": "The Young Innovators Program...",
  "level": 1,
  "parentParaId": "",
  "tokenCount": 29
}
```

### 6.4 summary.json 示例

```json
{
  "totalDocuments": 1,
  "successCount": 1,
  "failureCount": 0,
  "totalChunks": 67,
  "totalChunksByType": {
    "section_header": 19,
    "paragraph": 48
  },
  "details": [
    {
      "sysId": "0d9a89249ec5284dcd93fecf1f963262",
      "success": true,
      "totalChunks": 67,
      "chunksByType": {"section_header": 19, "paragraph": 48},
      "sourceFile": "2de741fe-....mmd",
      "errorMessage": ""
    }
  ]
}
```

---

## 7. 运行方式 (run_demo.py)

### 7.1 启动命令

```powershell
cd d:\work\NSTL\vector_library_sync\vector_sync_service
python app\demo\fulltextSplitDemo\run_demo.py
```

### 7.2 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TARGET_SYS_ID` | `"0d9a89249ec5284dcd93fecf1f963262"` | 目标文献 ID |
| `MILVUS_URI` | `"http://127.0.0.1:19530"` | Milvus 服务地址 |
| `MILVUS_COLLECTION` | `"scientific_documents_v30"` | Milvus Collection 名称 |
| `DOWNLOAD_BASE_URL` | `"https://static.nstl.gov.cn/"` | .mmd 文件下载 CDN 前缀 |
| `max_chars` | `1500` | 固定切分最大字符数 |
| `overlap_chars` | `200` | 相邻块重叠字符数 |
| `vector_dim` | `2560` | 向量维度（占位） |

### 7.3 目录结构

```
fulltextSplitDemo/
├── __init__.py              ← 模块导出
├── models.py                ← 数据模型
├── mmd_parser.py            ← AST 解析器
├── text_splitter.py         ← 三级切分器
├── pipeline.py              ← 流水线编排
├── chunk_writer.py          ← JSONL 写入器
├── run_demo.py              ← 测试入口
├── sample_json/
│   └── sample.json          ← 本地元数据样本
├── mmd/                     ← .mmd 下载缓存目录
└── output/                  ← 输出目录
    ├── chunks_{sysId}.jsonl ← 单篇切分结果
    ├── all_chunks.jsonl     ← 批量合并结果
    └── summary.json         ← 汇总统计
```

### 7.4 预期输出示例

```
======================================================================
  全文数据三级切分 Demo (Milvus + 本地回退)
======================================================================

[Config]
  Target sysId: 0d9a89249ec5284dcd93fecf1f963262
  Milvus: http://127.0.0.1:19530 / scientific_documents_v30
  Download URL: https://static.nstl.gov.cn/
  MMD cache: ...\fulltextSplitDemo\mmd
  Local base: ...\sample_json

[Step 1] Probing Milvus ...
[Milvus] Record found! Fields (82): [...]

[Step 2] Running fulltext split pipeline ...
[Pipeline] Milvus connected: http://127.0.0.1:19530
[Pipeline] Found in Milvus: sysId=0d9a89249ec5284dcd93fecf1f963262
[Pipeline] Fulltext file: 2de741fe-....mmd
[Pipeline] Loaded MMD: ...\xxx.mmd (32782 chars)
[Pipeline] Chunks saved (JSONL, 67 lines): ...\chunks_0d9a89249ec5284dcd93fecf1f963262.jsonl

======================================================================
  Result
======================================================================

  Status: SUCCESS
  Total chunks: 67
  By type: {'section_header': 19, 'paragraph': 48}
  Source: 2de741fe-....mmd

  Output files:
    chunks_0d9a89249ec5284dcd93fecf1f963262.jsonl (67 lines, 67.4 KB)

======================================================================
  Done!
======================================================================
```
