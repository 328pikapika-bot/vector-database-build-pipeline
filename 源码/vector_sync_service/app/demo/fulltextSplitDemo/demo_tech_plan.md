# 全文数据三级切分 Demo 技术方案

> **创建日期**: 2026-05-25
> **基于**: NSTL向量库同步_分阶段实现方案_v4, M1-2 阶段
> **Demo 目录**: `vector_sync_service/app/demo/fulltextSplitDemo`

---

## 1. 概述

本 Demo 验证文献数据入库后，对全文字段 `hasFullTextFile` 进行检测、下载、解析、三级切分、向量化和存储的完整流程。

### 1.1 核心流程

```
元数据集合查询 -> 检测 hasFullTextFile -> 下载 .mmd 文件 -> Markdown 解析
    -> 三级切分(章节/段落/固定长度) -> 向量化(模拟) -> 存储到子块集合 -> 清理附件
```

### 1.2 设计的 Chunk 元数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunkId` | str | 子块唯一ID: `{sysId}_{chunkIndex}` |
| `sysId` | str | 所属文献ID |
| `chunkIndex` | int | 块序号(从0自增) |
| `chunkType` | str | 块类型: `section_header` / `paragraph` / `fixed_chunk` |
| `redundantMeta` | dict | 冗余元数据(标题/作者/期刊/年份等) |
| `content` | str | 子块文本内容 |
| `contentVector` | list[float] | 内容向量化数据(测试用零向量) |
| `paraStartIdx` | int | 段落起始字符偏移 |
| `paraEndIdx` | int | 段落结束字符偏移 |
| `sectionPath` | str | 章节层级路径 |
| `level` | int | 章节层级深度(1/2/3...) |
| `tokenCount` | int | 近似token数 |
| `parentParaId` | str | 上级段落索引(固定长度chunk所属的段落) |

---

## 2. 三级切分策略

### 2.1 第一级: 章节切分 (Section Split)
- 基于 Markdown `#` 标题层级解析
- 每个 `#` 标题及其下属内容构成一个 section chunk
- 类型: `section_header`
- 记录 `sectionPath` (如 `1.Introduction`) 和 `level`

### 2.2 第二级: 段落切分 (Paragraph Split)
- 在每个 section 内按空行分割为段落
- 每个段落为独立 chunk
- 类型: `paragraph`
- 记录 `paraStartIdx` / `paraEndIdx`

### 2.3 第三级: 固定长度切分 (Fixed-length Split)
- 对超过 `CHUNK_MAX_CHARS(1500)` 的段落进行固定长度再切分
- 重叠区域: `CHUNK_OVERLAP_CHARS(200)`
- 类型: `fixed_chunk`
- 记录 `parentParaId` 指向原段落

---

## 3. 数据处理流程

### 3.1 多文档处理
`hasFullTextFile` 字段是对象数组，可能包含多个 .mmd 文档。Demo 支持遍历处理所有文档。

### 3.2 数据来源
Demo 从元数据集合（本地 sample JSON）通过 `sysId` 获取数据，解析 `op.hasFullTextFile` 判断是否存在全文。

### 3.3 模拟向量化
因 Demo 环境可能无 Embedding API，使用零向量(2560维)模拟向量化结果。

---

## 4. 文件结构

```
fulltextSplitDemo/
├── demo_tech_plan.md          # 本文档
├── __init__.py                 # 模块入口
├── models.py                   # Demo用的子块数据模型
├── mmd_parser.py              # MMD/Markdown 文档解析器
├── text_splitter.py           # 三级文本切分器
├── chunk_writer.py            # 子块存储(JSON文件)
├── pipeline.py                # 主流程编排器
├── run_demo.py                # 测试入口脚本
└── output/                    # 输出目录
    └── chunks_*.json          # 切分结果
```

---

## 5. 测试数据

| 数据来源 | 路径 | 说明 |
|----------|------|------|
| 元数据样本 | `sample_json/sample.json` | 包含 hasFullTextFile 的文献数据 |
| 全文 .mmd | `sample_json/{sysId}/{uuid}.mmd` | 已下载的全文 MMD 文件 |
| 测试 sysId | `fffffdff137abc8dd13e929b4bde50c7` | 来自 sample.json |

---

## 6. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_MAX_CHARS` | 1500 | 固定长度块最大字符数 |
| `CHUNK_OVERLAP_CHARS` | 200 | 固定长度块重叠字符数 |
| `VECTOR_DIM` | 2560 | 向量维度 |
| `FULLTEXT_BASE_DIR` | `sample_json/` | 全文文件基础目录 |