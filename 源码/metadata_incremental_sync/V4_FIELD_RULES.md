# 17 个补充字段的 V4 规则对照

目标集合仍是 `default/scientific_documents_v2` 的固定 41 字段 Schema。此表只调整 Solr 到既有字段的取值逻辑，不采用 V4 的集合结构、字段长度或写入流程。参考：`docs/元数据生产库字段逻辑/build_milvus_db_all_v4/field_mapping.py` 与 `json_processor.py`。

| 目标字段 | Solr 来源 | 转换 |
|---|---|---|
| `hasSource` | `stringOM_hasSource` | 按 V4 解析每项 `title`，以 `;` 拼接；结构化来源完全缺失时兼容 `auto_stringITMS_hasSourcetitle` |
| `hasFile` | `stringOM_hasFile` | 按 V4 解析每项 `fileType`，以 `;` 拼接；兼容 `fileType≡fileType≡pdf` 嵌套格式 |
| `volume`、`issue` | `stringIMS_volume`、`stringIMS_issue` | 只取首项并去掉首个 `|` 之前的语言前缀 |
| `startPage`、`endPage`、`pageCount` | 各自的 `stringIS_*` | 只取首项，保留原文本 |
| `createTime` | `dateIS_createTime` | 只取首项，保留原文本 |
| `publisher` | `auto_stringITMS_publisher` | 只取首项，按 V4 保留语言前缀 |
| `labelEvaluateGrade` | `stringIMS_labelEvaluateGrade` | 只取首项并去语言前缀 |
| `sysProcessingUnit` | `stringIS_sysProcessingUnit` | 只取首项，保留原文本 |
| `hasHoldinglibraryCode` | `stringIMS_hasHoldinglibraryCode` | 只取首项并去语言前缀 |
| `entityType` | `stringIMS_entityType` | 严格取数组首项并去语言前缀；首项为空不使用第二项 |
| `hasAuthor_str`、`hasSource_str`、`hasFile_str`、`hasCreateOrganization_str` | 分别为 `stringOM_hasAuthor`、`stringOM_hasSource`、`stringOM_hasFile`、`stringOM_hasCreateOrganization` | V4 原始结构化值序列化：数组用保留中文的 JSON，单值用原文本 |

V4 的 `source_type` 来自输入文件名，Kafka 消息只有文献 ID，所以不能据文件名实施 Book、Report 等分支；`labelEvaluateGrade` 只在 Solr 明确返回时写入。Solr 某字段缺失或首项为空时不把空值写到已有 Milvus 记录。原 24 字段沿用现有集合的数组与三个向量契约；`hasSourceidSet` 增加了 V4 中 `stringIMS_idSet` 的缺值候选来源。

目标集合部分字段长度小于 V4；超出目标 Schema 限制的值会记录单条失败，不截断。已有记录只用 `partial_update=True` 提交明确变化的字段，写后 Strong 读取核对未提交的历史字段。Solr 缺源不执行删除。三套独立 Milvus 的跨库原子写入仍未实现，当前 Stack 只连接 231 单目标。
