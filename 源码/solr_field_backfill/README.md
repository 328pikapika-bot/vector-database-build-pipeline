# Solr 补字段容器程序

> 2026-10-20：本地源码已接入 17 字段参考解析规则，尚未启动新的全量任务。原 16
> 字段本地 Worker 和 231 Worker 均保持停止。旧状态目录绑定 16 字段映射，不能直接
> 用新程序复用；后续运行必须使用独立的新状态目录并保留旧目录。操作说明见
> `../out/solr_backfill_local_20261019/README.md`。当前源码使用 PyMongo 3.13.0。

这是一个独立的 Solr → Milvus 补字段 Worker，目标是测试集合
`scientific_documents_v2`。目标是 17 个可空 `VarChar` 补充字段；缺失字段才加到
现有集合，并通过 Milvus `partial_update=True` 仅写入 `id` 与这 17 个字段。

程序没有删除、重建集合、删除字段、重建向量索引或完整记录覆盖写入的代码。原有
24 个字段（其中包含 `title_embed`、`abstract_embed`、`keyword_embed` 三个 2560 维
向量字段）保留；预检查强制校验主键和三个向量字段。每次写入前还会校验 payload
只能包含 `id` 和新增字段，并用 Strong 读取确认这些 ID 仍然存在。

## 模式

容器默认使用 `BACKFILL_MODE=check`。该模式只连接 Milvus/Solr、校验集合 Schema、
记录数，然后退出，不增加字段也不写记录；只读模式不能证明写权限。只有显式设置
`BACKFILL_MODE=run` 才会增加缺失字段并开始回填；run 仍然先执行完整预检查。

状态文件位于挂载的 `/app/state`：

* `checkpoint.db`：256 个两位十六进制 ID 前缀分片状态。
* `progress.json`、`summary.json`：进度和最终汇总。
* `failures.jsonl`：Solr 缺失、超长值、请求或 Milvus 写入失败的 ID。

收到 SIGTERM/SIGINT 时，程序停止领取新分片并保存状态。重新启动会跳过
`completed` 分片，未完成分片可安全重试。run 模式会对少量原字段和向量做前后哈希
校验，发现历史值变化则立即报错。

## 配置

默认配置对应测试环境：

```text
MILVUS_URI=http://127.0.0.1:19533
MILVUS_DB_NAME=default
MILVUS_COLLECTION=scientific_documents_v2
SOLR_URL=http://solr-bak1.example.invalid:8984/solr/paper6.6/select
BACKFILL_MODE=check
BACKFILL_CONCURRENCY=2
BACKFILL_BATCH_SIZE=200
SOLR_TIMEOUT_SEC=60
MILVUS_TIMEOUT_SEC=120
MAX_RETRIES=3
FLUSH_EVERY_BATCHES=20
```

解析规则位于 `production_parser.py`，参照
`docs/元数据生产库字段逻辑/build_milvus_db_all_v4/json_processor.py`：

* `hasSource` 从 `stringOM_hasSource` 的结构化 `title` 提取，多项用 `;` 连接；
  `hasFile` 从 `stringOM_hasFile` 的 `fileType` 提取，多项用 `;` 连接。只有明确的
  `fileType` 或单独的文件类型词才写入 `hasFile`，不把 URL 当作文件类型。
* `volume`、`issue`、`labelEvaluateGrade`、`hasHoldinglibraryCode`、`entityType`
  取 Solr 数组首项，再按首个 `|` 去语言前缀；`startPage`、`endPage`、
  `pageCount`、`createTime`、`publisher`、`sysProcessingUnit` 仅取首项。
* 四个 `_str` 字段分别将 Solr 的 `stringOM_hasAuthor`、`stringOM_hasSource`、
  `stringOM_hasFile`、`stringOM_hasCreateOrganization` 原始数组序列化为 JSON 字符串。
  Solr 缺值时保持 `null`。所有字段超出当前集合 Schema 长度均记录失败，不截断。

参考程序从本地 JSON 文件路径区分 `data_BookMatrix` 等来源类型；Solr 查询没有该
文件路径，因此本 Worker 无法复现依赖文件路径的 `labelEvaluateGrade` 来源类型门控。
此处仅在 Solr 明确返回该字段时解析它。已补的非 `null` 值继续保留，规则调整不会
自动覆盖它们。持续运行 Worker 遇到“Solr 现值与已补非空值不同”时，保留该 ID
全部原值，将其记入独立失败记录并继续处理同批其他 ID；Solr 字段缺值不算冲突。
这一类记录须在扫描完成后单独核查，不能将其算作“已按新规则补齐”。

已处理记录的只读抽样比对可运行：

```powershell
python solr_field_backfill/scripts/audit_existing_parser.py `
  --checkpoint out/solr_backfill_local_20261019/state/checkpoint.db `
  --sample-size 2000 `
  --output out/solr_backfill_v4_canary_20261020/existing_2000_readonly_audit.json
```

本次 2,000 条旧断点样本中，1,803 条 `hasAuthor_str`、1,018 条 `hasSource_str`、
139 条 `hasFile_str`、700 条 `hasCreateOrganization_str` 仍有可填来源；2 条已有
`publisher="Zenodo"`，而参考规则解析当前 Solr 值为 `"cs|Zenodo"`。这些旧值
未改写。抽样仅用于发现差异，不能推断 5,414,828 条均已按参考规则处理。

2026-10-20 本地实测：100 条新 ID 写入、20 条只读对照，17 字段值逐条核对通过；
120 条记录的原 24 字段（含 3 个完整向量）哈希未变，Strong 逻辑记录数保持
52,978,029，Schema、索引和集合 ID 未变。审计见
`../out/solr_backfill_v4_canary_20261020/state/summary.json`。该样本不代表全量已经
补齐；全量任务尚未启动。

## 本地测试

在仓库根目录执行：

```powershell
python -m pytest -q solr_field_backfill/tests
python -m py_compile solr_field_backfill/backfill.py
```

只读验证 100 或 1000 条：

```powershell
python solr_field_backfill/scripts/validate_sample.py --count 100
python solr_field_backfill/scripts/validate_sample.py --count 1000
```

## 限定样本的真实回填

`BACKFILL_MAX_RECORDS` 设置为 1–10000 时走独立的限定样本路径。先选择该数量的
已有记录和另外 20 条只读对照；17 个目标字段须全部为 `null`，已有值（包括空字符串）
的记录跳过。写入前保存固定 ID 白名单、集合身份、Strong 逻辑记录数、Schema、索引
定义和全部样本及对照的原字段与完整向量哈希。先写一批，核对原字段、向量和新增
字段，通过后才继续，每次最多两批在途。最后 flush 并逐条审计。样本目录已有
`sample_ids.json` 时拒绝自动重跑；不要把限定测试的状态目录用于全量任务。

限定 10,000 条测试可使用以下配置，程序默认仍为 `check`：

```text
BACKFILL_MODE=run
BACKFILL_MAX_RECORDS=10000
BACKFILL_SAMPLE_START_PREFIX=ff
BACKFILL_CONCURRENCY=2
BACKFILL_BATCH_SIZE=500
FLUSH_EVERY_BATCHES=8
BACKFILL_EXCLUDE_SAMPLE_MANIFEST=<上次测试 state/sample_ids.json 的绝对路径>
BACKFILL_STATE_DIR=<本次全新目录>/state
BACKFILL_LOG_DIR=<本次全新目录>/logs
```

排除清单必须与当前 Milvus 地址、数据库和集合一致，同时排除其中的 `ids` 和
`control_ids`；文件缺失、格式错误或目标不一致时停止。限定样本写入前还会检查目标
字段是否被其他任务填入，避免覆盖已补的值。Solr 缺失、请求失败、超长或审计不一致
会停止新批次，保存失败报告，不自动回滚或更换样本继续。
`BACKFILL_SAMPLE_START_PREFIX` 仅用于限定样本，从指定十六进制前缀开始选样，
避免重新扫描已经补过的前缀；全量模式不使用该配置。

每个工作线程复用独立 Solr HTTP 连接，程序退出时关闭。`summary.json` 保留旧的
`solr_seconds` 和 `write_seconds` 指标，新增每批 Solr 查询、转换、审计文件写入、Strong
检查、partial upsert、重试等待、排队耗时，以及独立的 flush 耗时。并发步骤的耗时
合计不是墙钟时间。`canary_audit.json` 记录首批核验结果。

`pipeline_rows_per_sec` 只计算首批核验后剩余记录，包含定期 flush，排除最终 flush；
`pipeline_rows_per_sec_including_final_flush` 包含最终 flush；
`backfill_rows_per_sec_including_final_flush` 计算全部写入，包含最终 flush、扣除首批
审计；`total_seconds_including_audits` 包含选样和全部前后审计。逻辑记录数有任何变化
都不直接判定通过。Milvus stats 的 `row_count` 可能包含旧版本，不能代替 Strong
`count(*)` 判断记录是否增加或丢失。

## 当前版本 v1.2.0：持续运行

231 的 Stack 名称为 `nstl-solr-field-backfill`，继续使用原状态卷和日志卷。
配置文件为 `portainer-stack-solr-field-backfill.yml`，发布时固定镜像摘要。
镜像默认 `check`；正式 Stack 显式设置 `run`。双并发、每批 500 条、Solr 提前查询、
每 8 批 flush，flush 时等待窗口内写入结束。231 现有网络是 `milvus`。
Worker 不监听 HTTP，因此无需端口映射。

### 历史数据保护

每次只提交 `id` 和 17 个补充字段，强制 `partial_update=True`，写入前 Strong 读取
确认记录存在。原有 24 字段和 3 个完整向量不进入 payload。没有删除、重建集合、
完整覆盖或重建索引的调用。目标字段已有任何非 null 内容（包括空字符串）均保留。
Solr 缺值保留 null，四个 `_str` 字段仅从上述原始结构化来源序列化。

启动时复核原集合身份、Strong 逻辑记录数、Schema、索引及历史审计基线。新版本首批
500 条逐条核对全部历史字段、完整向量和期望新增值，通过后继续全量。其后每分片及
结束抽样审计；这不是对 5,297 万条记录逐条读取原字段审计。每批 Strong 检查与 upsert
是两个请求，运行期间应避免其他程序删除这些记录或修改这 17 个目标字段。

### 异常行为

| 情况 | 行为 |
| --- | --- |
| 单条 Solr 缺源 | 记录 ID，同批正常记录继续写入，缺源记录不写入 |
| 单条超长或转换失败 | 记录 ID、字段及原因，不截断，继续其他记录 |
| Solr/Milvus 网络故障 | 请求重试最多 3 次，再按 30 至 300 秒退避等待恢复，不批量登记后续 ID 为失败 |
| upsert 超时 | 持久化 intended payload，Strong 读取后再决定重试；实际已成功的不重写、不重复累计 |
| MongoDB 不可用 | 本地 SQLite outbox 保留，后台每 15 秒补同步，主扫描继续 |
| 状态存储不可用 | 暂停新批次和新写入，持续检查存储恢复 |
| 历史值、身份、字段白名单异常 | 保存 `safety_pause.json`，暂停写入并保持进程运行，重启不能绕过 |
| 全量扫描完成 | 容器进入待命，存在失败时显示 `completed_with_failures` |

安全暂停必须人工排查；确认历史基线和异常原因已解决后，先 Stop Worker，归档
`safety_pause.json`（保留证据），再将原文件移出状态目录后 Start。不能把删除暂停标记
当作常规故障恢复。网络故障自动恢复，无需操作安全锁。

### 断点、计数与 MongoDB

SQLite 使用 WAL 和 FULL 同步。原表和原报告保留，新建 `bf_*` 表并事务迁移。
旧版未完成 `00` 分片缺少逐条断点，重新扫描核对，已有值保留，内容相同则跳过写入。
新版本按排序后的固定 ID manifest 保存每批状态，不依赖查询返回顺序。
`pending → inflight → acked → done`：写入意图与失败事件先事务落盘；成功回执后 acked；
flush 确认后才 done。恢复时对 inflight Strong 核对，done 跳过。意外停止不自动回滚。

失败写入现有 MongoDB 数据库 `nstl_vector_sync_scientific_prod` 的独立集合
`solr_field_backfill_failures`，不进入 `doc_sync_failures`。确定性 `_id` 去重、无 TTL、
不保存向量，含任务、目标身份、文献 ID、阶段、原因、时间、尝试次数和处理状态。
`MONGO_URI` 通过 Portainer Stack 环境变量注入，不放在配置文件或日志中。

| 进度项 | 含义 |
| --- | --- |
| `scanned` | 已枚举并持久化到固定 manifest 的 ID 数 |
| `processed` | 已完成且 flush 确认的记录数 |
| `updated` | 其中实际补充了至少一个字段的记录数 |
| `unchanged` | 查到源，但目标值无需变化的记录数 |
| `failed` | 已完成批次中失败待处理的记录数 |
| `acknowledged_pending_flush` | 已收到回执、尚未 flush 确认的处理数 |
| `registered_failures` | 本地已持久化的失败事件数，可能早于批次完成 |
| `mongo_pending` | 仍待同步至 MongoDB 的失败事件数 |

完成批次必须满足 `processed = updated + unchanged + failed`。扫描最终对账为原逻辑
记录数；有失败只表示扫描完成，不能宣称全部字段补齐。新增字段 null 数量分别保存在
每个批次的 `result.null_counts` 中，统计的是 Solr 转换结果。

### 运维

固定卷 `nstl-solr-field-backfill-state:/app/state` 和
`nstl-solr-field-backfill-logs:/app/logs`，不可删除或覆盖代码目录。单实例 OS 文件锁；
231 使用独立 Compose，`restart: unless-stopped` 支持意外退出和服务器重启后恢复。
完成待命和安全暂停不退出；手工 Stop 通过信号中断等待，保留所有断点。

`progress.json`/`heartbeat.json` 每 15 秒更新；日志每 30 秒输出状态：`running`、
`waiting_dependencies`、`waiting_storage`、`safety_paused`、`completed`、
`completed_with_failures`。健康检查同时检查 PID 和最近 90 秒心跳，健康只说明进程
有响应，是否推进需要看状态和 `processed` 的变化。

保留 `full_baseline.json`、`shard_XX_original_hashes.json` 与旧报告。新增
`resilient_canary.json` 保存首批原值哈希和写前期望值，
`resilient_canary_report.json` 保存核验结果。`summary.json` 在扫描完成后生成，
`scan_complete` 与 `all_records_enriched` 分别表示扫描和无失败完成。
旧版停止报告仍保留至扫描结束，不应把旧版 summary 当作当前运行状态。

### 验证和构建

```powershell
python -m pytest -q solr_field_backfill/tests
docker -H tcp://127.0.0.1:2375 build --progress=plain -t docker-hosted.example.invalid/nstl/vector-sync/solr-field-backfill:v1.2.0 solr_field_backfill
docker -H tcp://127.0.0.1:2375 push docker-hosted.example.invalid/nstl/vector-sync/solr-field-backfill:v1.2.0
```

发布证据目录：`../out/solr_backfill_release_v1.2.0`。升级前先只读核验并备份旧状态，
同一 Stack 仅替换 Worker，首批通过后继续，观察至少 10 分钟且处理至少 10,000 条。
595 条/秒来自旧版小样本流水线测试，当前包含持久化恢复与分片审计，须重新测量，
不预先承诺全量完成时间。

### Milvus 底层阻塞时的处理（2026-10-19）

保持每批 500 条、双并发、每 8 批 flush。任一 Milvus 请求暂时失败后，整个
Worker 统一停止提交新请求，已有 RPC 等待结束。每次仅一个线程执行只读 Strong
恢复探测，探测超时上限 15 秒；失败后按 30～300 秒退避。探测成功后继续原批次，
超时批次仍需 Strong 核对后才允许再次局部更新。不会用探测写入或 Eventually
读替代这些保护。默认 `MILVUS_SLOW_REQUEST_SEC=15`，成功但缓慢的读、写或
flush 同样触发退避；设为 0 只关闭慢请求触发，超时熔断仍保留。

状态新增 `milvus_backpressure`、`active_milvus_rpcs`、`last_milvus_rpc`、
`processed_per_second_recent`、`seconds_without_confirmed_progress`。运行时间和
平均速度随心跳重新计算；当前 60 秒没有确认完成就显示近期吞吐为 0。
`scanned` 是已建立分片清单的数量，`processed` 才是 flush 后确认的数量。
启动时先展示持久化进度并进行只读 Strong 探测，通过原有身份、数量及历史哈希
校验后才恢复提交；不会因重启绕过审计。

Milvus 2.6.9 的 partial upsert 在服务端读取原记录并合并后写入消息日志，物理
写入量大于客户端 17 个字段的 payload。RocksMQ 的全范围压缩可能涉及大量旧
日志文件；减小单批不能取消该压缩。程序熔断可降低重试压力，不能宣称已解除
数据库底层 write stall。服务端维护必须单独核实影响与授权，禁止手动删除
RocksDB、SST 或 WAL 文件。

持续回填可设置 `BACKFILL_MAX_WRITE_ROWS_PER_SEC=350`，两个写线程共享批次节奏，
500 条的批次至少间隔约 1.43 秒进入写前核对。停顿期间不积攒突发额度，重试也
遵守节奏；停止信号可中断等待。此配置限制提交压力，不保证完成速度，不替代
Strong 校验和全局退避。默认 0 保留原有不限速行为，本地恢复配置显式使用 350。

此次 RocksMQ 维护将现有容器基础配置 `rocksmq.compactionInterval` 从 86400
调整为 604800 秒，给全量回填留出避免每日全范围压缩的窗口。常规自动压缩、
消息确认和保留规则继续运行。这不是永久消除 RocksDB 写放大的修复，也不能
保证所有依赖永不超时。配置和断点均已备份；当前文件改动随同一个容器重启
保留，重新创建 Milvus 容器前须把该设置纳入其部署配置。
