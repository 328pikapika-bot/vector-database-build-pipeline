# NSTL 文献元数据每日增量同步

独立项目和 Stack 名称：`nstl-metadata-incremental-sync`。Kafka Topic：
`VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS`。消息体只放 32 位十六进制文献 ID；
Topic 为一个分区，沿用现有 Kafka 集群和认证，消费组为
`nstl_metadata_incremental_sync`。当前 Stack **只指向 231 测试 Milvus**。

## 数据路径

1. Kafka 单分区顺序消费；消息先以 Topic/partition/offset 写入 `checkpoint.db`，
   SQLite 事务确认后再提交 Kafka offset。重启会先处理本地未完成事件。
2. 逐 ID 查询当时的 Solr 文献，不判断 Solr 内容是否为最终版本。Solr 查不到时
   记录 `source_missing`，不调用 Milvus 写入，更不会删除记录。
3. 按 41 字段固定 Schema 转换：原 24 字段、原 16 个补充字段、`entityType`。
   17 个补充字段的 Solr 来源与转换规则见 [V4_FIELD_RULES.md](V4_FIELD_RULES.md)。
   `entityType` 严格取 `stringIMS_entityType` 的数组首元素，再按 V4 去语言前缀。
   四个 `_str` 从 Solr 对应的结构化原值生成；Solr 本次缺失的字段不清空已有值。
   字段超长则记录失败，不截断。
4. 以现有 `qwen3-embed` 规则生成标题、摘要、关键词三个 2560 维向量；缺少文本时
   新记录使用现有程序的非零占位向量，已有记录保留旧向量。暂时请求失败重试；
   同一事件达到 `MAX_ATTEMPTS`（默认 3）后持久化失败，继续处理其他消息。
5. Strong 读取目标 ID。存在时仅提交 `id`、本次明确取得的字段和重算的向量，
   `partial_update=True`；不存在时仅通过 `insert` 插入 41 字段新记录。写后 Strong
   读回并核查未在 payload 中的历史字段。代码没有删除集合、删除记录、清空索引的入口。

## 启动门槛

- `SYNC_MODE=check` 是默认值：连接 Kafka/Solr/Milvus、核对 Topic 和 41 字段 Schema，
  保持容器运行，但不消费消息、不写 Milvus。
- `SYNC_MODE=run` 还要求 `VECTOR_PARTIAL_UPDATE_VERIFIED=1`。该值只能在目标 Milvus
  的独立小集合运行 `tools/verify_vector_partial_update.py` 且三个向量局部更新核验通过后设置。
- 正式运行必须配置 `MONGO_URI`；数据库为 `nstl_vector_sync_scientific_prod`，
  专用失败集合为 `metadata_incremental_sync_failures`。不复用旧程序的 `doc_sync_failures`。
- 231 的 `entityType` 历史任务及后续 17 字段存量任务不能与本程序同时写同一 ID；
  现有原数据程序也不能与本程序并发插入同一 ID。先完成写入顺序协调和存量基线规则，
  再开启 `run`。仅部署 `check` 不会触碰 5,000 多万条历史记录。
- 41 字段不全、集合身份变化、历史未更新字段被改动、状态目录不能持久化等情况
  会进入安全暂停，进程保留但不继续写入；重启不会绕过已有 `safety_pause.json`。
  单条缺源、超长、转换失败及请求重试耗尽记录后继续处理其他消息。新记录 insert
  超时且无法确认是否成功也会安全暂停，防止盲目重试产生重复主键。

## 日报与失败记录

独立挂载卷包含：

```text
/app/state/checkpoint.db               完整逐消息断点和失败记录
/app/state/preflight.json               只读检查结果
/app/state/progress.json                每 10 秒更新的状态和计数
/app/state/safety_pause.json            安全暂停原因（若发生）
/app/state/daily/YYYY-MM-DD.json        按消息收到的上海日期统计
/app/state/daily_activity/YYYY-MM-DD.json 按实际处理的上海日期统计
/app/state/failures_recent.json         最近 100 条失败的只读摘要
/app/logs/incremental.log              每条完成/失败日志，按日轮换
```

日报包含首次收到时间、首次开始处理时间、最后完成时间、处理跨度、接收数、新增数、
更新数、无变化数、Solr 缺源数、失败数和待处理数。`daily_activity` 另记实际处理首末时间、
有效处理秒数、重试数；跨上海午夜的处理耗时拆分到对应日期。首末时间的跨度可能包含
空闲，不能当作纯处理耗时。上游只发 ID，不发“当日结束”消息，
因此空闲 300 秒后写出的窗口报告是**观测到的空闲**，不代表上游正式宣布结束。
`/health` 与 `/status` 仅在容器内部 8099 端口提供；Stack 不映射宿主机端口。

失败与 SQLite 断点在同一事务保存，再通过本地 outbox 幂等写入生产 MongoDB。
MongoDB 暂时不可用不影响后续正常消息，待同步队列保留在状态卷内，恢复后补写。
进度文件包含 `mongo_status`、`mongo_pending`；Mongo 失败记录带文献 ID、Topic、
partition、offset、目标身份、阶段、原因、尝试次数和时间，不保存向量。
查询业务待处理记录使用 `{status: "pending"}`；验收用记录单独标记
`validation_test=true,status="validation_test"`。全部失败记录保存在 SQLite 和 Mongo，
`failures_recent.json` 仅展示最近 100 条。日志按日轮转保留 30 份。

生产 MongoDB 实测为 3.2.5，使用已经验证的 `pymongo==3.13.0`，不能自动升级为不支持
该服务端版本的 PyMongo 4。内网 Solr/向量 HTTP 连接禁用桌面环境代理，避免请求误送代理
产生 HTTP 503。

## 构建和切换

本轮本地验收已完成，详见 [VALIDATION.md](VALIDATION.md)。用户确认后已构建并上传
正式镜像 `v0.2.0`，包含 Mongo 失败补同步、日报和持续运行修复；
[portainer-stack-231.yml](portainer-stack-231.yml) 已固定镜像摘要并显式设置 `SYNC_MODE=run`。
231 已预拉镜像并准备只读凭据卷，实际容器网络下的只读预检查通过。
正式 Worker 尚未替用户启动。复制配置和运行说明见 [RELEASE.md](RELEASE.md)。
如需在修改代码后运行自动测试：

```powershell
python -m pytest tests -q
```

`tools/verify_vector_partial_update.py` 仅操作独立探测集合；
`tools/verify_live_100.py` 则是真实写入 231 测试集合的有限验收工具，只能按固定的
100-ID 白名单执行。本次已运行完成，不要为查看报告再次运行。它使用独立采样组和
`assign`，不提交正式组 offset；测试结束不会继续消费。
正式部署需保留状态及日志卷。修改环境变量需要更新 Stack，单纯 Restart 不装入新配置。

Topic 创建命令（需要 Kafka 管理权限）使用同一镜像和相同 Kafka 环境变量，执行
`python -m incremental_sync create-topic`。它只创建上文命名的单分区、三副本 Topic；
已存在时直接返回。正常 Worker 不自动建 Topic。

## 三套生产 Milvus 的边界

当前 Worker 只实现单目标。三套独立 Milvus 无跨库原子事务，无法满足“任一失败时
另两套绝对没有写入”的瞬时一致性要求。不能把本 Stack 填三个地址后直接用于生产。
生产三套库的历史复制、目标写入协调和故障语义见 [DESIGN.md](DESIGN.md)。
