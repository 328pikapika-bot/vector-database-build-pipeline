# 2026-10-20 验证记录

## 本轮本地验收结论（后续结果，以此节为准）

本地 100 条真实增量验收通过。用户确认后已上传正式镜像 `v0.2.0`，
231 容器只读检查也通过；正式 Stack 留给用户在 Portainer 启动，尚未持续正式消费。

- 真实 Topic：`VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS`；独立采样消费者
  `assign` 读取，不提交正式组 `nstl_metadata_incremental_sync` 位点。测试前后正式组
  committed offsets 均为 `[]`，未改变；Topic 当时高水位 5,767。
- 从真实 Kafka 消息选 100 个不同 ID，查询实时 Solr、生成向量并写入
  `scientific_documents_v2`：新增 **40**、更新 **60**、失败 **0**。
  有效处理耗时 **60.28 秒**，含审计、最终 flush 的验收总耗时 **71.58 秒**。
  本次增量包含向量重算与逐条 Strong 审计，耗时口径不同于存量仅补字段测试。
- Strong 逻辑记录数 **52,978,029 → 52,978,069**，正好增加 40；
  100 条逐条读回通过，未提交字段变化 0，30 条只读对照变化 0，41 字段 Schema 不变。
  无删除、重建集合、修改 Schema 或重建索引操作。
- 初次 `state_02/summary.json` 的 `passed=false` 原因是索引 hash 包含实时的
  `total_rows/indexed_rows/pending_index_rows/state`。原报告保留。补充核验将当前
  **18 个索引定义**与留存的存量基线逐项比较，全部一致；基线集合 ID 和 Schema hash
  也与测试前相同。该证据来自较早留存基线，没有伪造本次测试前的索引定义快照。
- 生产 MongoDB `nstl_vector_sync_scientific_prod.metadata_incremental_sync_failures`
  连接、实际写入、按 `_id` 回读、重复投递去重均通过。仅生成 **1 条合成验证记录**，
  带 `validation_test=true,status=validation_test`；没有业务失败。记录保留供核查，
  未写入旧 `doc_sync_failures`，没有把真实 ID 伪造为失败。
- MongoDB 服务端 3.2.5，固定兼容且实测成功的 PyMongo 3.13.0。
- 自动测试 **36 passed**，覆盖完整 Worker 循环中缺源、无效消息、三次重试耗尽后
  继续处理后续消息、Mongo 离线继续处理、失败补同步、断点原子性、实际日期日报、
  安全暂停重启不能绕过、局部更新和索引审计。
- 此前 Solr HTTP 503 是本机环境代理误转发内网请求；已对 Solr 和向量 HTTP Session
  设置 `trust_env=False`。真实 100 条端到端查询成功。
- 同时观察到另一项存量补齐任务运行，本次采样避开其当时活跃分片及后续 16 个分片。
  没有停止、重启或修改该任务；正式增量长期运行前仍需协调同 ID 的并发写入。

本轮文件位于 `out/metadata_incremental_acceptance_20261020/`：

- `state_02/final_verification.json`：最终通过结论。
- `state_02/summary.json`：初次报告，原样保留。
- `state_02/index_definition_verification.json`：索引定义补充核验及基线来源。
- `state_02/mongo_failure_delivery_verification.json`：生产 Mongo 失败链路验收。
- `state_02/before.json`、`write_intents.json`、`sample_ids.json`：固定白名单及写前审计。
- `state_02/daily_activity/2026-10-20.json`：处理日期、首末时间、数量与耗时。
- `logs_02/incremental.log`：逐条处理日志。

本轮代码与旧镜像 `v0.1.1-rc1` 不同。正式镜像 `v0.2.0` 已包含本轮修复，
可复制 Stack 也已补全。普通业务失败记录后继续，
历史完整性、集合身份、状态存储或新记录写入结果不确定时仍安全暂停，保护历史库。

## v0.2.0 发布验证

- 远程构建及上传完成，固定摘要：
  `sha256:fea6c846c81a4b04562294515a4c9e8f1201574cfed2e597eef7ea843b11a8c7`。
- Linux 镜像内依赖导入和 `pip check` 通过；默认 `SYNC_MODE=check`，41 字段、
  3 个向量、Topic 和消费组正确，源码逐文件 SHA-256 与本地验收版本一致。
  Linux 单实例锁拒绝第二实例、关闭后可重新取得锁；上海时区偏移为 28800 秒。
- 镜像已预拉到 231，独立凭据卷已预置。临时检查容器以 `check` 模式启动，
  连通 Kafka、Solr、Milvus、生产 MongoDB 和向量接口；核对集合身份、41 字段、
  18 索引，向量维度 2560，健康状态 `check_passed`。
- 231 本轮发布检查 Milvus 写入 0、Kafka 订阅 0；检查容器已移除，
  没有部署正式增量 Stack，没有修改其他业务容器或存量回填任务。
- `docker compose -f portainer-stack-231.yml config --quiet` 通过。
- 发布报告在 `out/metadata_incremental_release_v0.2.0/`：`release_report.json`、
  `image_runtime_check.json`、`prepared_231.json`、`container_check_231.json`。
  报告和镜像不含连接凭据。启动说明见 [RELEASE.md](RELEASE.md)。

以下为此前候选版本的历史验证记录，其“仍待验收”状态已由上面的本轮结果更新。

## 已完成

- 原候选镜像阶段本地 `python -m pytest tests -q`：13 项通过。覆盖字段映射、`entityType`
  数组首元素、超长拒绝、缺源不写、已有记录局部更新、新记录 41 字段插入、Kafka
  offset 先持久化后提交、重启断点、单实例锁、暂时故障排队、写入超时读回和历史字段异常暂停。
- 231 测试 Milvus 只读 Schema 检查通过：集合 ID
  `467035880531670094`，41 个字段，18 个索引；单条 Strong 读取返回 41 字段和
  三个完整向量。
- 231 独立探测集合 `incremental_vector_probe_20261020_082749_73cb77`
  验证通过：只局部更新 `title_embed`，其他两个向量和历史标量保持不变。
  探测集合保留，未删除或重建 `scientific_documents_v2`。
- 231 目标集合 Strong `count(*)`：**52,978,029**。本项目没有对该集合启动真实写入。
- Kafka 旧候选 Topic `VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS_V2`：
  1 分区、3 副本，变更名称前 offset 水位 `(0, 0)`。`_V1` 只有 1 副本；
  两个旧 Topic 在确认无现存消息、无活跃消费者后于 2026-10-20 删除。
- 常态化增量 Topic `VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS` 已创建并核实：
  1 分区、3 副本，创建后 offset 水位 `(0, 0)`。独立消费组为
  `nstl_metadata_incremental_sync`；尚未启动消息消费或 Milvus 写入。
- Docker 候选镜像已构建、运行时导入检查通过并推送：
  `docker-hosted.example.invalid/nstl/vector-sync/nstl-metadata-incremental-sync:v0.1.0-rc3`
  摘要 `sha256:ffccfc32ba82ed54d865fb5547e99609352c8e13e7e1b5710b26190f8e6616fb`。
  231 Stack 已固定该摘要；镜像内默认 Topic 与消费组已核对。

## 仍待验收

- 本地对既有 Solr `paper6.6/select` 的只读请求返回 HTTP 503，因此
  Kafka/Solr/Milvus 的完整 `check` 预检查尚未通过。Worker 遇到该暂时故障会
  保持运行并退避等待，不能称端到端已验证。
- 原 `entityType` 存量任务后来已完成 5,414,828 条扫描并通过核验，其中 2 条失败待处理。增量 Worker 还没有
  对 231 测试集合执行首条真实插入或更新，也没有启动长期 `run` 模式。
- 在存量补齐程序、原数据同步程序与本增量程序同写一个集合前，需完成同 ID
  写入顺序和新 ID 导致的存量记录数基线规则协调。三套独立生产 Milvus 的
  跨库原子要求仍是架构限制，当前镜像只允许单目标。

## V4 转换调整（本次）

- 41 字段契约未变。17 个补充字段按 `V4_FIELD_RULES.md` 映射；结构化 `hasSource`、
  `hasFile` 和四个 `_str` 改为读取对应 `stringOM_*`，其余标量按 V4 首项和语言前缀规则。
- 本地 `python -m pytest tests -q`：23 项通过；含 17 字段对照、空首项、字段缺失保留、
  结构化文件格式、逻辑删除标记只作字段更新、原字段不进入局部更新 payload、
  超长拒绝和写后历史字段核验。
- 2026-10-20 再次只读请求 `paper6.6/select` 返回 HTTP 503；尚未完成真实 Solr 样本
  和 Kafka 端到端验证，不应将此候选版本切换成 `run`。
- 新候选镜像 `v0.1.1-rc1` 构建、容器内导入及映射冒烟检查通过，已推送；
  摘要为 `sha256:7d271f26120fc35a5ef8170777c51ed6521738579cb9b67fd8787891a9d9bf88`。
  231 Stack 文件已固定该摘要，仍默认 `SYNC_MODE=check`。
