# 231 增量程序 v0.3.0（批量版）

Stack 名称继续使用 `nstl-metadata-incremental-sync`。
完整配置：[portainer-stack-231.yml](portainer-stack-231.yml)，已补齐中文注释。
镜像已上传、预拉到 231，并在 231 现有网络和只读凭据下通过 check 预检：

```text
docker-hosted.example.invalid/nstl/vector-sync/nstl-metadata-incremental-sync:v0.3.0@sha256:f8c66baf14fff833d9c0de5f56b29aab6a6c26263b7fad6e29a3c5f154c3946d
```

## 验证结果

- 自动测试 42 项通过，覆盖批量写入、失败记录、重试、安全暂停、旧断点恢复。
- 独立 Kafka Topic、消费组和 231 独立测试库，真实插入 1000 条，失败 0；全部 41 字段及 1000 条 Strong 读回检查通过。
- 该轮从 Worker 启动到停止约 66.7 秒，约 15 条/秒；计时不含停止后的最终 flush 和完整读回验收。实际正式库吞吐仍取决于 Solr、向量服务和 Milvus 负载。
- 镜像运行代码与本地源码哈希一致，依赖、上海时区、Linux 单实例锁检查通过。
- 231 check 模式验证 41 字段、18 索引、原集合身份、Kafka、Solr、生产 MongoDB和向量接口；没有订阅 Kafka 或写入正式 Milvus。
- 只读复制真实 SQLite 到本地后，新版成功打开且保留 5767 条已完成记录：新增 2116、更新 3595、无需变化 56，待处理 0。位点范围 0–5766。
- 本次没有更新正式 Stack；原 v0.2.0 消费者仍健康运行。

验证报告保存在根目录 `out/metadata_incremental_release_v0.3.0/`；
真实性能验收报告为 `out/metadata_incremental_live_kafka_20261021_04/summary.json`。

## 在原 Stack 更新

1. 进入 Portainer **231 环境**，打开原 Stack `nstl-metadata-incremental-sync` 的 Editor。
2. 用本目录 `portainer-stack-231.yml` 完整替换内容，点击 **Update the stack**。
3. 仅点 Restart 不会加载新的 YAML；不要另建第二个正式消费者。
4. 更新时旧进程退出，新镜像启动；初次检查允许 90 秒，随后持续消费、空闲等待。
5. 不要删除三个已有卷，不要重置 Kafka 消费组位点。

Topic 和消费组保持不变：

- `VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS`
- `nstl_metadata_incremental_sync`

消息首先持久化 SQLite，再提交 Kafka 位点；升级后先恢复未完成消息，已完成的
`topic/partition/offset` 会跳过。相同文献 ID 在新的消息位点再次出现会重新查 Solr 并更新。
这不承诺故障窗口中外部写请求恰好只执行一次；发生结果不确定时仍有读回核验和安全暂停保护。
不要修改 Milvus URI 的字面形式，否则现有断点身份检查会拒绝继续。

## 批量参数和处理流程

使用通过真实测试的 `SYNC_BATCH_SIZE=1000`、`SYNC_BATCH_WAIT_SEC=0.25`：
最多聚合 1000 条消息、最多等待 0.25 秒，不必满批才处理。实际批次通常小于1000。
Solr 批查、向量批算、Milvus Strong 查询及写入各自分块，子请求最多200条，
写后核验每次最多50条；依然保留逐消息持久化和消费位点提交。

不再对每条 ID 重复执行整个远程请求链路。
已有文献使用 `partial_update=True` 更新查到的字段及重算向量，Solr缺少的字段保留旧值；
新文献 insert 完整41字段。没有删除记录、删除集合、重建集合或重建索引入口。

## 日志、状态和日报

- Portainer → Containers → `nstl-metadata-incremental-sync` → Logs，查看实时 `batch_done` 和 `daily_activity`。
- 文件日志 `/app/logs/incremental.log`，按天轮换，保留30份。
- 当前状态 `/app/state/progress.json`：状态、累计结果、Mongo待同步数和 `latest_batch`（最近一批速度和 Solr/向量/读写核验耗时）。空闲时最近批次速度是历史批次值，不代表仍在处理。
- 容器内部 `http://127.0.0.1:8099/status` 同样读取当前状态，不暴露服务器端口。
- 实际处理日期日报 `/app/state/daily_activity/YYYY-MM-DD.json`：上海日期、数量、首次处理、最后完成、有效处理秒数、跨度秒数。
- 消息收到日期报表 `/app/state/daily/YYYY-MM-DD.json`。
- 同一天多次投递会持续处理并累计日报；没有批次结束消息，所以日志用空闲窗口标识等待，日报不会将其解释为严格的上游批次结束。

Portainer容器Console可执行（仅查看，不修改）：

```sh
cat /app/state/progress.json
cat /app/state/daily_activity/2026-10-21.json
```

## 失败和持续运行

Solr缺源、单条转换异常、普通依赖重试耗尽会保存记录并继续处理后面的消息。
失败记录在本地 SQLite，并幂等同步生产 MongoDB：
`nstl_vector_sync_scientific_prod.metadata_incremental_sync_failures`，待处理筛选
`{status: "pending"}`。`/app/state/failures_recent.json` 是最近100条快照。
Mongo临时离线先保存在本地outbox，恢复后补同步。

集合身份、未提交历史字段、存储或结果不确定的严重异常会持久化安全暂停并保留进程；
重启不会绕过保护。`restart: unless-stopped` 只负责进程意外退出和主机重启恢复。
健康检查核对心跳和安全暂停状态；`idle` 表示正常等待，容器健康不代表当前一定有吞吐。

三个外部卷必须保留：`nstl-metadata-incremental-state`、
`nstl-metadata-incremental-logs`、`nstl-metadata-incremental-credentials`。
外部网络仍使用231已有 `milvus`。凭据仅从已有只读卷加载，没有写入镜像或Stack。

## 当前边界

当前 Stack 只支持231的一套目标 Milvus。三套独立生产 Milvus 的跨库原子写入尚未实现，
相关限制仍见 [DESIGN.md](DESIGN.md)。存量程序与本程序没有跨进程写锁，应避免同一ID同时写。
