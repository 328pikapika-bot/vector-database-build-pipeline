# `nstl-metadata-incremental-sync` 实施计划

本项目独立于 `vector_sync_service/` 和 `solr_field_backfill/`。本轮默认只为 231 测试集合实现单目标写入；三套独立 Milvus 的跨库原子提交限制见 [DESIGN.md](DESIGN.md)。

## 阶段与验收

1. **字段与输入**：固定 41 字段 Schema 契约，移植 Solr 原 24 字段、16 个补字段及 `entityType` 的转换规则；Kafka 消息只接受文献 ID。对照已有程序的典型样本。
2. **独立 Worker**：连接既有 Kafka 和 Solr；生成三个 2560 维向量；Milvus Strong 检查，已有 ID 仅局部更新，新 ID 仅在确认缺失后插入。没有 delete/drop/truncate 调用。
3. **断点与运行状态**：固定消费组、手动 offset、持久化事件账本、失败 ID、重试、超时读回核对、每日 JSON 报告和容器日志。依赖故障时进程保持运行；状态存储和 Schema 安全异常停止写入。
4. **验证**：单元与模拟集成测试；独立小集合验证向量局部更新；231 只读预检查及限定样本前后审计。保护原集合历史记录与索引。
5. **容器交付**：独立 Dockerfile、`.dockerignore`、requirements、231 Portainer Stack、运行说明。镜像默认 `check`；正式启动须显式配置 `run`。

## 当前边界

- 现有 `entityType` 历史任务和 16 字段存量任务不因本项目构建而停止、重置或改动。
- 新 Topic 名称定为 `VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS`，沿用现有 Kafka。只有文献 ID 消息；一个分区可保证该 Topic 的消息顺序，首次部署时须记录起始 offset。带 `_V1`、`_V2` 后缀的旧 Topic 不作为正式投递目标，并已于 2026-10-20 按用户要求清理。
- 测试 Stack 仅连接 231 的 `default/scientific_documents_v2`。生产目标地址、基线复制和三目标一致性策略尚未启用。
