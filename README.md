# 向量库构建流程

[English](README.en.md) | 简体中文

将文献题录、全文和图片等资源接入处理流水线，构建可追溯的向量索引，并处理增量更新、重试、背压和数据一致性。

## 主要功能

- 消费 Kafka 文献事件，生成文档级向量并按版本和标识更新。
- 全文通过异步队列执行下载、清洗、切块和向量写入；图片链路生成图像与文本特征。
- 保留元数据同步、Solr 字段回填、旧数据导入、状态监控及诊断工具。

## 源码与运行入口

`源码/vector_sync_service/` 主服务；`源码/metadata_incremental_sync/` 增量同步；`源码/solr_field_backfill/` 字段回填；`源码/milvus_import_old/` 历史导入工具；`源码/kafbat-ui/` Kafka UI 示例。

Python、Kafka、Milvus、MongoDB、Solr、对象存储，以及文本/视觉向量服务。各子项目依赖和配置分别管理，不需要一次运行全部子项目。

主服务入口 `源码/vector_sync_service/app/main.py`，配置入口 `源码/vector_sync_service/app/config.py`。先选择要运行的子项目，按其依赖和 `.env.example` 配置测试服务。

## 配置

Kafka 使用 `KAFKA_SASL_USERNAME`/`KAFKA_SASL_PASSWORD`；Milvus 使用 `MILVUS_USER`/`MILVUS_PASSWORD`；向量模型 Key 通过相应环境变量注入。示例地址只代表本机占位。

配置示例位于源码中的 `.env.example`。真实凭证只保存在本机环境或不跟踪的 `.env` 中。`127.0.0.1` 与 `example.invalid` 为本地/不可解析示例地址，部署前替换为自己的配置。

## 验证与边界

旧导入工具是历史兼容资产，不等于主服务当前默认流程。真实数据迁移、消费位点和写入行为未在本次执行；运行迁移或压力脚本前须自行配置隔离目标。

本目录于 2026-11-04 完成公开范围整理与静态核验。代码发布不等于已部署或全部功能可运行。配置与依赖准备好后，应再执行适用于目标环境的测试。

公开范围及 GitHub/Gitee 接手规则见[仓库发布说明](仓库发布说明.md)。资料库原有的文档、会话、经验和管理记录继续保留在本目录，只供本地知识库使用。
