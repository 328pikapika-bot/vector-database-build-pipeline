"""CLI: python -m incremental_sync [check|run|create-topic]."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from .config import DEFAULT_TOPIC, Settings
from .worker import IncrementalWorker, install_signals


def configure_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = TimedRotatingFileHandler(
        settings.log_dir / "incremental.log", when="midnight", backupCount=30, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def create_topic() -> None:
    from confluent_kafka.admin import AdminClient, NewTopic
    topic_name = os.getenv("KAFKA_INCREMENTAL_TOPIC", DEFAULT_TOPIC).strip()
    brokers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not topic_name or not brokers:
        raise ValueError("KAFKA_INCREMENTAL_TOPIC and KAFKA_BOOTSTRAP_SERVERS are required")
    protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    config: dict[str, str] = {"bootstrap.servers": brokers, "security.protocol": protocol}
    if protocol.startswith("SASL"):
        username = os.getenv("KAFKA_SASL_USERNAME", "")
        password = os.getenv("KAFKA_SASL_PASSWORD", "")
        if not username or not password:
            raise ValueError("Kafka SASL credentials are required")
        config.update({"sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
                       "sasl.username": username, "sasl.password": password})
    admin = AdminClient(config)
    metadata = admin.list_topics(timeout=10)
    existing = metadata.topics.get(topic_name)
    if existing and existing.error is None:
        if len(existing.partitions) != 1 or any(len(p.replicas) != 3 for p in existing.partitions.values()):
            raise ValueError("existing Topic is not one partition with three replicas")
        logging.info("topic_exists name=%s", topic_name)
        return
    future = admin.create_topics([NewTopic(topic_name, num_partitions=1, replication_factor=3)])[topic_name]
    future.result(timeout=30)
    logging.info("topic_created name=%s partitions=1 replication=3", topic_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("check", "run", "create-topic"), default=None)
    args = parser.parse_args()
    if args.command == "create-topic":
        logging.basicConfig(level=logging.INFO)
        create_topic()
        return 0
    settings = Settings.from_env()
    command = args.command or settings.mode
    configure_logging(settings)
    if command != settings.mode:
        logging.error("command %s differs from SYNC_MODE=%s", command, settings.mode)
        return 2
    worker = IncrementalWorker(settings)
    install_signals(worker)
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
