"""Start the metadata-only Kafka consumer."""
from app.cli.worker_runner import main


if __name__ == "__main__":
    main("metadata")
