"""Write a new, private Compose environment from the source MinIO container."""

from __future__ import annotations

import argparse
import json
import os
import ipaddress
import re
import subprocess
from pathlib import Path


KEYS = ("MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not ipaddress.ip_address(args.target_host).is_private:
        parser.error("--target-host must be a private IPv4 or IPv6 address")

    result = subprocess.run(
        ["docker", "inspect", "scientific-milvus-minio"],
        check=True, capture_output=True, text=True,
    )
    source = json.loads(result.stdout)[0]
    environment = dict(item.split("=", 1) for item in source["Config"]["Env"] if "=" in item)
    values = {key: environment[key] for key in KEYS}
    if not all(re.fullmatch(r"[A-Za-z0-9._@-]+", value) for value in values.values()):
        raise ValueError("MinIO credential contains characters requiring a different Compose encoding")

    lines = [
        "REPLICA_DATA_ROOT=/data/scientific-milvus-replica-231",
        f"TARGET_HOST={args.target_host}",
        *(f"{key}={values[key]}" for key in KEYS),
    ]
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"Created private environment for {args.target_host}: {args.output}")


if __name__ == "__main__":
    main()
