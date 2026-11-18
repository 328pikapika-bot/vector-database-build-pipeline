"""Read-only SSH observation; never stops or changes the remote runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import paramiko


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=int, default=3600)
    p.add_argument("--interval", type=int, default=30)
    args = p.parse_args()
    remote = "/data/scientific-fulltext-source-id-backfill-20261113"
    local = Path(__file__).resolve().parents[1] / "outputs/fulltext_source_id_backfill_20261113"
    local.mkdir(exist_ok=True, parents=True)
    client = None
    failures = 0
    try:
        while True:
            try:
                if client is None:
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    client.connect("192.0.2.15", username="root",
                                   password=os.environ["BACKFILL_SSH_PASSWORD"], timeout=15)
                    client.get_transport().set_keepalive(30)
                with client.open_sftp() as sftp:
                    with sftp.open(remote+"/status.json") as f:
                        status = json.loads(f.read().decode())
                    try:
                        with sftp.open("/proc/"+str(int(status["pid"]))+"/cmdline") as f:
                            status["process_alive"] = b"run_fulltext_source_id_backfill.py" in f.read()
                    except FileNotFoundError:
                        status["process_alive"] = False
                (local/"status_latest.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
                with (local/"observation.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(status, ensure_ascii=False)+"\n")
                fields = ("updated_at", "status", "pid", "runtime_seconds", "backfill_elapsed_seconds", "scanned_chunks",
                          "eligible_verified_per_host", "newly_filled_per_host", "no_source_chunks",
                          "completed_batches", "sustained_scanned_rows_per_second", "retry_events",
                          "integrity_failure_events", "current_run_retry_events", "error", "prior_records_checked_per_host")
                print(json.dumps({k: status[k] for k in fields+("process_alive",) if k in status}, ensure_ascii=False), flush=True)
                failures = 0
                if status["status"] == "paused_error":
                    raise SystemExit(2)
                if not status["process_alive"]:
                    raise SystemExit(3)
                if status.get("runtime_seconds", 0) >= args.seconds and status["status"] == "running":
                    (local/"one_hour_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
                    print("OBSERVATION_COMPLETE_REMOTE_RUNNER_NOT_STOPPED", flush=True)
                    return
            except (OSError, paramiko.SSHException, ValueError) as exc:
                failures += 1
                print(json.dumps({"monitor_connection_error": repr(exc), "attempt": failures}), flush=True)
                if client:
                    client.close()
                client = None
                if failures >= 6:
                    raise
            time.sleep(args.interval)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
