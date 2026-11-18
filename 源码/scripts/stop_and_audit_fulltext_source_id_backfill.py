"""Settle ONLY existing pending batches, then read-only audit every handled ID.

Requires the user stop marker. Never enumerates or prepares a new batch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from run_fulltext_source_id_backfill import Runner, decode, now, log
from benchmark_fulltext_source_id_100 import COLLECTION, FIELD, HOSTS, digest, plain, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    root = Path(args.workdir)
    if not (root / "STOP_REQUESTED").exists():
        raise RuntimeError("Missing explicit user stop marker")
    r = Runner(root)
    started = time.monotonic()
    report = {"started_at": now(), "scope": COLLECTION, "field": FIELD,
              "status": "settling_existing_pending_only", "settled_pending": []}
    def save():
        (root/"stop_consistency_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        r.retry(r.preflight, "stop_preflight")
        r.status("stopping_consistency_check")
        pending = r.db.execute("SELECT seq,payload FROM batches WHERE status='pending' ORDER BY seq").fetchall()
        for seq, blob in pending:
            result = r.retry(lambda: r.apply_and_verify(seq, decode(blob)), "settle_existing_pending")
            report["settled_pending"].append({"seq": seq, "hosts": result})
            save()
            log("stop_pending_settled", seq=seq, hosts=result)
        # All batch original fields/vectors have already been checked before and
        # after their writes. Recheck EVERY handled PK/document/source value now,
        # including skipped rows, with bounded lightweight queries.
        report["status"] = "read_only_all_handled_id_audit"
        report["checked_ids_per_host"] = 0
        save()
        records, batch_count = [], 0
        baselines = {}
        def audit_group():
            nonlocal records
            if not records:
                return
            ids = [x["id"] for x in records]
            expected = {x["id"]: x for x in records}
            def read(host):
                rows = r.clients[host].get(COLLECTION, ids=ids,
                    output_fields=["id", "doc_id", FIELD], consistency_level="Strong", timeout=90)
                mapped = {row["id"]: plain(dict(row)) for row in rows}
                if len(mapped) != len(ids) or set(mapped) != set(ids):
                    raise RuntimeError("Handled primary-key set differs on "+host)
                for i, row in mapped.items():
                    target = expected[i]
                    if row["doc_id"] != i.rsplit("_", 1)[0] or row.get(FIELD) != target[FIELD]:
                        raise RuntimeError("Handled document/source value mismatch on "+host+"/"+i)
                return digest(mapped)
            hashes = r.retry(lambda: r.parallel(read), "read_only_stop_audit")
            if len(set(hashes.values())) != 1:
                raise RuntimeError("Handled replica records differ")
            report["checked_ids_per_host"] += len(ids)
            report["last_group_hashes"] = hashes
            save()
            r.status("stopping_consistency_check", audited_ids_per_host=report["checked_ids_per_host"])
            log("stop_audit_progress", checked_ids_per_host=report["checked_ids_per_host"])
            records = []
        for seq, blob in r.db.execute("SELECT seq,payload FROM batches WHERE status='completed' ORDER BY seq"):
            payload = decode(blob)
            patches = {row["id"]: row[FIELD] for row in payload["patches"]}
            for i in payload["ids"]:
                values = [payload["original_values"][h][i] for h in HOSTS]
                if i not in patches and len({digest(v) for v in values}) != 1:
                    raise RuntimeError("Skipped row had differing original source values")
                records.append({"id": i, FIELD: patches.get(i, values[0])})
            batch_count += 1
            if len(records) >= 4000:
                audit_group()
        audit_group()
        r.retry(r.count_guard, "stop_count_guard")
        final_snapshots = r.retry(lambda: r.parallel(lambda h: snapshot(r.clients[h])), "stop_collection_snapshots")
        if final_snapshots != r.state("baseline"):
            raise RuntimeError("Collection snapshots differ from diagnosed baseline")
        pending_count = r.db.execute("SELECT COUNT(*) FROM batches WHERE status='pending'").fetchone()[0]
        if pending_count:
            raise RuntimeError("Pending batches remain")
        if report["checked_ids_per_host"] != r.state("scanned"):
            raise RuntimeError("Stop audit did not cover all handled primary keys")
        filled_counts = r.retry(lambda: r.parallel(lambda h: int(r.clients[h].query(COLLECTION,
            filter=FIELD+" is not null", output_fields=["count(*)"], consistency_level="Strong", timeout=90)[0]["count(*)"])),
            "stop_filled_counts")
        if len(set(filled_counts.values())) != 1 or filled_counts[HOSTS[0]] != r.state("eligible_verified"):
            raise RuntimeError("Final filled source-ID counts differ")
        report.update({"status": "stopped_and_verified", "finished_at": now(),
            "elapsed_seconds": round(time.monotonic()-started, 2), "completed_batches": batch_count,
            "pending_batches": 0, "all_handled_ids_match_across_hosts": True,
            "filled_counts_per_host": filled_counts, "newly_filled_per_host": r.state("newly_filled"),
            "collection_counts_per_host": {h:v[COLLECTION]["count"] for h,v in final_snapshots.items()},
            "all_collection_snapshots_match_baseline": True,
            "duplicate_keys_skipped": r.state("duplicate_keys_skipped",0),
            "no_source_chunks": r.state("no_source",0),
            "cursor": r.state("cursor"), "audit_process_pid": os.getpid()})
        r.put("stopped_at", now())
        r.put("stop_consistency_verified_at", now())
        r.put("stop_verified_ids_per_host", report["checked_ids_per_host"])
        r.db.commit()
        r.status("stopped_by_user", consistency_verified=True,
                 audited_ids_per_host=report["checked_ids_per_host"], pending_batches=0)
        save()
        log("stopped_and_verified", **report)
    except Exception as exc:
        report.update({"status": "stopped_audit_requires_attention", "error": repr(exc)})
        save()
        r.status("stopped_audit_requires_attention", error=repr(exc))
        raise
    finally:
        r.pool.shutdown(wait=True)
        for c in r.clients.values():
            c.close()
        r.db.close()


if __name__ == "__main__":
    main()
